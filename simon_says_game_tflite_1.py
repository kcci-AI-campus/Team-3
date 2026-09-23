"""
설치: python -m pip install opencv-python ai-edge-litert==2.2.0
실행: python simon_says_game_tflite.py

SPACE: 시작 / 재시작
화면의 START / RESTART 버튼도 사용할 수 있습니다.
Q 또는 ESC: 종료
같은 폴더의 best (1).tflite를 사용합니다. Raspberry Pi OS 64비트 기준입니다.
"""

import random
import time
from pathlib import Path


# ============================================================
# 1. 게임 기본 설정
# ============================================================

GESTURES = [
    'spiderman',
    'thumb',
    'okay',
    'circle',
    'rock',
    'scissors',
    'paper'
]

CONF_THRESHOLD = 0.70     # 인식 신뢰도
GESTURE_TIMEOUT = 20.0    # 준비 + 4초 유지를 포함한 입력당 제한시간
HOLD_TIME = 4.0           # 손동작을 유지할 시간
NEXT_INPUT_TIME = 1.5     # 정답 후 다음 동작 안내 시간
SHOW_TIME = 3.0           # 새 동작을 보여줄 시간
MOVE_LIMIT = 0.05        # 손 위치의 허용 변화량 (화면 크기의 5%)

CAMERA_INDEX = 0
YOLO_READY = True        # 제공된 TFLite 모델 사용
MODEL_PATH = Path(__file__).with_name('best (1).tflite')
DETECT_THRESHOLD = 0.25  # 검출 후보 기준. 게임 입력 기준 0.70은 그대로 유지
MODEL_CLASS_NAMES = ['scissors', 'rock', 'paper', 'thumb', 'spiderman', 'okay', 'circle']

# LiteRT 모델은 처음 한 번만 준비한다.
tflite_interpreter = None
tflite_input = None
tflite_output = None

# 클래스별 안내 이미지 폴더
IMAGE_DIR = Path(__file__).with_name('images')

# 7개 클래스와 이미지 파일을 1:1로 연결
GESTURE_IMAGE_PATHS = {
    'spiderman': IMAGE_DIR / 'spiderman.png',
    'thumb': IMAGE_DIR / 'thumb.jpg',
    'okay': IMAGE_DIR / 'okay.jpg',
    'circle': IMAGE_DIR / 'circle.jpg',
    'rock': IMAGE_DIR / 'rock.jpg',
    'scissors': IMAGE_DIR / 'scissors.jpg',
    'paper': IMAGE_DIR / 'paper.jpg',
}

BEST_SCORE_FILE = Path(__file__).with_name('best_score.txt')
BUTTON_AREA = (250, 430, 550, 495)  # 왼쪽, 위쪽, 오른쪽, 아래쪽


def load_gesture_images(cv2):
    """7개 클래스의 안내 이미지를 프로그램 시작 시 한 번만 읽습니다."""
    images = {}

    for gesture in GESTURES:
        path = GESTURE_IMAGE_PATHS[gesture]

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            print(f'[경고] 이미지를 읽을 수 없습니다: {path}')
            images[gesture] = None
        else:
            images[gesture] = image

    return images


def load_best_score():
    # 저장 파일이 없거나 손상되었으면 0으로 시작합니다.
    try:
        return max(0, int(BEST_SCORE_FILE.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        return 0


# ============================================================
# 2. Simon Says 게임
# ============================================================

class SimonSaysGame:

    def __init__(self, gesture_images=None):
        self.gesture_images = gesture_images or {}
        self.sequence = []
        self.round = 0
        self.score = 0
        self.best_score = load_best_score()
        self.input_index = 0
        self.restart_requested = False
        self.detected_hands = []        # 현재 프레임의 손 박스 표시용

        self.state = 'READY'
        self.message = 'Press SPACE to start'

        self.next_input_time = 0
        self.show_end_time = 0
        self.input_start_time = 0
        self.last_frame_time = None


        self.reset_hold()

    # --------------------------------------------------------
    # 게임 시작
    # --------------------------------------------------------

    def start_game(self):
        self.sequence = []
        self.round = 0
        self.score = 0
        self.input_index = 0
        self.detected_hands = []

        self.last_frame_time = None

        self.next_round(time.monotonic())

    # --------------------------------------------------------
    # 다음 라운드: 새 동작 하나만 보여준다.
    # --------------------------------------------------------

    def next_round(self, now):
        self.round += 1
        self.sequence.append(random.choice(GESTURES))

        self.input_index = 0
        self.state = 'SHOW_SEQUENCE'
        self.show_end_time = now + SHOW_TIME
        self.reset_hold()

        self.message = 'NEW: ' + self.sequence[-1]
        print(f'Round {self.round} / {self.message}')

    # --------------------------------------------------------
    # 누적 순서 입력 시작
    # --------------------------------------------------------

    def start_input(self, now):
        self.state = 'WAIT_INPUT'
        self.input_index = 0           # 항상 누적 순서의 첫 동작부터 입력
        self.reset_hold()
        self.input_start_time = now


    # --------------------------------------------------------
    # 4초 유지 카운트 초기화
    # --------------------------------------------------------

    def reset_hold(self):
        self.hold_gesture = None
        self.hold_box = None
        self.hold_start_time = None

    # --------------------------------------------------------
    # 손 개수와 신뢰도 확인
    # --------------------------------------------------------

    def check_hand(self, hands):
        if len(hands) == 0:
            self.message = 'Show one hand'
            return None

        if len(hands) >= 2:
            self.message = 'Only ONE hand, please'
            return None

        hand = hands[0]

        if not (CONF_THRESHOLD <= hand['confidence'] <= 1.0):
            self.message = 'Low confidence: show the gesture clearly'
            return None

        if hand['gesture'] not in GESTURES:
            self.message = 'Gesture not supported'
            return None

        # box는 (왼쪽, 위쪽, 오른쪽, 아래쪽) 정규화 좌표
        left, top, right, bottom = hand['box']
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            self.message = 'Invalid hand position'
            return None

        return hand

    # --------------------------------------------------------
    # 같은 동작과 손 위치를 4초 유지했는지 확인
    # --------------------------------------------------------

    def check_hold(self, hand, now):
        gesture = hand['gesture']
        box = tuple(hand['box'])

        # 손 위치는 유지 시작 시점의 위치와 비교한다.
        moved = False
        if self.hold_box is not None:
            for old_position, new_position in zip(self.hold_box, box):
                if abs(old_position - new_position) > MOVE_LIMIT:
                    moved = True
                    break

        changed = gesture != self.hold_gesture

        if self.hold_start_time is None or changed or moved:
            self.hold_gesture = gesture
            self.hold_box = box
            self.hold_start_time = now

        remaining = max(0, self.hold_start_time + HOLD_TIME - now)
        self.message = f'Hold still: {remaining:.1f} s'

        if now >= self.hold_start_time + HOLD_TIME:
            self.process_gesture(gesture, now)

    # --------------------------------------------------------
    # 확정된 동작을 정답과 비교
    # --------------------------------------------------------

    def process_gesture(self, gesture, now):
        if self.state != 'WAIT_INPUT' or self.check_timeout(now):
            return

        expected = self.sequence[self.input_index]

        if gesture != expected:
            self.game_over(f'Wrong! Expected {expected}, received {gesture}')
            return

        self.input_index += 1
        # 다음 입력은 다시 4초 유지해야 한다. 손을 내릴 필요는 없다.
        self.reset_hold()

        if self.input_index == len(self.sequence):
            self.score += 1             # 라운드를 모두 맞히면 성공 횟수 +1
            self.update_best_score()
            self.next_round(now)
        else:
            self.state = 'NEXT_INPUT'
            self.next_input_time = now + NEXT_INPUT_TIME
            self.message = 'Correct! Show the next gesture'

    # --------------------------------------------------------
    # 제한시간 확인
    # --------------------------------------------------------

    def check_timeout(self, now):
        if self.state != 'WAIT_INPUT':
            return False

        if now >= self.input_start_time + GESTURE_TIMEOUT:
            self.game_over('Time is up!')
            return True

        return False

    # --------------------------------------------------------
    # 매 프레임의 처리 순서
    # --------------------------------------------------------

    def update(self, hands, now):
        self.detected_hands = hands

        if self.state not in ('SHOW_SEQUENCE', 'WAIT_INPUT', 'NEXT_INPUT'):
            return

        # 1. 시간이 끝났으면 게임 종료
        if self.check_timeout(now):
            return

        # 영상 처리가 오래 멈췄다면, 그 시간을 손 유지 시간으로 세지 않는다.
        if self.last_frame_time is not None:
            if now - self.last_frame_time > 0.75:
                self.reset_hold()
        self.last_frame_time = now

        # 2. 문제를 보여주는 시간이 끝나면 입력 시작
        if self.state == 'SHOW_SEQUENCE':
            if now < self.show_end_time:
                return
            self.start_input(now)

        # 정답 안내 중에는 입력을 받지 않는다. 안내 후 제한시간을 새로 시작한다.
        if self.state == 'NEXT_INPUT':
            if now < self.next_input_time:
                return
            self.state = 'WAIT_INPUT'
            self.input_start_time = now
            self.reset_hold()

        # 3. 손이 하나이며, 인식 결과가 충분히 확실한지 확인
        hand = self.check_hand(hands)
        if hand is None:
            self.reset_hold()
            return

        # 4. 4초 유지되면 정답 확인
        self.check_hold(hand, now)

    # --------------------------------------------------------
    # 게임 오버: 프로그램을 종료하지 않고 결과 화면에서 기다린다.
    # --------------------------------------------------------

    def game_over(self, reason):
        self.state = 'GAME_OVER'
        self.message = reason
        self.reset_hold()

        print('\n========== GAME OVER ==========')
        print(reason)
        print('성공 횟수:', self.score)
        print('최고 기록:', self.best_score)

    def update_best_score(self):
        if self.score <= self.best_score:
            return

        self.best_score = self.score

        try:
            BEST_SCORE_FILE.write_text(str(self.best_score), encoding='utf-8')
        except OSError as error:
            # 파일 저장 실패 때문에 게임이 꺼지지 않도록 한다.
            print('최고 기록을 파일에 저장하지 못했습니다:', error)

    def show_error(self, message):
        # 장치 오류도 창을 닫지 않고 RETRY 버튼으로 다시 시도한다.
        self.state = 'ERROR'
        self.message = message
        self.reset_hold()


# ============================================================
# 3. YOLO 연결 자리
# ============================================================

def load_tflite_model():
    """제공된 NCHW / float32 모델을 처음 한 번만 불러온다."""
    global tflite_interpreter, tflite_input, tflite_output

    if tflite_interpreter is not None:
        return

    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'모델 파일을 같은 폴더에 넣으세요: {MODEL_PATH}')

    print('TFLite 모델 준비 중...')
    interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=4)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    # 다른 형식의 모델을 잘못 연결하는 것을 막는다.
    if tuple(input_info['shape']) != (1, 3, 640, 640):
        raise ValueError('이 코드는 입력 크기 (1, 3, 640, 640)인 제공 모델용입니다.')
    if tuple(output_info['shape']) != (1, 11, 8400):
        raise ValueError('이 코드는 출력 크기 (1, 11, 8400)인 제공 모델용입니다.')
    if input_info['dtype'] != np.float32 or output_info['dtype'] != np.float32:
        raise ValueError('제공된 float32 모델을 사용하세요.')

    tflite_input = input_info
    tflite_output = output_info
    tflite_interpreter = interpreter


def prepare_yolo_image(frame):
    """가로세로 비율을 유지한 채 640x640 RGB 입력을 만든다."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    ratio = min(640 / width, 640 / height)
    new_width = round(width * ratio)
    new_height = round(height * ratio)
    pad_left = (640 - new_width) // 2
    pad_top = (640 - new_height) // 2

    resized = cv2.resize(frame, (new_width, new_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = np.full((640, 640, 3), 114, dtype=np.uint8)
    image[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = rgb

    # 모델 입력: RGB, 0~1, (배치, 채널, 높이, 너비)
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...])
    return image, ratio, pad_left, pad_top


def read_yolo_hands(output, frame_shape, ratio, pad_left, pad_top):
    """모델 출력을 기존 게임의 gesture/confidence/box 목록으로 변환한다."""
    import cv2
    import numpy as np

    height, width = frame_shape[:2]

    # 각 후보: 중심 x, 중심 y, 너비, 높이, 7개 클래스 점수
    predictions = output[0].T
    class_ids = predictions[:, 4:].argmax(axis=1)
    scores = predictions[:, 4:].max(axis=1)
    valid = np.isfinite(predictions).all(axis=1)
    valid = valid & (scores >= DETECT_THRESHOLD) & (scores <= 1.0)

    boxes = []
    confidences = []
    gestures = []

    for candidate, class_id, confidence in zip(predictions[valid], class_ids[valid], scores[valid]):
        # 이 Ultralytics LiteRT 모델의 좌표는 640 입력에 대해 정규화되어 있다.
        center_x, center_y, box_width, box_height = candidate[:4] * 640

        # 전처리할 때 넣은 여백을 빼고 원래 웹캠 좌표로 되돌린다.
        left = (center_x - box_width / 2 - pad_left) / ratio
        top = (center_y - box_height / 2 - pad_top) / ratio
        right = (center_x + box_width / 2 - pad_left) / ratio
        bottom = (center_y + box_height / 2 - pad_top) / ratio

        left = max(0.0, min(float(left), width))
        top = max(0.0, min(float(top), height))
        right = max(0.0, min(float(right), width))
        bottom = max(0.0, min(float(bottom), height))

        if right <= left or bottom <= top:
            continue

        boxes.append([left, top, right - left, bottom - top])
        confidences.append(float(confidence))
        gestures.append(MODEL_CLASS_NAMES[int(class_id)])

    if not boxes:
        return []

    # 같은 손의 겹친 박스를 제거한다. 서로 떨어진 여러 손은 모두 남긴다.
    selected = cv2.dnn.NMSBoxes(boxes, confidences, DETECT_THRESHOLD, 0.5)
    hands = []

    for index in np.asarray(selected).reshape(-1):
        index = int(index)
        left, top, box_width, box_height = boxes[index]
        hands.append({
            'gesture': gestures[index],
            'confidence': confidences[index],
            'box': (
                left / width,
                top / height,
                (left + box_width) / width,
                (top + box_height) / height
            )
        })

    return hands


def get_hands_from_yolo(frame):
    """기존 main()에서 호출하는 함수. 반환 형식은 기존 코드와 동일하다."""
    load_tflite_model()
    image, ratio, pad_left, pad_top = prepare_yolo_image(frame)

    tflite_interpreter.set_tensor(tflite_input['index'], image)
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(tflite_output['index'])

    return read_yolo_hands(output, frame.shape, ratio, pad_left, pad_top)


# ============================================================
# 4. 웹캠 실행
# ============================================================

def main():
    try:
        import cv2
        import numpy as np
    except ImportError:
        print('먼저 설치하세요: python -m pip install opencv-python')
        return

    camera = None

    # 7개 클래스 이미지를 프로그램 시작 시 한 번만 로드
    gesture_images = load_gesture_images(cv2)

    game = SimonSaysGame(gesture_images)
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return

        left, top, right, bottom = BUTTON_AREA
        if left <= x <= right and top <= y <= bottom:
            if game.state in ('READY', 'GAME_OVER', 'ERROR'):
                game.restart_requested = True

    try:
        cv2.namedWindow('Simon Says')
        cv2.setMouseCallback('Simon Says', on_mouse)

        # 첫 모델 준비 시간을 게임 제한시간에 포함하지 않는다.
        if YOLO_READY:
            try:
                get_hands_from_yolo(frame)
                print('TFLite 준비 완료. START 버튼으로 시작하세요.')
            except Exception as error:
                print('TFLite 초기화 오류:', error)
                game.show_error('Model load failed. Check terminal and press RETRY.')

        while True:
            # 결과/오류 화면에서는 마지막 영상을 유지하고 버튼 입력만 기다린다.
            if game.state not in ('GAME_OVER', 'ERROR'):
                try:
                    if camera is None:
                        camera = cv2.VideoCapture(CAMERA_INDEX)

                    if not camera.isOpened():
                        raise OSError('웹캠을 열 수 없습니다. 연결을 확인하세요.')

                    success, new_frame = camera.read()
                    if not success or new_frame is None or new_frame.size == 0:
                        raise OSError('웹캠 영상을 읽지 못했습니다.')

                    frame = new_frame

                except (cv2.error, OSError) as error:
                    print('웹캠 오류:', error)
                    game.show_error('Camera error. Check connection and press RETRY.')

            # 모델이 없어도 게임과 제한시간은 진행한다.
            if game.state in ('SHOW_SEQUENCE', 'WAIT_INPUT', 'NEXT_INPUT'):
                hands = []

                # 나중에 연결할 외부 모델의 오류를 처리한다.
                if YOLO_READY:
                    try:
                        hands = get_hands_from_yolo(frame)
                    except Exception as error:
                        print('YOLO 추론 오류:', error)
                        game.show_error('YOLO error. Check model and press RETRY.')

                # 잘못된 모델 반환 형식은 조용히 무시하지 않고 안내한다.
                if game.state != 'ERROR':
                    try:
                        if not isinstance(hands, list):
                            raise TypeError('YOLO 결과는 손 검출 목록(list)이어야 합니다.')
                        game.update(hands, time.monotonic())
                    except (KeyError, TypeError, ValueError) as error:
                        print('YOLO 결과 형식을 확인하세요:', error)
                        game.show_error('Invalid YOLO result. Check terminal and press RETRY.')

            show_screen(cv2, frame, game)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

            if cv2.getWindowProperty('Simon Says', cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord(' '):
                game.restart_requested = True

            if game.restart_requested:
                game.restart_requested = False

                # 오류 재시도 시 카메라를 다시 연결한다.
                if game.state == 'ERROR':
                    if camera is not None:
                        camera.release()
                    camera = None
                    game.state = 'READY'

                if game.state in ('READY', 'GAME_OVER'):
                    game.start_game()

    except KeyboardInterrupt:
        print('\n게임을 종료합니다.')

    except (cv2.error, OSError) as error:
        print('웹캠 또는 화면 오류:', error)

    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


def draw_center_text(cv2, frame, text, y, size=1.0, color=(255, 255, 255)):
    # 글자의 실제 너비를 구해서 화면 가운데에 배치한다.
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, size, 2)

    # 긴 안내 문구도 화면 밖으로 잘리지 않도록 크기를 조절한다.
    if width > 680:
        size = size * 680 / width
        (width, height), baseline = cv2.getTextSize(text, font, size, 2)

    x = (800 - width) // 2
    cv2.putText(frame, text, (x, y), font, size, color, 2, cv2.LINE_AA)


def draw_hand_boxes(cv2, frame, hands):
    # YOLO 좌표는 원본 기준, 화면은 좌우 반전된 800x600 영상이다.
    for hand in hands:
        try:
            left, top, right, bottom = hand['box']
            confidence = float(hand['confidence'])
            gesture = str(hand['gesture'])

            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                continue
            if not (0 <= confidence <= 1):
                continue

            x1 = int((1 - right) * 799)
            x2 = int((1 - left) * 799)
            y1 = int(top * 599)
            y2 = int(bottom * 599)

        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        color = (80, 220, 100)           # 초록: 유효한 손 검출
        if confidence < CONF_THRESHOLD or gesture not in GESTURES:
            color = (0, 210, 255)       # 노랑: 불확실한 검출
        if len(hands) >= 2:
            color = (60, 60, 255)       # 빨강: 손 두 개 이상

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f'{gesture[:16]} {confidence:.0%}'
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_x = max(5, min(x1, 790 - width))
        label_y = max(25, y1 - 8)

        # 검정 테두리를 넣어 밝은 영상에서도 이름이 보이게 한다.
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_gesture_image(cv2, frame, image, gesture):
    """랜덤으로 선택된 클래스의 안내 이미지를 화면 중앙에 표시합니다."""
    if image is None:
        draw_center_text(
            cv2,
            frame,
            f'IMAGE NOT FOUND: {gesture}',
            220,
            0.75,
            (80, 100, 255)
        )
        return

    max_width = 200
    max_height = 150

    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        return

    scale = min(
        max_width / width,
        max_height / height
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    x = 15
    y = 5

    # 이미지가 화면을 벗어나지 않도록 확인
    if (
        x < 0 or
        y < 0 or
        x + new_width > frame.shape[1] or
        y + new_height > frame.shape[0]
    ):
        return

    frame[
        y:y + new_height,
        x:x + new_width
    ] = resized


def show_screen(cv2, frame, game):
    # OpenCV 기본 폰트는 한글을 지원하지 않으므로 화면 안내는 영어 사용
    frame = cv2.resize(cv2.flip(frame, 1), (800, 600))
    if game.state in ('SHOW_SEQUENCE', 'WAIT_INPUT', 'NEXT_INPUT'):
        draw_hand_boxes(cv2, frame, game.detected_hands)

    # 진행 중에는 상단에만 안내를 표시해 중앙의 손 영상을 가리지 않는다.
    header_height = 170 if game.state in ('SHOW_SEQUENCE', 'WAIT_INPUT', 'NEXT_INPUT') else 95
    cv2.rectangle(frame, (0, 0), (800, header_height), (20, 20, 20), -1)
    draw_center_text(cv2, frame,
                     f'Round: {game.round} / Score: {game.score} / Best: {game.best_score}',
                     35, 0.75)

    if game.state == 'WAIT_INPUT':
        elapsed = time.monotonic() - game.input_start_time
        remaining = max(0, GESTURE_TIMEOUT - elapsed)
        draw_center_text(cv2, frame,
                         f'Input: {game.input_index + 1}/{len(game.sequence)} / Time for this gesture: {remaining:.1f}s',
                         75, 0.7, (100, 220, 255))

    # 중앙 안내판은 시작 대기와 게임 오버/오류 화면에서만 사용한다.
    if game.state in ('READY', 'GAME_OVER', 'ERROR'):
        panel = frame.copy()
        cv2.rectangle(panel, (30, 205), (770, 415), (15, 15, 15), -1)
        frame = cv2.addWeighted(panel, 0.75, frame, 0.25, 0)

    if game.state == 'SHOW_SEQUENCE':
        gesture = game.sequence[-1]

        draw_center_text(
            cv2,
            frame,
            'NEW: ' + gesture.upper(),
            105,
            1.3,
            (100, 240, 160)
        )

        # 랜덤으로 선택된 클래스와 대응하는 이미지를 함께 표시
        draw_gesture_image(
            cv2,
            frame,
            game.gesture_images.get(gesture),
            gesture
        )

        remaining = max(
            0,
            game.show_end_time - time.monotonic()
        )

        draw_center_text(
            cv2,
            frame,
            f'Remember it! Start in {remaining:.1f} s',
            555,
            0.75
        )

    elif game.state == 'WAIT_INPUT':
        draw_center_text(cv2, frame, game.message, 120, 1.0, (100, 220, 255))


    elif game.state == 'NEXT_INPUT':
        panel = frame.copy()
        cv2.rectangle(panel, (50, 230), (750, 355), (15, 15, 15), -1)
        frame = cv2.addWeighted(panel, 0.65, frame, 0.35, 0)
        draw_center_text(cv2, frame, 'Correct!', 275, 1.2, (100, 240, 160))
        draw_center_text(cv2, frame, 'Show the next gesture', 325, 1.0)

    elif game.state in ('GAME_OVER', 'ERROR'):
        title = 'GAME OVER' if game.state == 'GAME_OVER' else 'DEVICE / MODEL ERROR'
        draw_center_text(cv2, frame, title, 250, 1.2, (80, 100, 255))
        draw_center_text(cv2, frame, game.message, 295, 0.65)
        draw_center_text(cv2, frame, f'Score: {game.score}', 345, 0.9)
        draw_center_text(cv2, frame, f'Best: {game.best_score}', 395, 0.9, (100, 230, 150))

    else:
        draw_center_text(cv2, frame, 'SIMON SAYS', 280, 1.4)
        draw_center_text(cv2, frame, 'Click START to play', 350, 0.95)

    if game.state in ('READY', 'GAME_OVER', 'ERROR'):
        labels = {'READY': 'START', 'GAME_OVER': 'RESTART', 'ERROR': 'RETRY'}
        left, top, right, bottom = BUTTON_AREA
        cv2.rectangle(frame, (left, top), (right, bottom), (80, 150, 60), -1)
        draw_center_text(cv2, frame, labels[game.state], top + 43, 0.9)

    cv2.rectangle(frame, (0, 530), (800, 600), (20, 20, 20), -1)
    draw_center_text(cv2, frame, 'Click button or SPACE: start/retry / Q or ESC: quit', 555, 0.6)
    if not YOLO_READY:
        draw_center_text(cv2, frame, 'YOLO not connected: gesture input is unavailable', 585, 0.55)

    cv2.imshow('Simon Says', frame)


if __name__ == '__main__':
    main()