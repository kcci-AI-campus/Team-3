# Team-3

## 구현 목표
  1. On-Device 환경에서 안정적인 게임 구동
     - Raspberry Pi에서 별도의 PC나 서버 연결 없이 손동작 인식 및 게임 로직을 실시간으로 수행
     - 경량화된 모델을 적용하여 제한된 하드웨어 환경에서도 안정적인 동작 구현
  2. 기존 RPS 게임 대비 손동작 인식 성능 향상
     - 손동작 데이터를 기반으로 모델을 학습하여 Scissors, Rock, Paper 외 다양한 손동작을 인식
     - 실제 웹캠 환경을 고려한 데이터 다양화와 모델 학습을 통해 손동작 인식 정확도 향상
### 기대 효과 
  - Raspberry Pi 기반의 독립적인 On-Device AI 게임 구현
  - 실시간 손동작 인식을 통한 사용자 입력 편의성 및 게임 몰입도 향상
  - 데이터 증강 및 모델 최적화를 통한 다양한 환경에서의 손동작 인식 안정성 확보

---

## 게임 설명
- 카메라로 사용자의 손동작을 인식하여 제시된 손동작과 일치하는지 판단하는 게임
- 제시된 손동작을 기억하고 순서대로 입력하며, 성공 횟수에 따라 점수를 획득하는 방식
- 클래스 분류 : sicssors, rock, paper, spiderman, thumb, okey, circle

<div align="center">
  <img src="https://github.com/user-attachments/assets/222ec26a-8a69-48ef-9ac1-9d1155799717" height="150">
  <img src="https://github.com/user-attachments/assets/2d271dfb-916d-4840-9895-f64e88c10b01" height="150">
  <img src="https://github.com/user-attachments/assets/e97d7163-d86f-43a2-8869-90b99ec8b96b" height="150">
  <img src="https://github.com/user-attachments/assets/085fa243-04bc-4620-9fca-8ed6657a0a94" height="150">
  <img src="https://github.com/user-attachments/assets/9f110ea6-55d1-4b8d-affb-d95d5dee0af0" height="150">
  <img src="https://github.com/user-attachments/assets/3def095f-4e53-4219-b9c4-7719192b78c5" height="150">
  <img src="https://github.com/user-attachments/assets/58942c6d-4ddc-482b-9a61-f1b424367005" height="150">
</div>

---

## 학습 환경
## 📊 데이터셋 구성

### 1. 클래스별 데이터 수집 현황

| 구분     | Scissors |  Rock | Paper |  Okay | Circle | Thumb | Spiderman |    기타 |
| ------ | -------: | ----: | ----: | ----: | -----: | ----: | --------: | ----: |
| 제작     |        - |     - |     - |     - |      - |     - |         - |     - |
| 다운     |        - |     - |     - |     - |      - |     - |         - |     - |
| 추가     |        - |     - |     - |     - |      - |     - |         - |     - |
| **합계** |    **-** | **-** | **-** | **-** |  **-** | **-** |     **-** | **-** |

### 2. 학습 데이터 분할 현황

| 클래스       | Train |   Val |  Test |
| --------- | ----: | ----: | ----: |
| Scissors  |     - |     - |     - |
| Rock      |     - |     - |     - |
| Paper     |     - |     - |     - |
| Okay      |     - |     - |     - |
| Circle    |     - |     - |     - |
| Thumb     |     - |     - |     - |
| Spiderman |     - |     - |     - |
| **합계**    | **-** | **-** | **-** |
