# Team-3
## 목차 
  ### 1.구현 목표
  ### 2.학습 환경
  ### 3.결과 예시
  ### 4.개선 요소
---

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

### 게임 방법
  - 무작위 동작 하나를 3초 동안 제시
  - 사용자는 동작들을 처음부터 순서대로 입력
  - 4초 카운트다운이 끝날 때의 인식 결과로 판정
  - 라운드의 모든 동작을 맞히면 점수가 1 올라가고 새 동작이 추가
  
  - SPACE 또는 START 버튼으로 시작
  - 틀린 동작을 입력하거나 동작별 제한시간 20초가 지나면 종료



---

## 학습 환경
### 📊 데이터셋 구성

  1. 클래스별 데이터 수집 현황

| 구분     | Scissors |  Rock | Paper |  Okay | Circle | Thumb | Spiderman |   
| ------ | -------: | -----: | ----: | ----: | -----: | ----: | --------: | 
| 제작     |        - |     - |     - |  140  |   100  |  100  |       140 |  
| 다운     |     137 |    191 |   83  |   34  |   119  |   114 |         40|    
| 추가     |        - |     - |     - |     - |      - |     - |         - |   
| **합계** |    **-** | **-** | **-** | **-** |  **-** | **-** |     **-** | 

  2. 학습 데이터 분할 현황

| 구분     | Scissors |  Rock | Paper |  Okay | Circle | Thumb | Spiderman |   
| ------- | -------: | ----: | ----: | ----: | -----: | ----: | --------: | 
| train    |        - |     - |     - |     - |      - |     - |         - |  
| val      |        - |     - |     - |     - |      - |     - |         - |    
| test     |        - |     - |     - |     - |      - |     - |         - |   
| **합계** |    **-** | **-** | **-** | **-** |  **-** | **-** |     **-** | 

### ⚙️ 모델 및 실행 환경

| **항목**   | **내용**                             |
| -------- | ---------------------------------- |
| 모델       | YOLO11n (Ultralytics)              |
| 학습 입력 크기 | 640 × 640 (`TRAIN_SIZE`) |
| 변환 입력 크기 | 640 × 640 / 320 × 320              |
| 양자화       | INT8                |
| 학습 환경    | Google Colab (**T4 GPU**)          |
| 실행 환경    | Raspberry Pi (**LiteRT**)          |

### ⚙️ 모델 선정 이유


---
## 결과 예시 






---
## 개선 요소
