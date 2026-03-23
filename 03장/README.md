# 🧠 KLUE YNAT 뉴스 분류 (ch03)

한국어 뉴스 제목을 7개의 카테고리로 분류하는 텍스트 분류 모델입니다.  
Hugging Face Transformers를 활용하여 `klue/roberta-base` 모델을 파인튜닝하였습니다.

---

## 📌 프로젝트 개요

- Task: 뉴스 카테고리 분류 (Text Classification)
- Model: klue/roberta-base
- Dataset: KLUE/YNAT
- Framework: Hugging Face Transformers

---

## 📂 데이터셋

KLUE/YNAT는 뉴스 제목을 기반으로 카테고리를 분류하는 데이터셋입니다.

### ✔ 카테고리 (7개)
- IT과학
- 경제
- 사회
- 생활문화
- 세계
- 스포츠
- 정치

### ✔ 주요 컬럼
- `title`: 뉴스 제목
- `label`: 카테고리

---

## ⚙️ 학습 설정

- Epoch: 1
- Batch Size: 8
- Learning Rate: 5e-5
- Trainer API 사용

---

## 📊 성능

- Validation Accuracy: 0.834  
- **Test Accuracy: 0.856**

짧은 학습(epoch 1)에도 불구하고 안정적인 성능을 보였습니다.

---

## 🚀 사용 방법

```python
from transformers import pipeline

model_id = "eunzzang/roberta-base-klue-ynat-classification"

classifier = pipeline("text-classification", model=model_id)

texts = [
    "삼성전자 주가 상승",
    "AI 기술 발전 가속화",
    "애플 신제품 출시",
    "환율 상승으로 수출 증가",
    "사회적 거리두기 완화"
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"{text} → {result['label']} ({result['score']:.2f})")

```