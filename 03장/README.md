---

````markdown
# 🧠 KLUE YNAT 뉴스 분류 모델

한국어 뉴스 제목을 7개의 카테고리로 분류하는 모델입니다.  
Hugging Face Transformers를 활용하여 `klue/roberta-base` 모델을 파인튜닝하였습니다.

---

## 📌 모델 정보

- Base Model: klue/roberta-base
- Task: Text Classification (뉴스 카테고리 분류)
- Dataset: KLUE/YNAT
- Labels (7개):
  - IT/과학
  - 경제
  - 사회
  - 생활/문화
  - 세계
  - 스포츠
  - 정치

---

## ⚙️ 학습 설정

- Epoch: 1
- Batch Size: 8
- Learning Rate: 5e-5

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
    "손흥민 경기 활약"
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"{text} → {result['label']} ({result['score']:.2f})")
````

---

## 🔗 모델 링크

👉 [https://huggingface.co/eunzzang/roberta-base-klue-ynat-classification](https://huggingface.co/eunzzang/roberta-base-klue-ynat-classification)

---

## 📝 설명

본 모델은 KLUE/YNAT 데이터셋을 활용하여 뉴스 제목을 자동으로 분류하도록 학습되었습니다.
한국어 자연어 처리 기반 분류 작업에 활용할 수 있습니다.

```

---