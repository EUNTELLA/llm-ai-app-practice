
---

# 📌 QLoRA 기반 Text2SQL 미세조정 실험

## 🧠 프로젝트 소개

본 프로젝트는 자연어 질문을 SQL 쿼리로 변환하는 **Text2SQL 시스템**을 구축하고,
소형 언어 모델(sLLM)인 **beomi/Yi-Ko-6B**를 기반으로
**QLoRA (4-bit Quantized LoRA)** 기법을 활용하여 미세조정을 수행한 실험이다.

Google Colab의 제한된 GPU 환경에서도 효율적인 학습이 가능하도록 설계하였다.

---

## 🎯 프로젝트 목표

* 자연어 → SQL 변환 모델 구현
* ko_text2sql 데이터셋 활용
* 도메인별 SQL 데이터 직접 구축 (21개 QA 쌍)
* Baseline vs Fine-tuning 성능 비교
* QLoRA 기반 효율적인 미세조정 실험

---

## 🏗 프로젝트 구조

```text
.
├── data/
│   └── train.csv
├── results/
│   ├── baseline_results.csv
│   ├── finetuned_results.csv
│   ├── error_analysis.csv
│   └── performance_analysis.png
├── yi-ko-6b-text2sql/
└── README.md
```

---

## 📚 데이터셋 구성

### 1️⃣ 공개 데이터

* `shangrilar/ko_text2sql` (train/test split)

### 2️⃣ 커스텀 도메인 데이터 (직접 생성)

* 쇼핑몰 도메인 기반 SQL QA 21개 구성

---

### 📌 대표 예시

**질문**

> 고객 이름과 주문한 상품명을 같이 보여줘

**SQL**

```sql
SELECT c.name AS customer, p.name AS product
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id;
```

---

## 🧪 모델 구성

* **Base Model**: `beomi/Yi-Ko-6B`
* **Fine-tuning 방식**: QLoRA (4-bit LoRA)
* **Quantization**: NF4 (4-bit)
* **Framework**

  * HuggingFace Transformers
  * PEFT (LoRA)
  * BitsAndBytes
  * AutoTrain Advanced

---

## ⚙️ 학습 설정

| 항목            | 값    |
| ------------- | ---- |
| LoRA r        | 4    |
| LoRA alpha    | 8    |
| dropout       | 0.05 |
| batch size    | 1    |
| epochs        | 1    |
| learning rate | 2e-4 |
| quantization  | int4 |

---

## 📊 평가 방법

### ✔ Metric: Exact Match (EM)

```python
EM = 1 if predicted_sql == gold_sql else 0
```

* 예측 SQL과 정답 SQL이 완전히 일치하면 1점

---

## 📈 실험 결과

| 구분                  | EM                  |
| ------------------- | ------------------- |
| Baseline (Yi-Ko-6B) | `{baseline_em:.4f}` |
| QLoRA Fine-tuned    | `{ft_em:.4f}`       |

---

### 📌 성능 변화

* 향상폭: **`{ft_em - baseline_em:+.4f}`**

---

## 🔍 오답 분석

### 주요 오류 유형

* JOIN 구조 누락
* GROUP BY 미생성
* HAVING 절 누락
* 서브쿼리 생성 실패
* 컬럼 / 테이블명 오류

---

### 원인 분석

* 복합 SQL 구조 학습 데이터 부족
* JOIN + GROUP BY + SUBQUERY 조합 일반화 부족
* 도메인 다양성 부족으로 인한 overfitting 경향

---

## 📌 핵심 결과

* QLoRA를 통해 저비용 환경에서도 성능 향상 가능
* 단순 SQL 생성 능력은 명확히 개선됨
* 복잡한 SQL 구조에서는 여전히 한계 존재

---

## 🚀 결론

본 실험을 통해 QLoRA 기반 미세조정이
소형 언어 모델의 Text2SQL 성능을 효과적으로 향상시킬 수 있음을 확인하였다.

다만 복잡한 SQL 구조에 대한 일반화 성능 개선은 향후 연구 과제로 남는다.

---

## 🧰 사용 기술 스택

* Python
* PyTorch
* HuggingFace Transformers
* PEFT (LoRA)
* BitsAndBytes (4-bit quantization)
* AutoTrain Advanced
* Google Colab

---

## 👨‍💻 프로젝트 정보

* 과목: 인공지능과 언어
* 주제: Text2SQL + QLoRA 미세조정
* 환경: Google Colab

---
