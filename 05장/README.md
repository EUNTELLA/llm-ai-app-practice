
---

# 📊 단일 GPU 효율적으로 활용하기 

## 🔹 실험 환경

* 모델: `EleutherAI/polyglot-ko-1.3b`
* GPU:
* CUDA 버전:
* PyTorch 버전:

---

# 1️⃣ 기본 모델 로딩

| 항목              | 값 |
| --------------- | - |
| 초기 GPU 메모리 사용량  |   |
| 모델 로드 후 메모리 사용량 |   |
| 데이터 타입          |   |

---

# 2️⃣ 배치 사이즈별 메모리 사용량

| 배치 사이즈 | 최대 GPU 메모리 (GB) | Gradient 메모리 (GB) | Optimizer 메모리 (GB) | OOM 여부 |
| ------ | --------------- | ----------------- | ------------------ | ------ |
| 4      |                 |                   |                    |        |
| 8      |                 |                   |                    |        |
| 16     |                 |                   |                    |        |

---

# 3️⃣ Gradient Accumulation 적용

* 설정:

  * batch_size = 4
  * gradient_accumulation_steps = 4

| 항목            | 값 |
| ------------- | - |
| 최대 GPU 메모리    |   |
| Gradient 메모리  |   |
| Optimizer 메모리 |   |
| OOM 여부        |   |

---

# 4️⃣ Gradient Checkpointing 적용

* 설정:

  * batch_size = 16
  * gradient_checkpointing = True

| 항목            | 값 |
| ------------- | - |
| 최대 GPU 메모리    |   |
| Gradient 메모리  |   |
| Optimizer 메모리 |   |
| OOM 여부        |   |

---

# 5️⃣ LoRA 적용

* 설정:

  * batch_size = 16
  * LoRA (r=8)

| 항목             | 값 |
| -------------- | - |
| 최대 GPU 메모리     |   |
| Gradient 메모리   |   |
| Optimizer 메모리  |   |
| 학습 가능한 파라미터 비율 |   |
| OOM 여부         |   |

---

# 6️⃣ QLoRA (4bit) 적용

* 설정:

  * batch_size = 16
  * 4bit NF4 + LoRA

| 항목             | 값 |
| -------------- | - |
| 최대 GPU 메모리     |   |
| Gradient 메모리   |   |
| Optimizer 메모리  |   |
| 학습 가능한 파라미터 비율 |   |
| OOM 여부         |   |

---

# 📈 결과 분석

## 🔹 1. 배치 사이즈 영향

*

## 🔹 2. Gradient Accumulation 효과

*

## 🔹 3. Gradient Checkpointing 효과

*

## 🔹 4. LoRA 효과

*

## 🔹 5. QLoRA 효과

*

---

# 🧠 결론

*

---

