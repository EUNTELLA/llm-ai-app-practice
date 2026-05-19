# 11장 시험 대비 노트

## 핵심 주제

11장은 크게 다음 내용을 다룬다.

- 문장 임베딩 모델 학습
- STS 데이터셋을 이용한 의미 유사도 학습
- MRC 데이터셋을 이용한 검색용 임베딩 모델 미세 조정
- Bi-Encoder와 Cross-Encoder 비교
- FAISS를 이용한 벡터 검색
- Cross-Encoder를 이용한 검색 결과 재정렬
- Hit Rate를 이용한 검색 성능 평가

전체 흐름은 다음과 같다.

```text
문장 쌍 유사도 학습
→ STS 데이터로 Bi-Encoder 학습
→ MRC 데이터로 검색용 임베딩 모델 미세 조정
→ Cross-Encoder 학습
→ FAISS로 Top-k 검색
→ Cross-Encoder로 재정렬
```

## 1. 문장 임베딩 모델

문장 임베딩 모델은 문장을 고정 길이 벡터로 변환하는 모델이다.

의미가 비슷한 문장은 벡터 공간에서 가깝게 위치하고, 의미가 다른 문장은 멀게 위치하도록 학습한다.

노트북에서는 `klue/roberta-base`를 불러와 `SentenceTransformer` 구조로 감싼다.

```python
from sentence_transformers import SentenceTransformer, models

transformer_model = models.Transformer('klue/roberta-base')

pooling_layer = models.Pooling(
    transformer_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

embedding_model = SentenceTransformer(
    modules=[transformer_model, pooling_layer]
)
```

### Pooling

Transformer 모델은 문장 전체 벡터를 바로 주는 것이 아니라 토큰별 임베딩을 출력한다.

문장 임베딩을 만들려면 토큰 임베딩 여러 개를 하나의 벡터로 합쳐야 한다. 이 과정을 `pooling`이라고 한다.

노트북에서는 `mean pooling`을 사용한다.

```text
토큰 임베딩들의 평균 → 문장 임베딩
```

시험 포인트:

```text
Pooling은 토큰 단위 임베딩을 문장 단위 임베딩 하나로 변환하는 과정이다.
```

## 2. STS 데이터셋

STS는 Semantic Textual Similarity의 약자이다.

두 문장이 의미적으로 얼마나 비슷한지를 점수로 나타낸 데이터셋이다.

KLUE STS 데이터는 다음 구조를 가진다.

```text
sentence1
sentence2
label
```

예시는 다음과 같다.

```text
sentence1: 숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다.
sentence2: 숙박시설의 위치는 쉽게 찾을 수 있고 한국의 대표적인 반지하 숙박시설입니다.
label: 3.7
```

원래 label은 0점부터 5점까지이므로 학습을 위해 0부터 1 사이로 정규화한다.

```python
label = data['labels']['label'] / 5.0
```

시험 포인트:

```text
STS는 문장 쌍의 의미 유사도를 학습하기 위한 데이터셋이다.
label을 0~1로 정규화하는 이유는 모델이 예측하는 유사도 범위와 맞추기 위해서이다.
```

## 3. InputExample

`InputExample`은 `sentence-transformers`에서 학습 데이터를 표현하는 기본 객체이다.

STS 학습에서는 다음처럼 사용한다.

```python
InputExample(
    texts=[sentence1, sentence2],
    label=similarity_score
)
```

검색 학습에서는 질문과 문서를 쌍으로 넣는다.

```python
InputExample(
    texts=[question, context],
    label=1
)
```

시험 포인트:

```text
InputExample은 학습에 사용할 문장 쌍과 label을 담는 객체이다.
```

## 4. CosineSimilarityLoss

STS 학습에서는 `CosineSimilarityLoss`를 사용한다.

```python
from sentence_transformers import losses

train_loss = losses.CosineSimilarityLoss(model=embedding_model)
```

이 손실 함수는 두 문장 임베딩의 코사인 유사도가 실제 label과 가까워지도록 학습한다.

```text
문장1 → 벡터1
문장2 → 벡터2
cosine similarity(벡터1, 벡터2) ≈ 정답 label
```

노트북의 결과:

```text
학습 전 성능: 약 0.36
학습 후 성능: 약 0.89
```

시험 포인트:

```text
CosineSimilarityLoss는 두 문장 임베딩의 코사인 유사도가 정답 유사도 점수와 가까워지도록 학습하는 손실 함수이다.
```

## 5. Bi-Encoder

Bi-Encoder는 두 문장을 각각 따로 인코딩한 뒤, 벡터 간 유사도를 계산하는 방식이다.

```text
질문 → 임베딩
문서 → 임베딩
질문 벡터와 문서 벡터의 거리 또는 유사도 계산
```

장점:

- 문서 임베딩을 미리 계산할 수 있다.
- 검색 속도가 빠르다.
- FAISS 같은 벡터 검색 인덱스를 사용할 수 있다.

단점:

- 질문과 문서를 함께 보지 않는다.
- 세밀한 관련성 판단은 Cross-Encoder보다 약할 수 있다.

시험 포인트:

```text
Bi-Encoder는 질문과 문서를 따로 임베딩하므로 빠른 검색에 적합하다.
```

## 6. MRC 데이터셋을 검색 데이터로 활용

MRC는 Machine Reading Comprehension의 약자이다.

원래는 지문을 읽고 질문에 답하는 기계 독해 데이터셋이다.

기본 구조는 다음과 같다.

```text
question
context
answer
```

노트북에서는 MRC 데이터를 검색 학습에 활용한다.

질문과 정답 문맥은 관련 있는 쌍이다.

```text
(question, context) → positive pair
```

질문과 관련 없는 문맥은 관련 없는 쌍이다.

```text
(question, irrelevant_context) → negative pair
```

관련 없는 문맥은 제목이 다른 기사에서 임의로 하나를 뽑아 만든다.

```python
df.query(f"title != '{title}'").sample(n=1)['context'].values[0]
```

시험 포인트:

```text
MRC 데이터셋의 question-context 쌍은 검색 모델 학습을 위한 positive pair로 활용할 수 있다.
```

## 7. MultipleNegativesRankingLoss

MRC 기반 임베딩 모델 미세 조정에서는 `MultipleNegativesRankingLoss`를 사용한다.

```python
from sentence_transformers import losses

loss = losses.MultipleNegativesRankingLoss(sentence_model)
```

이 손실 함수는 같은 배치 안에서 정답 쌍이 아닌 다른 문서들을 자동으로 negative로 간주한다.

예를 들어 배치가 다음과 같다고 하자.

```text
(q1, c1)
(q2, c2)
(q3, c3)
```

모델은 다음을 학습한다.

```text
q1은 c1과 가까워야 한다.
q1은 c2, c3과는 멀어야 한다.

q2는 c2와 가까워야 한다.
q2는 c1, c3과는 멀어야 한다.

q3은 c3과 가까워야 한다.
q3은 c1, c2와는 멀어야 한다.
```

시험 포인트:

```text
MultipleNegativesRankingLoss는 배치 안의 다른 positive 문서들을 자동으로 negative로 간주하여 학습한다.
검색용 Bi-Encoder 학습에 자주 사용된다.
```

## 8. NoDuplicatesDataLoader

`NoDuplicatesDataLoader`는 하나의 배치 안에 중복 문장이 들어가지 않도록 한다.

```python
from sentence_transformers import datasets

loader = datasets.NoDuplicatesDataLoader(
    train_samples,
    batch_size=16
)
```

`MultipleNegativesRankingLoss`는 배치 안의 다른 샘플을 negative로 보기 때문에, 실제로 같은 의미의 문장이 배치 안에 들어가면 잘못된 negative가 생길 수 있다.

따라서 중복을 제거하는 것이 중요하다.

시험 포인트:

```text
NoDuplicatesDataLoader는 MultipleNegativesRankingLoss에서 잘못된 negative pair가 생기는 것을 줄이기 위해 사용한다.
```

## 9. Cross-Encoder

Cross-Encoder는 질문과 문서를 하나의 입력으로 함께 넣고 관련성 점수를 직접 예측한다.

```text
[질문, 문서] → 모델 → 관련성 점수
```

노트북에서는 다음 모델을 사용한다.

```python
from sentence_transformers.cross_encoder import CrossEncoder

cross_model = CrossEncoder('klue/roberta-small', num_labels=1)
```

Bi-Encoder와 Cross-Encoder 비교:

| 구분 | Bi-Encoder | Cross-Encoder |
|---|---|---|
| 입력 방식 | 질문과 문서를 따로 인코딩 | 질문과 문서를 함께 인코딩 |
| 속도 | 빠름 | 느림 |
| 검색 인덱스 | 사용 가능 | 직접 전체 검색은 비효율적 |
| 정확도 | 상대적으로 낮음 | 상대적으로 높음 |
| 활용 | 1차 검색 | 재정렬 |

시험 포인트:

```text
Bi-Encoder는 빠른 후보 검색에 적합하고, Cross-Encoder는 후보 문서의 정밀한 재정렬에 적합하다.
```

## 10. Reranking

Reranking은 1차 검색으로 가져온 후보 문서들의 순서를 다시 매기는 과정이다.

노트북의 흐름은 다음과 같다.

```text
1. Bi-Encoder + FAISS로 Top-30 문서 검색
2. Cross-Encoder로 질문-문서 관련성 점수 계산
3. 점수가 높은 순서대로 재정렬
4. 최종 Top-10 평가
```

관련성 점수를 기준으로 내림차순 정렬한다.

```python
relevance_scores = cross_model.predict(input_examples)
reranked_indices = indices[np.argsort(relevance_scores)[::-1]]
```

노트북의 성능 비교:

```text
기본 임베딩 모델 hit rate: 0.88
MRC로 미세 조정한 임베딩 모델 hit rate: 0.946
미세 조정 임베딩 + Cross-Encoder 재정렬 hit rate: 0.973
```

다만 재정렬은 시간이 많이 걸린다.

```text
임베딩 검색: 약 14초
재정렬 포함: 약 1103초
```

시험 포인트:

```text
Reranking은 검색 정확도를 높이지만 계산 비용과 시간이 증가한다.
```

## 11. FAISS

FAISS는 벡터 검색 라이브러리이다.

문서들을 임베딩한 뒤 인덱스에 저장하고, 질문 임베딩과 가까운 문서를 빠르게 찾는다.

```python
import faiss

def make_embedding_index(sentence_model, corpus):
    embeddings = sentence_model.encode(corpus)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index
```

검색할 때는 질문을 임베딩한 뒤 가까운 문서를 찾는다.

```python
def find_embedding_top_k(query, sentence_model, index, k):
    embedding = sentence_model.encode([query])
    distances, indices = index.search(embedding, k)
    return indices
```

`IndexFlatL2`는 L2 거리, 즉 유클리드 거리를 기준으로 가장 가까운 벡터를 찾는다.

시험 포인트:

```text
FAISS는 대량의 벡터에서 가장 가까운 벡터를 빠르게 찾기 위한 라이브러리이다.
```

## 12. Hit Rate

Hit Rate는 검색 결과 Top-k 안에 정답 문서가 포함되어 있는 비율이다.

```text
Hit Rate = 정답 문서를 찾은 질문 수 / 전체 질문 수
```

예를 들어 1000개 질문 중 946개에서 Top-10 안에 정답 문서가 있으면 다음과 같다.

```text
Hit Rate@10 = 0.946
```

노트북에서는 질문의 정답 context가 검색 결과 안에 있는지 확인하여 hit 여부를 판단한다.

```python
if contexts[pred] == contexts[idx]:
    hit_count += 1
    break
```

시험 포인트:

```text
Hit Rate@k는 Top-k 검색 결과 안에 정답이 포함되었는지를 평가하는 지표이다.
```

## 핵심 개념 요약

| 개념 | 의미 |
|---|---|
| 문장 임베딩 | 문장을 의미 벡터로 변환하는 것 |
| Pooling | 토큰 임베딩을 문장 임베딩 하나로 합치는 과정 |
| STS | 문장 쌍의 의미 유사도 데이터셋 |
| CosineSimilarityLoss | 예측 코사인 유사도와 정답 유사도를 맞추는 손실 함수 |
| Bi-Encoder | 질문과 문서를 따로 임베딩하여 빠르게 검색하는 구조 |
| Cross-Encoder | 질문과 문서를 함께 입력하여 관련성을 정밀하게 판단하는 구조 |
| MNR Loss | 배치 안의 다른 문서를 negative로 사용하는 검색 학습 손실 함수 |
| NoDuplicatesDataLoader | 배치 내 중복으로 인한 잘못된 negative를 방지 |
| FAISS | 벡터 유사도 검색 라이브러리 |
| Reranking | 1차 검색 결과를 Cross-Encoder로 다시 정렬하는 과정 |
| Hit Rate@k | Top-k 안에 정답 문서가 포함된 비율 |

## 예상 시험 문제

### 1. Bi-Encoder와 Cross-Encoder의 차이를 설명하시오.

답:

```text
Bi-Encoder는 질문과 문서를 각각 따로 임베딩한 뒤 벡터 유사도로 관련성을 계산한다.
문서 임베딩을 미리 저장할 수 있어 검색 속도가 빠르다.
Cross-Encoder는 질문과 문서를 함께 입력하여 관련성 점수를 직접 예측한다.
따라서 정확도는 높지만 모든 문서 쌍을 계산해야 하므로 느리다.
```

### 2. MultipleNegativesRankingLoss의 동작 방식을 설명하시오.

답:

```text
한 배치 안에서 정답 쌍은 positive로 보고, 같은 배치의 다른 문서들은 자동으로 negative로 간주한다.
질문이 자신의 정답 문서와는 가까워지고 다른 문서와는 멀어지도록 학습한다.
```

### 3. Reranking을 사용하는 이유는 무엇인가?

답:

```text
Bi-Encoder는 빠르게 후보 문서를 찾을 수 있지만 정밀한 관련성 판단에는 한계가 있다.
따라서 Bi-Encoder로 Top-k 후보를 먼저 찾고 Cross-Encoder로 후보들의 순서를 다시 매기면 검색 정확도를 높일 수 있다.
```

### 4. FAISS의 역할은 무엇인가?

답:

```text
FAISS는 문서 임베딩 벡터들을 인덱스에 저장하고, 질문 임베딩과 가까운 문서 벡터를 빠르게 검색하는 라이브러리이다.
```

### 5. Hit Rate@k의 의미를 설명하시오.

답:

```text
검색 결과 상위 k개 안에 정답 문서가 포함된 비율이다.
예를 들어 Hit Rate@10이 0.946이면 전체 질문 중 94.6%에서 정답 문서가 상위 10개 결과 안에 포함되었다는 뜻이다.
```

### 6. STS 데이터셋에서 label을 정규화하는 이유는 무엇인가?

답:

```text
KLUE STS의 원래 label은 0점부터 5점까지이다.
CosineSimilarityLoss에서 사용하는 유사도 점수 범위와 맞추기 위해 5로 나누어 0부터 1 사이의 값으로 정규화한다.
```

### 7. NoDuplicatesDataLoader를 사용하는 이유는 무엇인가?

답:

```text
MultipleNegativesRankingLoss는 배치 안의 다른 샘플을 negative로 사용한다.
만약 배치 안에 같은 문장이나 같은 의미의 문장이 중복되면 실제 positive가 negative처럼 취급될 수 있다.
이를 줄이기 위해 NoDuplicatesDataLoader를 사용한다.
```

## 11장 한 줄 요약

11장은 문장을 벡터로 바꾸는 임베딩 모델을 학습하고, 이를 이용해 문서를 빠르게 검색한 뒤, Cross-Encoder로 검색 결과를 재정렬하여 정확도를 높이는 방법을 다룬다.
