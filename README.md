# Reddit Mental Health Risk Classification
**SNS 텍스트 기반 정신건강 위험군 분류 — Naïve Bayes 직접 구현**

---

## 프로젝트 개요
Reddit의 r/depression, r/SuicideWatch(위험군)와 r/happy(비위험군) 게시물을 분류하는 Naïve Bayes 모델을 구현했습니다.  
단순히 라이브러리를 사용하는 것에 그치지 않고, 수업에서 배운 **Naïve Bayes 원리를 직접 구현**하고 sklearn 결과와 비교 검증했습니다.

---

## 데이터
- **출처**: Kaggle — Reddit Depression Dataset
- **사용 subreddit**: depression, SuicideWatch (위험군 = 1) / happy (비위험군 = 0)
- **클래스 불균형 처리**: 위험군 480,422건 → 24,609건 언더샘플링 (1:1 균형)
- **최종 데이터**: 49,218건 (위험군 24,609 / 비위험군 24,609)

---

## 방법론
### Naïve Bayes 직접 구현
수업에서 배운 원리를 그대로 적용했습니다.

**핵심 수식**
```
P(Y=y | X=x) ∝ P(Y=y) × ∏ P(Xj=xj | Y=y)
```

**구현 순서**
1. 텍스트 전처리 (소문자 변환, 특수문자 제거)
2. CountVectorizer로 Bag of Words 행렬 생성 (binary=True, vocab=1,000)
3. 클래스별 단어 등장 횟수 카운팅 `C(Xj, Y=y)`
4. MAP 추정으로 조건부 확률 계산 (Beta prior, alpha=beta=1 → smoothing)
5. Log probability로 분류

**Smoothing 적용**
```
P(Xj=xj | Y=y) = (C(Xj=xj, Y=y) + α - 1) / (C(Y=y) + α + β - 2)
```
α=β=1 → 분자에 +1 (Laplace smoothing과 동일)

---

## 결과

### 분류 정확도
| 모델 | 정확도 |
|------|--------|
| 직접 구현 Naïve Bayes | **83.66%** |
| sklearn BernoulliNB | **83.66%** |

직접 구현 결과가 sklearn과 일치

### 위험군에서 많이 나오는 단어 Top 20
| 단어 | 확률 차이 |
|------|-----------|
| dont | 0.0640 |
| want | 0.0625 |
| feel | 0.0549 |
| im | 0.0543 |
| depression | 0.0515 |
| like | 0.0363 |
| suicide | 0.0347 |
| depressed | 0.0330 |
| kill | 0.0307 |
| die | 0.0307 |
| help | 0.0302 |
| anymore | 0.0276 |
| know | 0.0257 |
| need | 0.0235 |
| hate | 0.0214 |
| suicidal | 0.0188 |
| life | 0.0180 |
| think | 0.0177 |
| does | 0.0174 |
| people | 0.0173 |

**해석**: 단순한 명사("suicide", "depression")보다 감정 동사와 부정어("dont", "want", "feel", "anymore")가 상위에 위치한다는 점이 흥미롭습니다. 위험군 텍스트는 특정 단어보다 **감정 표현 패턴**에서 차별화됩니다.

---

## 한계
- `title`만 사용 (body 결측치 46만건으로 제외)
- 언더샘플링으로 위험군 데이터 대부분 미사용
- 단어 간 순서/문맥 정보 미반영 (Bag of Words의 한계)
- False Negative(위험군을 비위험군으로 분류)가 실제 적용 시 더 위험할 수 있음

---

## 사용 기술
- Python, Pandas, NumPy
- scikit-learn (CountVectorizer, BernoulliNB)
- Jupyter Notebook
