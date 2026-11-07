# 리뷰 없는 신제품의 피부유형별 부정 경험 위험 선별

> **출시 전에 확보 가능한 정보만으로 검토 우선순위를 정하는 머신러닝 연구**  
> 2026 한국데이터마이닝학회 하계학술대회 포스터 발표 · 제1저자

리뷰가 아직 존재하지 않는 신제품에서도 **출시 전에 확보 가능한 성분·제품명·피부유형 정보만으로**
피부유형별 부정 경험 위험이 높은 조합을 우선 선별할 수 있는지 탐색한 Cold-Start Machine Learning 프로젝트입니다.

---

## 확인할 수 있는 개발·연구 역량

- **문제 정의:** 리뷰가 없는 신제품에서 어떤 피부유형을 먼저 검토할지 예측 대상으로 설정
- **데이터 설계:** 실제 출시 전에 알 수 없는 리뷰는 입력에서 제외하고, 성분·제품명·피부유형으로 변수를 직접 구성
- **검증 방법:** 유사 처방군을 학습·시험에 겹치지 않게 나눠 과대평가 위험을 줄임
- **결과 확인:** 독립 테스트에서 고위험 조합 65개 중 41개 탐지(재현 조건과 한계는 아래 명시)

> 4인 팀의 연구 결과입니다. 성분 전처리와 입력 변수 설계, 연구 통합·제1저자 발표를 주도했고 모델 비교·검증에는 팀과 함께 참여했습니다.

## 1. 핵심 문제

기존 화장품 분석은 출시 후 쌓인 리뷰를 활용하는 경우가 많습니다.

하지만 신제품은 아직 리뷰가 없기 때문에 다음 질문에서 프로젝트를 시작했습니다.

> **리뷰가 0개인 신제품의 피부유형별 부정 경험 위험을 출시 전에 선별할 수 있을까?**

이 프로젝트의 핵심 원칙은 다음과 같습니다.

```text
과거 제품의 Review
        │
        └──> Risk Label 생성에만 사용
                    │
                    ▼
            Product × Skin Type
              High-Risk Label

신제품 예측 입력
        │
        ├── Ingredient Information
        ├── Product Name / Claim
        └── Skin Type
                    │
                    ▼
              ML Model
                    │
                    ▼
       Skin-Type-Specific Risk Score
```

### 중요

**신제품 예측 시 Review text, Review count, Review statistics를 입력 Feature로 사용하지 않습니다.**

과거 리뷰는 학습용 Target(Label)을 구성하는 데에만 활용합니다.

---

## 2. 데이터 규모

최종 학회 분석 기준:

| 항목 | 규모 |
|---|---:|
| Cream Products | 231 |
| Historical Reviews | 47,487 |
| Skin Types | 7 |
| Product × Skin Observations | 1,124 |
| Held-out Test Observations | 255 |
| Actual High-Risk in Test | 65 |

원본 리뷰와 수집 데이터는 공개하지 않습니다.

자세한 데이터 공개 원칙은 [`data/README.md`](data/README.md)를 참고해주세요.

---

## 3. 담당 역할

4인 팀 프로젝트에서 **팀장**으로 참여했으며, 다음 영역을 중심으로 수행했습니다.

- 화장품 원천 데이터 및 성분 데이터 전처리
- 성분 문자열 정규화 및 Product–Ingredient 구조화
- Feature Engineering 설계·구현
- 모델링 및 결과 분석 참여
- 연구 문제정의 및 실험 구조 정리
- 학술 포스터 전체 통합 및 제1저자

> 모델링 전체를 단독 수행한 프로젝트가 아니라 팀 프로젝트이며, 본 저장소는 제가 직접 설명 가능한 전처리·Feature Engineering·검증 구조와 최종 연구 결과를 중심으로 정리합니다.

---

## 4. 성분 데이터 전처리

원천 전성분 데이터는 모델에 바로 사용할 수 없는 형식이 많아 별도의 정제 Pipeline을 구성했습니다.

실제 전처리 코드에서는 다음과 같은 문제를 처리합니다.

- `전성분 정제수` → `정제수`
- `스쿠알란(150,000ppm)` → 성분명 / 함량 분리
- 마침표로 연결된 성분 문자열 분리
- `사용방법`, 설명문, 마케팅 문구 제거
- 잘못 닫힌 `[크림}` 같은 Section 표기 보정
- 본품 / 증정품 / 다중 옵션 Section 구분
- MFDS 표준 성분명 Mapping 및 Alias 처리
- Product–Ingredient Long Format 변환

실제 Notebook:

[`notebooks/01_ingredient_preprocessing.ipynb`](notebooks/01_ingredient_preprocessing.ipynb)

---

## 5. 출시 전 예측 대상과 정답 설계

과거 리뷰는 예측 입력에 넣지 않고 **제품 × 피부유형 단위 Risk Label 생성**에만 사용했습니다.

```text
Historical Reviews
        │
        ▼
Product × Skin Type
Negative Experience Rate
        │
        ▼
Bayesian Smoothing
        │
        ▼
Top 25%
HIGH-RISK LABEL
```

이후 신제품 Prediction 단계에서는 리뷰 정보를 제거하고 출시 전에 확보 가능한 정보만 사용합니다.

---

## 6. 입력 변수 설계

단순 성분 유무 외에도 화장품 Domain 정보를 반영하도록 여러 Feature Group을 구성했습니다.

### 입력 변수 구성

| Feature Group | 내용 |
|---|---|
| Ingredient Presence | 빈도 필터 성분 존재 여부 |
| Functional Groups | 보습·유연·장벽·진정 등 성분 기능군 |
| Skin × Ingredient Interaction | 문헌 근거 기반 피부유형–성분 상호작용 |
| Product Claim Theme | 제품명에서 추출한 강조/Claim Theme |
| Skin Type | 7개 피부유형 One-Hot |

최종 연구 과정에서는 **308개 Feature를 설계**했고, 전처리·분할 이후 실제 최종 학습에는 **301개 Feature**가 사용되었습니다.

실제 Feature Engineering Notebook:

- [`notebooks/02_final_feature_engineering.ipynb`](notebooks/02_final_feature_engineering.ipynb)
- [`notebooks/03_evidence_guided_interactions.ipynb`](notebooks/03_evidence_guided_interactions.ipynb)

---

## 7. 데이터 누수 방지와 검증

같은 제품 또는 동일 처방이 Train과 Validation/Test에 동시에 포함되면 모델이 성분 구성을 사실상 암기할 수 있습니다.

따라서 일반적인 Random Row Split 대신 **formula_group 기반 Group Split**을 적용했습니다.

```text
Formula A ───────── Train
Formula B ───────── Train
Formula C ───────── Train

Formula D ───────── Test
Formula E ───────── Test

Group Overlap = 0
```

### 최종 검증 구조

- Hold-out Train: **869 observations / 140 formula groups**
- Held-out Test: **255 observations / 36 formula groups**
- Train/Test formula-group overlap: **0**
- Train 내부: **5-Fold Stratified Group Cross-Validation**
- Threshold는 Test를 보지 않고 **Train OOF prediction만으로 선택**

모델링 및 평가 흐름은 다음 파일에 정리되어 있습니다.

[`docs/model-evaluation.md`](docs/model-evaluation.md)

---

## 8. 모델 비교·선정

다음 모델을 동일한 Group-CV 조건에서 비교했습니다.

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

최종 후보 비교에서 **V2 Feature Set + Random Forest**가 가장 높은 CV PR-AUC를 기록해 최종 모델로 선택했습니다.

| Model | CV PR-AUC |
|---|---:|
| Logistic Regression | 0.322 |
| **Random Forest** | **0.352** |
| XGBoost | 0.336 |
| CatBoost | 0.329 |

---

## 9. 독립 테스트 결과

### 최종 Random Forest

| Metric | Result |
|---|---:|
| **PR-AUC** | **0.384** |
| **Recall** | **63.1%** |
| **F1-score** | **0.443** |
| Precision | 0.342 |
| F0.5 | 0.376 |
| ROC-AUC | 0.623 |

Test의 실제 High-Risk 조합은 65개였고, 최종 모델은 그중 **41개를 탐지**했습니다.

```text
Actual High-Risk : 65
Detected         : 41
Risk Recall      : 63.1%
```

이 모델의 목적은 피부 부작용을 **진단**하는 것이 아니라,
리뷰가 없는 신제품에서 **우선 검토가 필요한 Product × Skin 조합을 Screening**하는 것입니다.

상세 결과는 [`results/final_metrics.md`](results/final_metrics.md)를 참고해주세요.

---

## 10. 파일 구조

```text
.
├── README.md
├── CONTRIBUTION.md
├── requirements.txt
├── notebooks/
│   ├── 01_ingredient_preprocessing.ipynb
│   ├── 02_final_feature_engineering.ipynb
│   ├── 03_evidence_guided_interactions.ipynb
├── docs/
│   ├── methodology.md
│   └── model-evaluation.md
├── data/
│   └── README.md
└── results/
    └── final_metrics.md
```

---

## 11. 학술 발표

**2026 한국데이터마이닝학회 하계학술대회**  
포스터 번호 **P1-62** · **제1저자**

**「리뷰 없는 신제품 화장품의 피부유형별 부정 경험 위험 선별: 출시 전 성분정보 기반 콜드스타트 접근」**

[학회 포스터 보기 (Google Drive)](https://drive.google.com/file/d/1T4tSPJzYOJ8ffjohgmTlTBM21td5WK6I/view)

---

## 12. 적용 범위와 한계

- 본 모델은 의료적 부작용을 진단하는 모델이 아닙니다.
- 데이터는 크림 카테고리를 중심으로 구성되어 있습니다.
- 리뷰 기반 Target 자체에도 사용자 선택 편향과 표현 편향이 존재할 수 있습니다.
- 목적은 자동 의사결정보다 **출시 전 Risk Screening과 검토 우선순위 지원**에 있습니다.
