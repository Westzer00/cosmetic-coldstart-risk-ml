# Methodology

## 1. Cold-Start Problem Definition

핵심 연구 질문:

> 리뷰가 없는 신제품에서도 출시 전에 확보 가능한 정보만으로 피부유형별 부정 경험 위험을 선별할 수 있는가?

과거 리뷰는 `Product × Skin Type` Risk Label을 생성하는 데만 사용하고,
신제품 예측 시에는 리뷰 정보를 Feature로 사용하지 않습니다.

## 2. Target

과거 리뷰의 부정 경험 정보를 제품×피부유형 단위로 집계한 뒤 Bayesian Smoothing을 적용하고,
조정된 부정 경험 위험이 높은 상위 25% 조합을 High-Risk로 정의했습니다.

## 3. Features

- Ingredient Presence
- Functional Groups
- Literature-based Skin × Ingredient Interaction
- Product Claim Themes
- Skin Type

## 4. Validation

동일 제품/처방 Leakage를 줄이기 위해 `formula_group` 단위 Group Split을 사용했습니다.

- Train: 869 observations / 140 formula groups
- Test: 255 observations / 36 formula groups
- Group overlap: 0
- Train 내부 5-Fold Stratified Group CV
- Threshold 선택은 Train OOF prediction만 사용
- Test는 최종 모델 평가에서만 사용

**제약:** 최종 실험은 300개 `GroupShuffleSplit` 후보 중 Train/Test 고위험 비율이 전체와 비슷한 분할을 선택했으며, 고위험군 라벨의 평활화/75% 기준도 전체 표본을 바탕으로 먼저 정했습니다. 그룹 중복 방지와 별개로, 엄격한 Train-only 타깃 기준 및 사전 고정 Hold-out에 대한 추가 민감도 분석이 필요합니다.

## 5. Model Selection

동일한 분할 조건에서 Logistic Regression, Random Forest, XGBoost, CatBoost를 비교했습니다.

최종 선택:
- Feature Set: V2
- Model: Random Forest
- CV PR-AUC: 0.3524

## 6. Interpretation

이 연구는 부작용 진단이 아니라 출시 전 Review-free Risk Screening을 목표로 합니다.
