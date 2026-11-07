# 공개 포스터 코드 검수 — 2026-11-08

## 실험 버전별 위치
- `notebooks/04_model_evaluation.ipynb`: 제품 ID(`product_id`) 기준 그룹 교차검증을 사용한 탐색용 실험. 포스터 최종 결과로 인용하지 않습니다.
- `notebooks/05_poster_final_formula_group.ipynb`: 유사 처방군(`formula_group`) 기준 Train/Test 분리와 Train 내부 StratifiedGroupKFold를 사용하는 포스터 최종 코드.

## 추가 검증 내용
`tests/test_poster_validation_protocol.py`는 **GitHub에 포함된 최종 노트북의 실제 함수 정의**(`select_feature_columns`, `make_balanced_group_split`)를 추출해 합성 데이터로 실행합니다. 모델 입력에서 리뷰/라벨/식별정보가 제외되는지, 처방군 Train/Test가 분리되는지, 시드가 같을 때 분할이 같은지, Train 내부 그룹 CV가 Hold-out과 겹치지 않는지를 점검합니다.

**테스트를 통과해도 47,487건 원본 리뷰 데이터, 최종 모델 학습, CV PR-AUC 및 65건 중 41건 탐지 성능을 재현한 것은 아닙니다.**

## 추가 방법론 검토사항
1. `make_balanced_group_split`는 300개 후보 중 전체와 Test의 라벨 비율이 유사한 분할을 선택합니다. 처방군 중복은 없지만 Test 라벨 분포를 분할 결정에 사용했으므로 무작위 1회 추출 Hold-out과 구분해야 합니다.
2. 고위험군 75% 기준 및 평활화용 전체 부정률은 Train/Test 분할 전에 정의됩니다. 사전에 정의된 타깃 체계로 취급할 수 있는지 확인하고, 엄격한 전향적 평가를 위해 Train-only 기준으로 추가 민감도 분석을 권장합니다.
3. 원본 데이터가 제공되지 않은 공개 저장소에서는 실제 학습 결과까지 자동 검증할 수 없습니다.
4. 합성 테스트는 함수의 핵심 계약을 확인합니다. 학회 성능 수치나 추론 모형의 보정 성능을 검증하는 것은 아닙니다.

> 기존 학회 결과는 변경하지 않고, 공개된 구현의 증명 범위를 명확하게 구분하기 위한 감사 기록입니다.
