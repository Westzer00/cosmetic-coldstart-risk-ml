# 모델링 실험 버전과 학술 포스터 결과

## 최종 발표 코드 (2026년 8월)

[05_poster_final_formula_group.ipynb](../notebooks/05_poster_final_formula_group.ipynb)

- 개인 작업 폴더의 `머신러닝_디벨롭_(모델_돌리기).ipynb`에서 원본 소스를 추출
- 모델 실행 출력·위젯 상태 제거, 기업·개인 실제 원천 데이터 미포함
- 47,487건 리뷰에 대한 제품×피부유형 Bayesian Smoothing, Prior Strength 20
- 최종 1,124개 표본, `formula_group` 기준 Train 869건 / Test 255건, 그룹 중복 0
- 최종 V2 입력 변수 + Random Forest, Test PR-AUC 0.384063 / ROC-AUC 0.623036
- 당시 기록된 임계값 0.47에서는 High-Risk 65건 중 41건 탐지

## 별도 탐색 실험 (2026년 7월)

[04_model_evaluation.ipynb](../notebooks/04_model_evaluation.ipynb)

- 최소 리뷰 수 5개, 제품 ID 기준 교차검증, 약 1,281개 학습·평가 표본
- 서로 다른 제품 ID가 같은 `formula_group`에 속할 때 처방 그룹 누수 가능
- **최종 학술 포스터의 결과를 산출한 실험과 다름**

## 재검증 조건

V1·V2 및 리뷰 원본을 이용해 최종 포스터 모델을 별도 재학습했을 때 Test PR-AUC 0.384063, ROC-AUC 0.623036 재현. 당시 임계값 0.47 고정 시 혼동행렬 [[111,79],[24,41]] 재현.

현재 scikit-learn 1.8.0 환경에서 임계값을 새로 선택하면 0.38이 계산됨. 확률 예측의 일치와 임계값 선택의 차이는 구분해야 하며, 원본 라이브러리 버전이 완전히 확정된 상태는 아닙니다.
