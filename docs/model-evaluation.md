# 신제품 부정 경험 고위험군 예측 모델 (전면 재설계)

**연구 목적**: 기존 제품의 리뷰 감성분석 결과를 학습 라벨로 활용하여, 리뷰가 없는 신제품의
**성분표 + 제품명 + 피부타입 정보만으로** 각 피부타입에서 부정적 사용경험 고위험군에 속할
가능성을 예측합니다.

**핵심 원칙**
- 리뷰 텍스트/리뷰 통계는 절대 X(입력)에 사용하지 않습니다 — Y(정답)를 만드는 데만 사용합니다.
- Y는 `groupby(["상품번호", "피부타입"])`로 **제품×피부타입** 단위로 만듭니다 (제품 단위 X 7반복 금지).
- V1(전체 전성분)과 V2(빈도필터 성분)는 **완전히 동일한 표본과 동일한 Fold**로 비교합니다.
- 모든 Train/Validation 분할(외부 CV, 내부 CV, Optuna, 임계값 선택, 조기종료)은 **product_id 그룹을 지킵니다.**
- 평가는 Accuracy/ROC-AUC 대신 **Precision, F0.5-score**를 최우선으로 봅니다.

!gdown 1EaOhd5RvAcupfuwVdrbUgU30gu2gz0Xq
!gdown 1YsZ8VKxY52FW3uKZW4n52SNUmzUTeBYb
!gdown 1P6RzYZJpjLI34cZloFyOeP4JAACJC1mu
!gdown 1nolpPr0VxXVaA1R8qDA92_PpIYz2--Kd
!gdown 1gL7UP8q3OwbtCgWt_2NySKROaUIHfAgU
!gdown 13gecxtV-Z7n3Je2SdD4qaEDhwhbp_EOE
!gdown 1wu7ZvZkVs2UB04CS0z8LjjC3LKDuHKYv
!gdown 1X3OXipT97x8l7giLOPyNcZ1e1pLLjUyr

## 1. 패키지 설치 및 환경 설정

!apt-get update -qq
!apt-get install -y fonts-nanum -qq
!fc-cache -fv
!rm -rf ~/.cache/matplotlib
!pip install catboost optuna -q

import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 60)

print("환경 설정 완료")

## 2. 설정값

# ===== 파일 경로 =====
V1_PATH = '/content/final_v1_전성분_v345.csv'
V2_PATH = '/content/final_v2_주성분_v345.csv'
REVIEW_PATH = '/content/gemini_리뷰감성분석_긍정부정_배송제외_최종_상품명추가.csv'
RESULTS_DIR = '/content/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== 최소 리뷰 수 필터 =====
MIN_REVIEWS_PER_GROUP = 5   # 제품×피부타입 조합의 최소 리뷰 수 (1, 3, 5 등으로 조정 가능)

# ===== 고위험군 라벨 정의 =====
RISK_THRESHOLD_MODE = "quantile"     # "quantile" 또는 "fixed"
RISK_QUANTILE = 0.75                 # 부정비율 상위 25% = 75% 분위수
FIXED_RISK_THRESHOLD = 0.082         # RISK_THRESHOLD_MODE == "fixed"일 때 사용

# ===== 교차검증 설정 =====
N_SPLITS = 5
RANDOM_STATE = 42

# ===== 임계값 선택 =====
MIN_RECALL_FOR_THRESHOLD = 0.30
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]

# ===== Optuna 튜닝 =====
RUN_OPTUNA = True
N_TRIALS = 30
OPTUNA_OBJECTIVE = "f05_pr_auc"   # "f05_pr_auc": 0.7*F0.5 + 0.3*PR-AUC / "f05_only": F0.5만 사용

print("설정값 로드 완료")
print(f"MIN_REVIEWS_PER_GROUP = {MIN_REVIEWS_PER_GROUP}")
print(f"RISK_THRESHOLD_MODE = {RISK_THRESHOLD_MODE} (RISK_QUANTILE={RISK_QUANTILE}, FIXED={FIXED_RISK_THRESHOLD})")
print(f"N_SPLITS = {N_SPLITS}, RANDOM_STATE = {RANDOM_STATE}")
print(f"RUN_OPTUNA = {RUN_OPTUNA}, N_TRIALS = {N_TRIALS}, OPTUNA_OBJECTIVE = {OPTUNA_OBJECTIVE}")

## 3. 데이터 로드

df_v1_raw = pd.read_csv(V1_PATH, encoding='utf-8-sig')
df_v2_raw = pd.read_csv(V2_PATH, encoding='utf-8-sig')
df_review_raw = pd.read_csv(REVIEW_PATH, encoding='utf-8-sig')

print("V1 (전체 전성분) shape:", df_v1_raw.shape)
print("V2 (빈도필터 성분) shape:", df_v2_raw.shape)
print("리뷰 데이터 shape:", df_review_raw.shape)

print("\n[V1 컬럼 전체 목록]")
print(df_v1_raw.columns.tolist())
print("\n[V2 컬럼 전체 목록]")
print(df_v2_raw.columns.tolist())
print("\n[리뷰 데이터 컬럼 전체 목록]")
print(df_review_raw.columns.tolist())

## 4. 컬럼 검사

# ---- 리뷰 데이터에 피부타입 컬럼이 있는지 검사 ----
SKIN_TYPE_CANDIDATES = ['피부타입', 'skin_type', '스킨타입', '피부 타입']
found_skin_col = None
for cand in SKIN_TYPE_CANDIDATES:
    if cand in df_review_raw.columns:
        found_skin_col = cand
        break

if found_skin_col is None:
    raise ValueError("리뷰 데이터에 피부타입 컬럼이 없어 제품×피부타입별 Y를 만들 수 없습니다.")

print(f"리뷰 데이터에서 피부타입 컬럼으로 '{found_skin_col}'을(를) 사용합니다.")

# ---- 필수 컬럼 존재 검사 ----
required_review_cols = ['상품번호', found_skin_col, '리뷰분류_1_0']
missing = [c for c in required_review_cols if c not in df_review_raw.columns]
if missing:
    raise ValueError(f"리뷰 데이터에 필수 컬럼이 없습니다: {missing}")

# ---- 리뷰분류_1_0이 0/1로만 구성되었는지 검사 ----
unique_vals = set(df_review_raw['리뷰분류_1_0'].dropna().unique().tolist())
if not unique_vals.issubset({0, 1}):
    raise ValueError(f"리뷰분류_1_0에 0/1이 아닌 값이 있습니다: {unique_vals}")
print("리뷰분류_1_0 값 검사 통과 (0/1로만 구성됨)")

# ---- V1, V2 접두사 검사 ----
def check_prefix_exists(df, prefix, name):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) == 0:
        raise ValueError(f"{name}에 '{prefix}' 접두사 컬럼이 없습니다.")
    return cols

v1_v1cols = check_prefix_exists(df_v1_raw, 'v1_', 'V1')
v1_v3cols = check_prefix_exists(df_v1_raw, 'v3_', 'V1')
v1_v4cols = check_prefix_exists(df_v1_raw, 'v4_', 'V1')
v1_v5cols = check_prefix_exists(df_v1_raw, 'v5_', 'V1')
v1_skincols = check_prefix_exists(df_v1_raw, 'skin_', 'V1')

v2_v2cols = check_prefix_exists(df_v2_raw, 'v2_', 'V2')
v2_v3cols = check_prefix_exists(df_v2_raw, 'v3_', 'V2')
v2_v4cols = check_prefix_exists(df_v2_raw, 'v4_', 'V2')
v2_v5cols = check_prefix_exists(df_v2_raw, 'v5_', 'V2')
v2_skincols = check_prefix_exists(df_v2_raw, 'skin_', 'V2')

print(f"V1: v1_*={len(v1_v1cols)}, v3_*={len(v1_v3cols)}, v4_*={len(v1_v4cols)}, v5_*={len(v1_v5cols)}, skin_*={len(v1_skincols)}")
print(f"V2: v2_*={len(v2_v2cols)}, v3_*={len(v2_v3cols)}, v4_*={len(v2_v4cols)}, v5_*={len(v2_v5cols)}, skin_*={len(v2_skincols)}")

for c in ['product_id', 'product_name', '피부타입', 'formula_group']:
    if c not in df_v1_raw.columns:
        raise ValueError(f"V1에 관리 컬럼 '{c}'가 없습니다.")
    if c not in df_v2_raw.columns:
        raise ValueError(f"V2에 관리 컬럼 '{c}'가 없습니다.")
print("V1/V2 필수 관리 컬럼 존재 확인 완료")

## 5. 제품×피부타입 Y 생성

df_review = df_review_raw.copy()
df_review = df_review.rename(columns={'상품번호': 'product_id', found_skin_col: '피부타입'})
df_review['product_id'] = df_review['product_id'].astype(str).str.strip()
df_review['피부타입'] = df_review['피부타입'].astype(str).str.strip()

print("리뷰 원본 행 수:", len(df_review))

y_group = (
    df_review
    .groupby(['product_id', '피부타입'])
    .agg(
        y_전체리뷰=('리뷰분류_1_0', 'count'),
        y_긍정=('리뷰분류_1_0', 'sum'),
    )
    .reset_index()
)
y_group['y_부정'] = y_group['y_전체리뷰'] - y_group['y_긍정']
y_group['y_부정비율'] = y_group['y_부정'] / y_group['y_전체리뷰']

print("제품×피부타입 Y 행 수:", len(y_group))

dup_keys = y_group.duplicated(['product_id', '피부타입']).sum()
if dup_keys > 0:
    raise ValueError(f"제품×피부타입 Y 키 중복이 {dup_keys}건 있습니다.")
print("제품×피부타입 Y 키 중복 없음 확인")

print("\n리뷰 수 분포:")
print(y_group['y_전체리뷰'].describe())
print("\n피부타입별 관측치 수:")
print(y_group['피부타입'].value_counts())
print("\n피부타입별 평균 부정률:")
print(y_group.groupby('피부타입')['y_부정비율'].mean().sort_values(ascending=False))

## 6. 최소 리뷰 수 필터

n_before = len(y_group)
y_filtered = y_group[y_group['y_전체리뷰'] >= MIN_REVIEWS_PER_GROUP].copy().reset_index(drop=True)
n_after = len(y_filtered)

print(f"필터링 전 제품×피부타입 수: {n_before}")
print(f"필터링 후 제품×피부타입 수: {n_after}")
print(f"제외된 행 수: {n_before - n_after} ({(n_before - n_after) / n_before * 100:.2f}%)")

print("\n피부타입별 남은 행 수:")
print(y_filtered['피부타입'].value_counts())

## 7. 고위험군 라벨 생성

if RISK_THRESHOLD_MODE == "quantile":
    risk_threshold = y_filtered['y_부정비율'].quantile(RISK_QUANTILE)
elif RISK_THRESHOLD_MODE == "fixed":
    risk_threshold = FIXED_RISK_THRESHOLD
else:
    raise ValueError(f"알 수 없는 RISK_THRESHOLD_MODE: {RISK_THRESHOLD_MODE}")

print(f"RISK_THRESHOLD_MODE = {RISK_THRESHOLD_MODE}")
print(f"제품×피부타입 기준 75% 분위수: {y_filtered['y_부정비율'].quantile(0.75):.4f}")
print(f"고위험군 기준 부정률(risk_threshold): {risk_threshold:.4f}")

y_filtered['high_risk'] = (y_filtered['y_부정비율'] >= risk_threshold).astype(int)

n_risk = (y_filtered['high_risk'] == 1).sum()
n_safe = (y_filtered['high_risk'] == 0).sum()
print(f"고위험군 행 수: {n_risk}")
print(f"일반군 행 수: {n_safe}")
print(f"고위험군 비율: {n_risk / len(y_filtered) * 100:.2f}%")

print("\n피부타입별 고위험군 비율:")
print(y_filtered.groupby('피부타입')['high_risk'].mean().sort_values(ascending=False))

risk_label_summary = y_filtered.groupby('피부타입').agg(
    n=('high_risk', 'size'), risk_rate=('high_risk', 'mean')
).reset_index()
risk_label_summary.to_csv(os.path.join(RESULTS_DIR, 'risk_label_summary.csv'), index=False, encoding='utf-8-sig')

excluded = y_group[~y_group.set_index(['product_id', '피부타입']).index.isin(
    y_filtered.set_index(['product_id', '피부타입']).index
)]
excluded.to_csv(os.path.join(RESULTS_DIR, 'excluded_low_review_groups.csv'), index=False, encoding='utf-8-sig')
print(f"\n제외된 저리뷰 그룹 {len(excluded)}건을 excluded_low_review_groups.csv로 저장")

## 8. V1·V2 병합 및 공통 키 정렬

def prepare_x_base(df_raw, name):
    df = df_raw.copy()
    df['product_id'] = df['product_id'].astype(str).str.strip()
    df['피부타입'] = df['피부타입'].astype(str).str.strip()
    dup = df.duplicated(['product_id', '피부타입']).sum()
    if dup > 0:
        raise ValueError(f"{name} product_id+피부타입 키 중복이 {dup}건 있습니다.")
    return df

df_v1 = prepare_x_base(df_v1_raw, 'V1')
df_v2 = prepare_x_base(df_v2_raw, 'V2')
print("V1 product_id+피부타입 키 중복 없음 확인")
print("V2 product_id+피부타입 키 중복 없음 확인")

n_v1_before, n_v2_before = len(df_v1), len(df_v2)

merged_v1 = pd.merge(df_v1, y_filtered, on=['product_id', '피부타입'], how='inner', validate='one_to_one')
merged_v2 = pd.merge(df_v2, y_filtered, on=['product_id', '피부타입'], how='inner', validate='one_to_one')

print(f"V1 병합 전 행 수: {n_v1_before}, 병합 후 행 수: {len(merged_v1)}")
print(f"V2 병합 전 행 수: {n_v2_before}, 병합 후 행 수: {len(merged_v2)}")

x_v1_excluded = n_v1_before - len(merged_v1)
x_v2_excluded = n_v2_before - len(merged_v2)
y_excluded_from_v1 = len(y_filtered) - len(merged_v1)
y_excluded_from_v2 = len(y_filtered) - len(merged_v2)
print(f"Y와 매칭되지 않아 제외된 V1 X 행 수: {x_v1_excluded}")
print(f"Y와 매칭되지 않아 제외된 V2 X 행 수: {x_v2_excluded}")
print(f"V1과 매칭되지 않아 제외된 Y 행 수: {y_excluded_from_v1}")
print(f"V2와 매칭되지 않아 제외된 Y 행 수: {y_excluded_from_v2}")

# ---- V1과 V2의 공통 키만 사용 ----
keys_v1 = set(zip(merged_v1['product_id'], merged_v1['피부타입']))
keys_v2 = set(zip(merged_v2['product_id'], merged_v2['피부타입']))
common_keys = keys_v1 & keys_v2
print(f"\nV1 키 수: {len(keys_v1)}, V2 키 수: {len(keys_v2)}, 공통 키 수: {len(common_keys)}")

common_keys_df = pd.DataFrame(list(common_keys), columns=['product_id', '피부타입'])
common_keys_df = common_keys_df.sort_values(['product_id', '피부타입']).reset_index(drop=True)
common_keys_df.to_csv(os.path.join(RESULTS_DIR, 'v1_v2_common_keys.csv'), index=False, encoding='utf-8-sig')

merged_v1 = pd.merge(merged_v1, common_keys_df, on=['product_id', '피부타입'], how='inner')
merged_v2 = pd.merge(merged_v2, common_keys_df, on=['product_id', '피부타입'], how='inner')

merged_v1 = merged_v1.sort_values(['product_id', '피부타입']).reset_index(drop=True)
merged_v2 = merged_v2.sort_values(['product_id', '피부타입']).reset_index(drop=True)

# ---- 일치 검증 ----
assert len(merged_v1) == len(merged_v2), "V1과 V2 행 수가 다릅니다."
assert (merged_v1['product_id'].values == merged_v2['product_id'].values).all(), "V1과 V2의 product_id 순서가 다릅니다."
assert (merged_v1['피부타입'].values == merged_v2['피부타입'].values).all(), "V1과 V2의 피부타입 순서가 다릅니다."
assert (merged_v1['high_risk'].values == merged_v2['high_risk'].values).all(), "V1과 V2의 high_risk 순서가 다릅니다."
assert np.allclose(merged_v1['y_부정비율'].values, merged_v2['y_부정비율'].values), "V1과 V2의 y_부정비율이 다릅니다."
assert merged_v1.duplicated(['product_id', '피부타입']).sum() == 0, "V1에 중복 키가 있습니다."
assert merged_v2.duplicated(['product_id', '피부타입']).sum() == 0, "V2에 중복 키가 있습니다."

print(f"\n최종 공통 표본 행 수: {len(merged_v1)}")
print("V1/V2 일치 검증 통과")

## 9. X 컬럼 선택 및 안전한 컬럼명 변환

MANAGEMENT_COLS = ['product_id', 'product_name', '피부타입', 'formula_group']
Y_LEAKAGE_COLS = ['y_전체리뷰', 'y_긍정', 'y_부정', 'y_부정비율', 'high_risk']

def select_feature_columns(df, allowed_prefixes):
    cols = []
    for c in df.columns:
        if c in MANAGEMENT_COLS or c in Y_LEAKAGE_COLS:
            continue
        if any(c.startswith(p) for p in allowed_prefixes):
            cols.append(c)
    return cols

v1_feature_cols = select_feature_columns(merged_v1, ['skin_', 'v1_', 'v3_', 'v4_', 'v5_'])
v2_feature_cols = select_feature_columns(merged_v2, ['skin_', 'v2_', 'v3_', 'v4_', 'v5_'])

print(f"V1 사용 피처 수: {len(v1_feature_cols)}")
print(f"V2 사용 피처 수: {len(v2_feature_cols)}")

# ---- v1_*가 V2에, v2_*가 V1에 섞이지 않았는지 검사 ----
assert not any(c.startswith('v2_') for c in v1_feature_cols), "V1 피처에 v2_* 컬럼이 섞여 있습니다."
assert not any(c.startswith('v1_') for c in v2_feature_cols), "V2 피처에 v1_* 컬럼이 섞여 있습니다."
print("V1/V2 교차 오염 없음 확인")

# ---- 관리 컬럼/Y 누수 컬럼이 X에 포함되지 않았는지 검사 ----
for c in MANAGEMENT_COLS + Y_LEAKAGE_COLS:
    assert c not in v1_feature_cols, f"V1 피처에 {c}가 포함되어 있습니다."
    assert c not in v2_feature_cols, f"V2 피처에 {c}가 포함되어 있습니다."
print("X에 관리 컬럼 및 Y 누수 컬럼 미포함 확인")

# ---- skin_*, v3_*, v4_*, v5_* 공통 블록 값이 V1/V2에서 동일한지 검사 ----
common_block_cols = [c for c in v1_feature_cols if c.startswith(('skin_', 'v3_', 'v4_', 'v5_'))]
mismatch = 0
for c in common_block_cols:
    if c in merged_v2.columns:
        if not np.allclose(merged_v1[c].values.astype(float), merged_v2[c].values.astype(float)):
            mismatch += 1
if mismatch > 0:
    raise ValueError(f"V1/V2 공통 블록 중 {mismatch}개 컬럼 값이 일치하지 않습니다.")
print(f"V1/V2 공통 블록({len(common_block_cols)}개 컬럼) 값 일치 확인")


def make_safe_column_names(columns, mapping_save_path):
    """특수문자를 '_'로 치환하고, 충돌 시 suffix를 붙여 원본-변환 매핑을 저장"""
    safe_names = []
    used = {}
    mapping_rows = []
    for orig in columns:
        safe = re.sub(r'[^a-zA-Z0-9\uac00-\ud7a3]', '_', str(orig))
        base_safe = safe
        cnt = used.get(base_safe, 0)
        if cnt > 0:
            safe = f"{base_safe}__dup{cnt}"
        used[base_safe] = cnt + 1
        safe_names.append(safe)
        mapping_rows.append({'original_column': orig, 'safe_column': safe})

    mapping_df = pd.DataFrame(mapping_rows)
    dup_safe = mapping_df['safe_column'].duplicated().sum()
    if dup_safe > 0:
        raise ValueError(f"안전한 컬럼명 변환 후에도 {dup_safe}건의 충돌이 남아있습니다.")

    mapping_df.to_csv(mapping_save_path, index=False, encoding='utf-8-sig')
    return safe_names, mapping_df


v1_safe_names, v1_mapping_df = make_safe_column_names(
    v1_feature_cols, os.path.join(RESULTS_DIR, 'feature_name_mapping_v1.csv')
)
v2_safe_names, v2_mapping_df = make_safe_column_names(
    v2_feature_cols, os.path.join(RESULTS_DIR, 'feature_name_mapping_v2.csv')
)

X_v1 = merged_v1[v1_feature_cols].copy()
X_v1.columns = v1_safe_names
X_v2 = merged_v2[v2_feature_cols].copy()
X_v2.columns = v2_safe_names

# ---- 모든 X가 숫자형인지 검사 ----
non_numeric_v1 = X_v1.select_dtypes(exclude=[np.number]).columns.tolist()
non_numeric_v2 = X_v2.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_v1:
    raise ValueError(f"V1 X에 숫자형이 아닌 컬럼이 있습니다: {non_numeric_v1}")
if non_numeric_v2:
    raise ValueError(f"V2 X에 숫자형이 아닌 컬럼이 있습니다: {non_numeric_v2}")
print("V1/V2 X 전부 숫자형 확인")

# ---- 결측값 검사 ----
na_v1 = X_v1.isna().sum().sum()
na_v2 = X_v2.isna().sum().sum()
if na_v1 > 0 or na_v2 > 0:
    raise ValueError(f"X에 결측값이 있습니다. V1: {na_v1}, V2: {na_v2}")
print("V1/V2 X 결측값 0 확인")

y_all = merged_v1['high_risk'].values  # V1과 V2 동일
y_ratio_all = merged_v1['y_부정비율'].values
product_id_groups = merged_v1['product_id'].values

na_y = pd.isna(y_all).sum()
if na_y > 0:
    raise ValueError(f"Y에 결측값이 {na_y}건 있습니다.")
print("Y 결측값 0 확인")

print(f"\nX_v1 shape: {X_v1.shape}")
print(f"X_v2 shape: {X_v2.shape}")
print(f"y_all shape: {y_all.shape}, 고위험군 비율: {y_all.mean():.4f}")

## 10. 공통 Fold 생성 (V1·V2·모든 모델이 공유)

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = list(cv.split(X_v1, y_all, groups=product_id_groups))

print(f"{N_SPLITS}-Fold 생성 완료 (V1, V2, 모든 모델이 이 folds를 공유)")

fold_product_lists = []
for i, (tr_idx, va_idx) in enumerate(folds):
    tr_products = set(product_id_groups[tr_idx])
    va_products = set(product_id_groups[va_idx])
    overlap = tr_products & va_products
    assert len(overlap) == 0, f"Fold {i}: train/validation product_id 교집합이 {len(overlap)}건 있습니다."

    tr_risk_rate = y_all[tr_idx].mean()
    va_risk_rate = y_all[va_idx].mean()

    print(f"Fold {i}: train 행={len(tr_idx)}(제품 {len(tr_products)}개, 위험비율 {tr_risk_rate:.3f}) "
          f"/ valid 행={len(va_idx)}(제품 {len(va_products)}개, 위험비율 {va_risk_rate:.3f})")

    fold_product_lists.append({
        'fold': i,
        'train_products': sorted(tr_products),
        'valid_products': sorted(va_products),
    })

with open(os.path.join(RESULTS_DIR, 'fold_product_lists.json'), 'w', encoding='utf-8') as f:
    json.dump(fold_product_lists, f, ensure_ascii=False, indent=2)

print("\n모든 Fold에서 train/validation product_id 교집합 0 확인 (누수 없음)")

## 11. 평가 함수

def compute_topk_metrics(y_true, y_prob, top_ratio=0.25):
    """예측확률 상위 top_ratio 비율을 고위험 예측군으로 지정하고 Precision/Recall/Lift 계산"""
    n = len(y_true)
    k = max(1, int(np.ceil(n * top_ratio)))
    order = np.argsort(-y_prob)
    top_idx = order[:k]

    top_true = y_true[top_idx]
    precision_at_k = top_true.mean() if k > 0 else np.nan
    overall_risk_rate = y_true.mean()
    recall_at_k = top_true.sum() / y_true.sum() if y_true.sum() > 0 else np.nan
    lift_at_k = precision_at_k / overall_risk_rate if overall_risk_rate > 0 else np.nan

    return precision_at_k, recall_at_k, lift_at_k


def evaluate_predictions(y_true, y_prob, y_pred):
    """Precision, F0.5, Recall, PR-AUC, F1, ROC-AUC, Accuracy, Top25% 지표를 한번에 계산"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
    acc = accuracy_score(y_true, y_pred)

    p25, r25, lift25 = compute_topk_metrics(y_true, y_prob, top_ratio=0.25)

    return {
        'Precision': precision, 'F0.5': f05, 'Recall': recall,
        'PR_AUC': pr_auc, 'F1': f1, 'ROC_AUC': roc_auc, 'Accuracy': acc,
        'Precision_at_25': p25, 'Recall_at_25': r25, 'Lift_at_25': lift25,
    }

print("평가 함수 정의 완료")

## 12. 임계값 선택 함수

def select_threshold(y_true, y_prob, thresholds=None, min_recall=MIN_RECALL_FOR_THRESHOLD):
    """외부 Train 내부(Threshold-Validation)에서 F0.5 기준으로 최적 임계값 선택.
    Recall >= min_recall 조건을 만족하는 임계값 중 F0.5가 가장 높은 것을 선택.
    조건을 만족하는 임계값이 없으면 F0.5가 가장 높은 임계값을 선택."""
    if thresholds is None:
        thresholds = THRESHOLD_GRID

    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        records.append({'threshold': t, 'precision': precision, 'recall': recall, 'f05': f05})

    rec_df = pd.DataFrame(records)
    candidates = rec_df[rec_df['recall'] >= min_recall]

    if len(candidates) == 0:
        candidates = rec_df

    max_f05 = candidates['f05'].max()
    candidates = candidates[candidates['f05'] == max_f05]
    max_precision = candidates['precision'].max()
    candidates = candidates[candidates['precision'] == max_precision]
    best_threshold = candidates['threshold'].max()

    return float(best_threshold)


def fit_predict_with_group_threshold_split(model, X_train, y_train, groups_train,
                                            X_valid, threshold_valid_size=0.2, random_state=RANDOM_STATE):
    """외부 Train 데이터를 다시 product_id 그룹 기준으로 Train/Threshold-Validation으로 나눠서
    임계값을 외부 Validation을 보지 않고 선택"""
    gss = GroupShuffleSplit(n_splits=1, test_size=threshold_valid_size, random_state=random_state)
    fit_idx, thr_idx = next(gss.split(X_train, y_train, groups=groups_train))

    fit_products = set(np.asarray(groups_train)[fit_idx])
    thr_products = set(np.asarray(groups_train)[thr_idx])
    assert len(fit_products & thr_products) == 0, "임계값 선택용 내부 분할에서 product_id가 겹칩니다."

    X_fit, X_thr = X_train.iloc[fit_idx], X_train.iloc[thr_idx]
    y_fit, y_thr = np.asarray(y_train)[fit_idx], np.asarray(y_train)[thr_idx]

    model.fit(X_fit, y_fit)
    thr_prob = model.predict_proba(X_thr)[:, 1]
    best_threshold = select_threshold(y_thr, thr_prob)

    valid_prob = model.predict_proba(X_valid)[:, 1]

    return best_threshold, valid_prob, model

print("임계값 선택 함수 정의 완료")

## 13. 기본 모델 정의

def get_model_spec(model_name, scale_pos_weight, tuned_params=None, random_state=RANDOM_STATE):
    """model_name별로 (ModelClass, params dict)를 반환.
    tuned_params가 주어지면 Optuna로 찾은 값으로 덮어쓰고, 없으면 baseline 기본값을 사용."""

    if model_name == 'LogisticRegression':
        params = dict(penalty='l2', C=1.0, max_iter=5000,
                      class_weight='balanced', random_state=random_state, solver='liblinear')
        if tuned_params:
            params['C'] = tuned_params.get('C', params['C'])
            params['penalty'] = tuned_params.get('penalty', params['penalty'])
        return LogisticRegression, params

    if model_name == 'RandomForest':
        params = dict(n_estimators=300, max_depth=6, min_samples_leaf=1, max_features='sqrt',
                      class_weight={0: 1.0, 1: float(scale_pos_weight)},
                      random_state=random_state, n_jobs=-1)
        if tuned_params:
            for k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features']:
                if k in tuned_params:
                    params[k] = tuned_params[k]
        return RandomForestClassifier, params

    if model_name == 'XGBoost':
        params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      scale_pos_weight=scale_pos_weight,
                      eval_metric='logloss', random_state=random_state)
        if tuned_params:
            for k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda']:
                if k in tuned_params:
                    params[k] = tuned_params[k]
        return XGBClassifier, params

    if model_name == 'CatBoost':
        params = dict(iterations=300, depth=3, learning_rate=0.05, l2_leaf_reg=3.0,
                      scale_pos_weight=scale_pos_weight,
                      eval_metric='Logloss', random_seed=random_state, verbose=0)
        if tuned_params:
            for k in ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg']:
                if k in tuned_params:
                    params[k] = tuned_params[k]
        return CatBoostClassifier, params

    raise ValueError(f"알 수 없는 model_name: {model_name}")

BASE_MODEL_NAMES = ['LogisticRegression', 'RandomForest', 'XGBoost', 'CatBoost']
EARLY_STOPPING_MODELS = ('XGBoost', 'CatBoost')

print("기본 모델 정의(get_model_spec) 완료. 모델:", BASE_MODEL_NAMES)

## 14. 5-Fold 그룹 교차검증 (공통 함수: baseline/tuned 재사용)

def run_group_cv(X, y, groups, dataset_name, folds, model_names,
                  tuned_params_dict=None, stage_label='baseline'):
    """공통 folds를 사용해 모델들을 5-Fold 그룹 교차검증으로 평가.
    각 Fold의 Train 내부에서만 scale_pos_weight와 임계값을 계산 (Validation 정보 미사용).
    XGBoost/CatBoost는 조기종료용 내부 검증셋도 product_id 그룹을 지켜서 분리."""
    tuned_params_dict = tuned_params_dict or {}
    fold_records = []
    oof_records = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        groups_tr = groups[tr_idx]

        n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        for model_name in model_names:
            tuned_params = tuned_params_dict.get(model_name)
            ModelClass, params = get_model_spec(model_name, scale_pos_weight, tuned_params=tuned_params)

            if model_name in EARLY_STOPPING_MODELS:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
                sub_tr_idx, sub_va_idx = next(gss.split(X_tr, y_tr, groups=groups_tr))

                sub_tr_products = set(np.asarray(groups_tr)[sub_tr_idx])
                sub_va_products = set(np.asarray(groups_tr)[sub_va_idx])
                assert len(sub_tr_products & sub_va_products) == 0, \
                    f"Fold {fold_i} {model_name} 조기종료 내부 분할에서 product_id가 겹칩니다."

                X_sub_tr, X_sub_va = X_tr.iloc[sub_tr_idx], X_tr.iloc[sub_va_idx]
                y_sub_tr, y_sub_va = y_tr[sub_tr_idx], y_tr[sub_va_idx]

                model = ModelClass(**params, early_stopping_rounds=30)
                if model_name == 'XGBoost':
                    model.fit(X_sub_tr, y_sub_tr, eval_set=[(X_sub_va, y_sub_va)], verbose=False)
                else:
                    model.fit(X_sub_tr, y_sub_tr, eval_set=(X_sub_va, y_sub_va), use_best_model=True)

                thr_prob = model.predict_proba(X_sub_va)[:, 1]
                best_threshold = select_threshold(y_sub_va, thr_prob)
            else:
                model = ModelClass(**params)
                best_threshold, _, model = fit_predict_with_group_threshold_split(
                    model, X_tr, y_tr, groups_tr, X_va
                )

            va_prob = model.predict_proba(X_va)[:, 1]
            va_pred = (va_prob >= best_threshold).astype(int)

            metrics = evaluate_predictions(y_va, va_prob, va_pred)
            metrics.update({
                'dataset': dataset_name, 'model': model_name, 'stage': stage_label, 'fold': fold_i,
                'threshold': best_threshold, 'n_train': len(tr_idx), 'n_valid': len(va_idx),
                'n_features': X.shape[1],
            })
            fold_records.append(metrics)

            for local_i, global_i in enumerate(va_idx):
                oof_records.append({
                    'product_id': product_id_groups[global_i],
                    'product_name': merged_v1['product_name'].values[global_i],
                    '피부타입': merged_v1['피부타입'].values[global_i],
                    'y_전체리뷰': merged_v1['y_전체리뷰'].values[global_i],
                    'y_부정': merged_v1['y_부정'].values[global_i],
                    'y_부정비율': merged_v1['y_부정비율'].values[global_i],
                    'high_risk': int(y_va[local_i]),
                    'dataset': dataset_name, 'model': model_name, 'stage': stage_label, 'fold': fold_i,
                    'predicted_probability': va_prob[local_i],
                    'selected_threshold': best_threshold,
                    'predicted_label': int(va_pred[local_i]),
                    'top25_selected': False,
                })

            print(f"[{dataset_name}][{stage_label}] Fold {fold_i} {model_name:18s} | "
                  f"Precision={metrics['Precision']:.3f} F0.5={metrics['F0.5']:.3f} "
                  f"Recall={metrics['Recall']:.3f} PR-AUC={metrics['PR_AUC']:.3f} "
                  f"ROC-AUC={metrics['ROC_AUC']:.3f} thr={best_threshold:.2f}")

    fold_df = pd.DataFrame(fold_records)
    oof_df = pd.DataFrame(oof_records)

    for (ds, model_name, stg, fold_i), grp in oof_df.groupby(['dataset', 'model', 'stage', 'fold']):
        k = max(1, int(np.ceil(len(grp) * 0.25)))
        top_idx = grp['predicted_probability'].sort_values(ascending=False).index[:k]
        oof_df.loc[top_idx, 'top25_selected'] = True

    return fold_df, oof_df

print("run_group_cv 함수 정의 완료")

print("=" * 80)
print("1단계: 고정 파라미터 5-Fold 그룹 교차검증 (V1 x 4모델, V2 x 4모델)")
print("=" * 80)

fold_df_v1, oof_df_v1 = run_group_cv(
    X_v1, y_all, product_id_groups, 'V1_전체전성분_V345', folds, BASE_MODEL_NAMES, stage_label='baseline'
)
fold_df_v2, oof_df_v2 = run_group_cv(
    X_v2, y_all, product_id_groups, 'V2_빈도필터성분_V345', folds, BASE_MODEL_NAMES, stage_label='baseline'
)

fold_df_baseline = pd.concat([fold_df_v1, fold_df_v2], ignore_index=True)
oof_df_baseline = pd.concat([oof_df_v1, oof_df_v2], ignore_index=True)

print(f"\nbaseline Fold별 결과 {len(fold_df_baseline)}행 생성 완료")

## 15. 결과 요약 (1단계: baseline)

def summarize_fold_metrics(fold_df):
    summary = fold_df.groupby(['dataset', 'model', 'stage']).agg(
        n_features=('n_features', 'first'),
        Precision_mean=('Precision', 'mean'), Precision_std=('Precision', 'std'),
        F05_mean=('F0.5', 'mean'), F05_std=('F0.5', 'std'),
        Recall_mean=('Recall', 'mean'), Recall_std=('Recall', 'std'),
        F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
        PR_AUC_mean=('PR_AUC', 'mean'), PR_AUC_std=('PR_AUC', 'std'),
        ROC_AUC_mean=('ROC_AUC', 'mean'), ROC_AUC_std=('ROC_AUC', 'std'),
        Accuracy_mean=('Accuracy', 'mean'), Accuracy_std=('Accuracy', 'std'),
        Precision_at_25_mean=('Precision_at_25', 'mean'),
        Recall_at_25_mean=('Recall_at_25', 'mean'),
        Lift_at_25_mean=('Lift_at_25', 'mean'),
        Threshold_mean=('threshold', 'mean'), Threshold_std=('threshold', 'std'),
    ).reset_index()

    summary = summary.rename(columns={
        'dataset': '데이터셋', 'model': '모델', 'stage': '단계', 'n_features': 'Feature 수'
    })
    return summary


baseline_summary = summarize_fold_metrics(fold_df_baseline)
baseline_summary_f05 = baseline_summary.sort_values(
    ['F05_mean', 'Precision_mean', 'PR_AUC_mean'], ascending=False
).reset_index(drop=True)

print("[baseline: F0.5 기준 정렬]")
display(baseline_summary_f05)

## 16. Optuna 튜닝 (2단계)

1단계 baseline 비교 결과와 별개로, `RUN_OPTUNA=True`이면 4개 모델 × 2개 데이터셋 전부에 대해
Optuna 탐색을 실행합니다 (계산량이 많습니다 — 필요하면 `RUN_OPTUNA=False`로 끄거나 `N_TRIALS`를 줄이세요).

- 목적함수는 F1·ROC-AUC 조합이 아니라 **F0.5 (또는 0.7×F0.5 + 0.3×PR-AUC)** 를 사용
- 모든 trial의 교차검증도 `StratifiedGroupKFold` + `groups=product_id`로 동일한 `folds`를 사용
- V1과 V2는 동일한 분할 Seed·Fold 정책을 공유

if RUN_OPTUNA:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("Optuna 튜닝을 실행합니다. 모델 수 x 데이터셋 수만큼 반복 학습하므로 계산량이 많을 수 있습니다.")
else:
    print("RUN_OPTUNA=False로 설정되어 있어 튜닝 관련 셀은 건너뜁니다.")

def optuna_objective_score(y_true, y_prob, y_pred):
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    if OPTUNA_OBJECTIVE == "f05_only":
        return f05
    elif OPTUNA_OBJECTIVE == "f05_pr_auc":
        return 0.7 * f05 + 0.3 * pr_auc
    else:
        raise ValueError(f"알 수 없는 OPTUNA_OBJECTIVE: {OPTUNA_OBJECTIVE}")


def cv_objective_for_params(build_model_fn, X, y, groups, folds):
    """주어진 모델 생성 함수로 공통 folds 전체에 대해 학습·평가하고 목적함수 평균 반환"""
    scores = []
    for tr_idx, va_idx in folds:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = build_model_fn(scale_pos_weight)
        model.fit(X_tr, y_tr)
        va_prob = model.predict_proba(X_va)[:, 1]
        va_pred = (va_prob >= 0.5).astype(int)

        scores.append(optuna_objective_score(y_va, va_prob, va_pred))

    return float(np.mean(scores))


def tune_model_with_optuna(model_type, X, y, groups, folds, n_trials=None, random_state=RANDOM_STATE, name=''):
    """model_type: 'LogisticRegression' | 'RandomForest' | 'XGBoost' | 'CatBoost'"""
    n_trials = n_trials or N_TRIALS

    if model_type == 'LogisticRegression':
        def build_model(trial, scale_pos_weight):
            C = trial.suggest_float('C', 0.001, 10.0, log=True)
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            return LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=5000,
                                       class_weight='balanced', random_state=random_state)

    elif model_type == 'RandomForest':
        def build_model(trial, scale_pos_weight):
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 400),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 10),
                max_features=trial.suggest_float('max_features', 0.3, 1.0),
                class_weight={0: 1.0, 1: float(scale_pos_weight)},
                random_state=random_state, n_jobs=-1
            )

    elif model_type == 'XGBoost':
        def build_model(trial, scale_pos_weight):
            return XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 400),
                max_depth=trial.suggest_int('max_depth', 2, 4),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 1.0, 10.0),
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss', random_state=random_state
            )

    elif model_type == 'CatBoost':
        def build_model(trial, scale_pos_weight):
            return CatBoostClassifier(
                iterations=trial.suggest_int('iterations', 100, 400),
                depth=trial.suggest_int('depth', 2, 4),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 3.0, 10.0),
                scale_pos_weight=scale_pos_weight,
                eval_metric='Logloss', random_seed=random_state, verbose=0
            )
    else:
        raise ValueError(f"알 수 없는 model_type: {model_type}")

    def objective(trial):
        def build_fn(scale_pos_weight):
            return build_model(trial, scale_pos_weight)
        return cv_objective_for_params(build_fn, X, y, groups, folds)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[{name}] {model_type} Optuna 최적 목적함수({OPTUNA_OBJECTIVE}) 값: {study.best_value:.4f}")
    print(f"[{name}] {model_type} Best params: {study.best_params}")

    return study

print("Optuna 튜닝 함수 정의 완료")

tuned_params_all = {'V1_전체전성분_V345': {}, 'V2_빈도필터성분_V345': {}}

if RUN_OPTUNA:
    for dataset_name, X_data in [('V1_전체전성분_V345', X_v1), ('V2_빈도필터성분_V345', X_v2)]:
        for model_type in BASE_MODEL_NAMES:
            print(f"\n{'=' * 70}\n[{dataset_name}] {model_type} Optuna 튜닝 시작 (n_trials={N_TRIALS})\n{'=' * 70}")
            study = tune_model_with_optuna(model_type, X_data, y_all, product_id_groups, folds, name=dataset_name)
            tuned_params_all[dataset_name][model_type] = study.best_params

    with open(os.path.join(RESULTS_DIR, 'tuned_params_all.json'), 'w', encoding='utf-8') as f:
        json.dump(tuned_params_all, f, ensure_ascii=False, indent=2)
    print("\nOptuna 튜닝 완료, tuned_params_all.json 저장")
else:
    print("RUN_OPTUNA=False로 설정되어 튜닝을 건너뜁니다.")

## 17. 최종 비교 (baseline vs tuned, V1 vs V2)

if RUN_OPTUNA:
    fold_df_v1_tuned, oof_df_v1_tuned = run_group_cv(
        X_v1, y_all, product_id_groups, 'V1_전체전성분_V345', folds, BASE_MODEL_NAMES,
        tuned_params_dict=tuned_params_all['V1_전체전성분_V345'], stage_label='tuned'
    )
    fold_df_v2_tuned, oof_df_v2_tuned = run_group_cv(
        X_v2, y_all, product_id_groups, 'V2_빈도필터성분_V345', folds, BASE_MODEL_NAMES,
        tuned_params_dict=tuned_params_all['V2_빈도필터성분_V345'], stage_label='tuned'
    )

    fold_df_tuned = pd.concat([fold_df_v1_tuned, fold_df_v2_tuned], ignore_index=True)
    oof_df_tuned = pd.concat([oof_df_v1_tuned, oof_df_v2_tuned], ignore_index=True)

    fold_df_combined = pd.concat([fold_df_baseline, fold_df_tuned], ignore_index=True)
    oof_df_combined = pd.concat([oof_df_baseline, oof_df_tuned], ignore_index=True)
else:
    fold_df_combined = fold_df_baseline.copy()
    oof_df_combined = oof_df_baseline.copy()

fold_df_combined.to_csv(os.path.join(RESULTS_DIR, 'fold_metrics.csv'), index=False, encoding='utf-8-sig')
oof_df_combined.to_csv(os.path.join(RESULTS_DIR, 'oof_predictions.csv'), index=False, encoding='utf-8-sig')

threshold_results = fold_df_combined[[
    'dataset', 'model', 'stage', 'fold', 'threshold', 'Precision', 'Recall', 'F0.5', 'F1',
    'PR_AUC', 'ROC_AUC', 'Accuracy', 'Precision_at_25', 'Recall_at_25', 'Lift_at_25'
]]
threshold_results.to_csv(os.path.join(RESULTS_DIR, 'threshold_results.csv'), index=False, encoding='utf-8-sig')

print(f"fold_metrics.csv ({len(fold_df_combined)}행), oof_predictions.csv ({len(oof_df_combined)}행), "
      f"threshold_results.csv 저장 완료")

summary_combined = summarize_fold_metrics(fold_df_combined)

summary_combined_f05 = summary_combined.sort_values(
    ['F05_mean', 'Precision_mean', 'PR_AUC_mean'], ascending=False
).reset_index(drop=True)
summary_combined_precision = summary_combined.sort_values(
    ['Precision_mean', 'F05_mean', 'PR_AUC_mean'], ascending=False
).reset_index(drop=True)

print("[F0.5 기준 최종 비교 (baseline + tuned)]")
display(summary_combined_f05)
print("\n[Precision 기준 최종 비교]")
display(summary_combined_precision)

summary_combined_f05.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_summary.csv'), index=False, encoding='utf-8-sig')

best_row = summary_combined_f05.iloc[0]
best_model_info = {
    'dataset': best_row['데이터셋'], 'model': best_row['모델'], 'stage': best_row['단계'],
    'F05_mean': float(best_row['F05_mean']), 'Precision_mean': float(best_row['Precision_mean']),
    'PR_AUC_mean': float(best_row['PR_AUC_mean']), 'ROC_AUC_mean': float(best_row['ROC_AUC_mean']),
    'Threshold_mean': float(best_row['Threshold_mean']),
}
with open(os.path.join(RESULTS_DIR, 'best_model_info.json'), 'w', encoding='utf-8') as f:
    json.dump(best_model_info, f, ensure_ascii=False, indent=2)

print("\n최종 최우수 모델:", best_model_info)

## 18. OOF 예측 기준 종합 성능

oof_summary_records = []
for (ds, mdl, stg), grp in oof_df_combined.groupby(['dataset', 'model', 'stage']):
    y_true = grp['high_risk'].values
    y_prob = grp['predicted_probability'].values
    y_pred = grp['predicted_label'].values
    m = evaluate_predictions(y_true, y_prob, y_pred)
    m.update({'dataset': ds, 'model': mdl, 'stage': stg})
    oof_summary_records.append(m)

oof_summary_df = pd.DataFrame(oof_summary_records).sort_values('F0.5', ascending=False).reset_index(drop=True)
print("[OOF 기준 데이터셋x모델x단계별 종합 성능]")
display(oof_summary_df)
oof_summary_df.to_csv(os.path.join(RESULTS_DIR, 'oof_summary.csv'), index=False, encoding='utf-8-sig')

## 19. 필수 시각화

plot_df = summary_combined_f05.copy()
plot_labels = plot_df['데이터셋'] + '\n' + plot_df['모델'] + '(' + plot_df['단계'] + ')'

# 1. 모델별 Precision / F0.5 / Recall / PR-AUC 비교 막대그래프
fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(len(plot_df))
width = 0.2
ax.bar(x - 1.5 * width, plot_df['Precision_mean'], width, label='Precision')
ax.bar(x - 0.5 * width, plot_df['F05_mean'], width, label='F0.5')
ax.bar(x + 0.5 * width, plot_df['Recall_mean'], width, label='Recall')
ax.bar(x + 1.5 * width, plot_df['PR_AUC_mean'], width, label='PR-AUC')
ax.set_xticks(x)
ax.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylim(0, 1)
ax.set_title('데이터셋 x 모델 x 단계별 Precision / F0.5 / Recall / PR-AUC 비교')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig1_model_metric_comparison.png'), dpi=150)
plt.show()

# 2. V1 vs V2 비교 (F0.5 기준, 모델별)
fig, ax = plt.subplots(figsize=(10, 5))
pivot_v1v2 = plot_df.pivot_table(index='모델', columns='데이터셋', values='F05_mean', aggfunc='mean')
pivot_v1v2.plot(kind='bar', ax=ax)
ax.set_title('V1 vs V2 - 모델별 F0.5 평균 비교')
ax.set_ylabel('F0.5 (mean)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig2_v1_vs_v2.png'), dpi=150)
plt.show()

# 3. 최종 우수 모델의 OOF Precision-Recall Curve
best_oof = oof_df_combined[
    (oof_df_combined['dataset'] == best_model_info['dataset']) &
    (oof_df_combined['model'] == best_model_info['model']) &
    (oof_df_combined['stage'] == best_model_info['stage'])
]
prec_curve, rec_curve, _ = precision_recall_curve(best_oof['high_risk'], best_oof['predicted_probability'])

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(rec_curve, prec_curve, color='darkorange', lw=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f"최종 우수 모델 OOF Precision-Recall Curve\n"
             f"({best_model_info['dataset']} - {best_model_info['model']} - {best_model_info['stage']})")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig3_best_model_pr_curve.png'), dpi=150)
plt.show()

# 4. 최종 우수 모델 Confusion Matrix (OOF)
best_pred_label = (best_oof['predicted_probability'] >= best_oof['selected_threshold']).astype(int)
cm = confusion_matrix(best_oof['high_risk'], best_pred_label)

fig, ax = plt.subplots(figsize=(5, 5))