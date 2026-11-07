"""Smoke tests of functions extracted from the actual published poster notebook.

Synthetic data only; passing tests do not reproduce scientific performance.
"""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "05_poster_final_formula_group.ipynb"

def load_notebook_function(name):
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in raw["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        if f"def {name}(" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                ns = {"np": np, "pd": pd, "GroupShuffleSplit": GroupShuffleSplit}
                module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
                exec(compile(module, str(NOTEBOOK), "exec"), ns)
                return ns[name]
    raise AssertionError(f"{name} not present in the published poster notebook")

def test_prelaunch_features_exclude_review_and_target_columns():
    select = load_notebook_function("select_feature_columns")
    frame = pd.DataFrame(columns=[
        "product_id", "product_name", "formula_group", "피부타입",
        "high_risk", "y_total_reviews", "y_negative_rate_raw",
        "y_negative_rate_smoothed", "review_text", "review_count",
        "skin_건성", "v1_sample", "v2_sample", "v3_count_humectant",
        "v3_ratio_bad", "v4_sensory_count", "v5_theme_진정",
    ])
    v1, v2 = select(frame, "V1"), select(frame, "V2")
    assert v1 == ["skin_건성", "v1_sample", "v3_count_humectant", "v4_sensory_count", "v5_theme_진정"]
    assert v2 == ["skin_건성", "v2_sample", "v3_count_humectant", "v4_sensory_count", "v5_theme_진정"]
    assert all(not x.startswith(("y_", "review_", "v3_ratio_")) for x in v1 + v2)

def test_group_holdout_is_disjoint_and_deterministic():
    split = load_notebook_function("make_balanced_group_split")
    groups = pd.Series(np.repeat([f"formula-{i}" for i in range(50)], 4))
    labels = pd.Series(np.tile([0, 0, 0, 1], 50))
    train, test = split(labels, groups, n_candidates=30, random_state=42)
    train2, test2 = split(labels, groups, n_candidates=30, random_state=42)
    assert np.array_equal(train, train2)
    assert np.array_equal(test, test2)
    assert len(train) + len(test) == len(labels)
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))

def test_cv_and_holdout_groups_remain_separate():
    split = load_notebook_function("make_balanced_group_split")
    groups = pd.Series(np.repeat([f"formula-{i}" for i in range(60)], 4))
    labels = pd.Series(np.tile([0, 0, 1, 1], 60))
    train, test = split(labels, groups, n_candidates=20, random_state=7)
    gtrain = groups.iloc[train].reset_index(drop=True)
    ytrain = labels.iloc[train].reset_index(drop=True)
    gtest = set(groups.iloc[test])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fit, val in cv.split(np.zeros((len(ytrain), 1)), ytrain, gtrain):
        fit_groups, val_groups = set(gtrain.iloc[fit]), set(gtrain.iloc[val])
        assert fit_groups.isdisjoint(val_groups)
        assert fit_groups.isdisjoint(gtest)
        assert val_groups.isdisjoint(gtest)

def test_same_formula_across_two_products_stays_in_one_partition():
    """A formula shared by different products must not cross the hold-out boundary."""
    split = load_notebook_function("make_balanced_group_split")
    formula = pd.Series(np.repeat([f"formula-{i}" for i in range(40)], 6))
    products = pd.Series([
        f"product-{i}-{j}"
        for i in range(40) for j in range(2) for _ in range(3)
    ])
    labels = pd.Series(np.tile([0, 1, 0, 0, 1, 1], 40))
    fit, holdout = split(labels, formula, n_candidates=30, random_state=42)
    assert len(set(formula.iloc[fit]) & set(formula.iloc[holdout])) == 0
    for group in formula.unique():
        group_indices = set(np.flatnonzero(formula.to_numpy() == group))
        assert group_indices.issubset(set(fit)) or group_indices.issubset(set(holdout))
    assert len(products.iloc[fit]) + len(products.iloc[holdout]) == len(products)
