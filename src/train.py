import os, json, math
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

DATA_PATH = "data/processed/training_data.csv"
MODEL_PATH = "models/race_predictor.pkl"
METRICS_PATH = "reports/metrics.json"

def safe_import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None

def build_Xy(df):
    # choose features that exist
    base = ["grid","qual_position","driver_points_to_date","constructor_points_to_date","points"]
    feats = [c for c in base if c in df.columns]
    X = df[feats].copy()
    y = df["win"].astype(int).values
    # groups for split (avoid leakage across the same race)
    groups = df["raceId"].values if "raceId" in df.columns else None
    return X, y, feats, groups

def summarize(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run `python -m src.data_prep` first.")

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X, y, feats, groups = build_Xy(df)

    # quick train/test split that respects group if present
    if groups is not None:
        # split by groups: put whole races into test
        rng = np.random.RandomState(42)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        cut = int(0.8 * len(unique_groups))
        train_groups = set(unique_groups[:cut])
        mask_tr = np.array([g in train_groups for g in groups])
        Xtr, Xte = X[mask_tr], X[~mask_tr]
        ytr, yte = y[mask_tr], y[~mask_tr]
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # compute imbalance ratio to help some models
    pos = (ytr == 1).sum()
    neg = (ytr == 0).sum()
    scale_pos_weight = (neg / pos) if pos else 1.0

    # Define candidate models (each returns a calibrated estimator)
    candidates = []

    # 1) Logistic Regression (balanced) + scaling + calibration
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
    ])
    candidates.append(("logreg_balanced", CalibratedClassifierCV(lr, cv=3, method="isotonic")))

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    candidates.append(("random_forest", CalibratedClassifierCV(rf, cv=3, method="isotonic")))

    # 3) Gradient Boosting (sklearn)
    gb = GradientBoostingClassifier(random_state=42)
    candidates.append(("grad_boosting", CalibratedClassifierCV(gb, cv=3, method="isotonic")))

    # 4) XGBoost (optional)
    xgb = safe_import_xgb()
    if xgb is not None:
        xgb_model = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="auc",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )
        candidates.append(("xgboost", xgb_model))  # XGB already outputs probabilities; usually well-calibrated

    results = []
    best_name, best_model, best_auc = None, None, -1.0

    for name, model in candidates:
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        m = summarize(yte, proba)
        m["model"] = name
        m["features_used"] = feats
        results.append(m)
        if m["roc_auc"] > best_auc:
            best_auc, best_name, best_model, best_metrics = m["roc_auc"], name, model, m

    # save the best
    dump(best_model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "selected_model": best_name,
            "metrics": best_metrics,
            "all_models": results
        }, f, indent=2)

    print(f"Selected model: {best_name}")
    print("Best metrics:", best_metrics)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")

if __name__ == "__main__":
    main()
