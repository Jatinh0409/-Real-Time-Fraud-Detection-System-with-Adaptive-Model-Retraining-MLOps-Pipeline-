"""
train_model.py
Trains an XGBoost fraud classifier with proper handling of class imbalance,
evaluates with metrics that actually matter for fraud (not accuracy), and
saves the model + preprocessing pipeline for serving.

Usage:
    python train_model.py --data ../data/transactions.csv --out ../models/model.pkl
"""

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "hour_of_day", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]
CATEGORICAL_FEATURES = ["type"]
TARGET = "isFraud"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features that matter a lot for fraud signal."""
    df = df.copy()
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_drained"] = (df["newbalanceOrig"] == 0).astype(int)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    return df


EXTRA_NUMERIC = ["balance_diff_orig", "balance_diff_dest", "orig_drained", "amount_to_balance_ratio"]


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES + EXTRA_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,  # handles class imbalance directly
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def main(data_path: str, out_path: str, metrics_path: str):
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + EXTRA_NUMERIC + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # scale_pos_weight = (# negative / # positive) — standard XGBoost imbalance handling
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    pipeline = build_pipeline(scale_pos_weight)

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)  # more meaningful than ROC-AUC for rare-class problems
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_fraud_class": round(report["1"]["precision"], 4),
        "recall_fraud_class": round(report["1"]["recall"], 4),
        "f1_fraud_class": round(report["1"]["f1-score"], 4),
        "confusion_matrix": cm,
        "train_time_seconds": round(train_time, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fraud_rate_train": round(float(y_train.mean()), 4),
    }

    print(json.dumps(metrics, indent=2))

    joblib.dump(pipeline, out_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {out_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/transactions.csv")
    parser.add_argument("--out", type=str, default="../models/model.pkl")
    parser.add_argument("--metrics_out", type=str, default="../models/metrics.json")
    args = parser.parse_args()
    main(args.data, args.out, args.metrics_out)
