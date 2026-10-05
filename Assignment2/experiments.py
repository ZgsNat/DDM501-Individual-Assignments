"""
Experiment Tracking & Matrix Execution Module
Course: DDM501 - Individual Assignment 2
Executes the planned 10-configuration experiment matrix, measures latency,
computes metrics, and exports comparison results.
"""

import time
import json
import os
import platform
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn import __version__ as sklearn_version

from utils import seed_everything, calculate_metrics, calculate_disparate_impact_ratio
from pipeline import CreditRiskPipeline
from config import settings


EXPERIMENT_CONFIGS = [
    {
        "exp_id": "01",
        "name": "Logistic Regression (Baseline)",
        "model_cls": LogisticRegression,
        "params": {"C": 1.0, "max_iter": 1000, "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "None",
        "status": "Baseline"
    },
    {
        "exp_id": "02",
        "name": "Logistic Regression (Balanced)",
        "model_cls": LogisticRegression,
        "params": {"C": 0.1, "class_weight": "balanced", "max_iter": 1000, "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "Balanced",
        "status": "Evaluated"
    },
    {
        "exp_id": "03",
        "name": "Random Forest (Shallow)",
        "model_cls": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "None",
        "status": "Evaluated"
    },
    {
        "exp_id": "04",
        "name": "Random Forest (Deep Balanced)",
        "model_cls": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced", "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "Balanced",
        "status": "Evaluated"
    },
    {
        "exp_id": "05",
        "name": "Gradient Boosting (GBDT)",
        "model_cls": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4, "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "None",
        "status": "Evaluated"
    },
    {
        "exp_id": "06",
        "name": "Gradient Boosting (Deep)",
        "model_cls": GradientBoostingClassifier,
        "params": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 6, "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "None",
        "status": "Evaluated"
    },
    {
        "exp_id": "07",
        "name": "HistGradientBoosting",
        "model_cls": HistGradientBoostingClassifier,
        "params": {"max_iter": 100, "learning_rate": 0.05, "max_depth": 5, "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "None",
        "status": "Evaluated"
    },
    {
        "exp_id": "08",
        "name": "Hist GBDT (Balanced)",
        "model_cls": HistGradientBoostingClassifier,
        "params": {"max_iter": 150, "learning_rate": 0.05, "max_depth": 6, "class_weight": "balanced", "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "Balanced",
        "status": "Evaluated"
    },
    {
        "exp_id": "09",
        "name": "HistGradientBoosting (tuned)",
        "model_cls": HistGradientBoostingClassifier,
        "params": {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 31, "class_weight": "balanced", "random_state": 42},
        "features": "Raw (23)",
        "class_weight": "Balanced",
        "status": "Evaluated"
    },
    {
        "exp_id": "10",
        "name": "Hist GBDT + Financial Ratios",
        "model_cls": HistGradientBoostingClassifier,
        "params": {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 31, "class_weight": "balanced", "random_state": 42},
        "features": "Raw + Ratios",
        "class_weight": "Balanced",
        "status": "Evaluated"
    }
]


def run_experiment_matrix() -> pd.DataFrame:
    seed_everything(settings.data.random_seed)
    pipeline = CreditRiskPipeline(random_seed=settings.data.random_seed)
    df = pipeline.ingest_data()
    df = pipeline.validate_data_gate(df)

    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=1.0 - settings.data.train_split,
        random_state=settings.data.random_seed,
        stratify=y,
    )

    preprocessor = pipeline.build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # For experiment 10, engineer financial ratios
    util_train = (X_train["BILL_AMT1"] / np.maximum(X_train["LIMIT_BAL"], 1.0)).to_numpy().reshape(-1, 1)
    util_val = (X_val["BILL_AMT1"] / np.maximum(X_val["LIMIT_BAL"], 1.0)).to_numpy().reshape(-1, 1)
    pay_ratio_train = (X_train["PAY_AMT1"] / np.maximum(np.abs(X_train["BILL_AMT2"]), 1.0)).to_numpy().reshape(-1, 1)
    pay_ratio_val = (X_val["PAY_AMT1"] / np.maximum(np.abs(X_val["BILL_AMT2"]), 1.0)).to_numpy().reshape(-1, 1)

    X_train_proc_ratios = np.hstack([X_train_proc, util_train, pay_ratio_train])
    X_val_proc_ratios = np.hstack([X_val_proc, util_val, pay_ratio_val])

    results = []

    print("\nExecuting Planned 10-Configuration Experiment Matrix...")
    print("-" * 80)

    for cfg in EXPERIMENT_CONFIGS:
        model = cfg["model_cls"](**cfg["params"])

        X_tr = X_train_proc_ratios if cfg["exp_id"] == "10" else X_train_proc
        X_vl = X_val_proc_ratios if cfg["exp_id"] == "10" else X_val_proc

        model.fit(X_tr, y_train)

        # Warm up before measuring p95 single-row model-call latency.
        for _ in range(10):
            model.predict_proba(X_vl[:1])
        latencies = []
        for _ in range(200):
            t0 = time.perf_counter()
            _ = model.predict_proba(X_vl[:1])
            latencies.append((time.perf_counter() - t0) * 1000.0)
        p95_latency = float(np.percentile(latencies, 95))

        probs = model.predict_proba(X_vl)[:, 1]
        preds = (probs >= settings.model.review_threshold).astype(int)

        metrics = calculate_metrics(
            y_val.to_numpy(), probs, threshold=settings.model.review_threshold
        )
        dir_score = calculate_disparate_impact_ratio(preds, X_val["SEX"])
        passed_gates = (
            metrics["roc_auc"] >= settings.model.min_roc_auc_gate
            and settings.fairness.dir_min_threshold <= dir_score <= settings.fairness.dir_max_threshold
        )

        results.append({
            "Exp #": cfg["exp_id"],
            "Model Architecture": cfg["name"],
            "Features": cfg["features"],
            "Class Weight": cfg["class_weight"],
            "ROC-AUC": metrics["roc_auc"],
            "F1-Score": metrics["f1_score"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Brier": metrics["brier_score"],
            "Latency p95 (ms)": round(p95_latency, 2),
            "DIR": dir_score,
            "Status": (
                "Baseline" if cfg["exp_id"] == "01"
                else "Eligible for review" if passed_gates
                else "Rejected by offline gate"
            )
        })

    res_df = pd.DataFrame(results)
    eligible = res_df[res_df["Status"] == "Eligible for review"]
    if not eligible.empty:
        best_idx = eligible["ROC-AUC"].idxmax()
        res_df.loc[best_idx, "Status"] = "Top offline candidate"
    res_df.to_csv("experiment_results.csv", index=False)
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "configured CSV" if settings.data.train_data_path else "synthetic generator",
        "rows": len(df),
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "default_rate": round(float(y.mean()), 4),
        "default_count": int(y.sum()),
        "seed": settings.data.random_seed,
        "validation_fraction": round(1.0 - settings.data.train_split, 4),
        "classification_threshold": settings.model.review_threshold,
        "minimum_roc_auc_gate": settings.model.min_roc_auc_gate,
        "dir_bounds": [
            settings.fairness.dir_min_threshold,
            settings.fairness.dir_max_threshold,
        ],
        "latency_measurement": (
            "p95 of 200 warmed-up single-row predict_proba calls; model-only, "
            "excluding preprocessing, API, database, and network"
        ),
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn_version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    with open("experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return res_df


if __name__ == "__main__":
    df_res = run_experiment_matrix()
    try:
        print("\n" + df_res.to_markdown(index=False))
    except Exception:
        print("\n" + df_res.to_string(index=False))
