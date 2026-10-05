"""
Utility Module: Reproducibility, Metrics, Fair Lending & Data Generation
Course: DDM501 - Individual Assignment 2
"""

import random
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)


def seed_everything(seed: int = 42) -> None:
    """
    Seed Python and NumPy generators used by the prototype.

    Estimators and splitters must also receive explicit seeds. For a fixed
    Python hash seed, configure PYTHONHASHSEED before starting the process.
    """
    random.seed(seed)
    np.random.seed(seed)


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.30
) -> Dict[str, float]:
    """
    Computes primary and secondary ML performance metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def calculate_disparate_impact_ratio(
    y_pred: np.ndarray,
    protected_attr: pd.Series,
    unprivileged_group: Any = 2,
    privileged_group: Any = 1
) -> float:
    """
    Calculates the Disparate Impact Ratio (DIR) under the Four-Fifths rule:
    DIR = Approval_Rate(Unprivileged) / Approval_Rate(Privileged)
    where Approval is y_pred == 0 (non-default).
    """
    # Approval means y_pred == 0
    approved = (y_pred == 0).astype(int)
    
    unprivileged_mask = (protected_attr == unprivileged_group)
    privileged_mask = (protected_attr == privileged_group)

    if len(y_pred) != len(protected_attr):
        raise ValueError("Predictions and protected attributes must have equal lengths.")
    if not unprivileged_mask.any() or not privileged_mask.any():
        raise ValueError("DIR is undefined unless both comparison groups are present.")

    rate_unprivileged = approved[unprivileged_mask].mean()
    rate_privileged = approved[privileged_mask].mean()
    if rate_privileged == 0.0:
        raise ValueError("DIR is undefined when the privileged-group approval rate is zero.")
    return round(float(rate_unprivileged / rate_privileged), 4)


def generate_synthetic_credit_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates illustrative synthetic rows matching the UCI dataset's feature schema.

    This generator does not reproduce or contain observations from the UCI dataset.
    """
    np.random.seed(seed)
    
    limit_bal = np.random.choice([20000, 50000, 100000, 200000, 300000, 500000], size=n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
    sex = np.random.choice([1, 2], size=n_samples, p=[0.40, 0.60])
    education = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.45, 0.15, 0.05])
    marriage = np.random.choice([1, 2, 3], size=n_samples, p=[0.45, 0.50, 0.05])
    age = np.random.randint(21, 65, size=n_samples)
    
    # Repayment status (-1 to 3)
    pay_0 = np.random.choice([-1, 0, 1, 2, 3], size=n_samples, p=[0.30, 0.50, 0.10, 0.07, 0.03])
    pay_2 = np.random.choice([-1, 0, 1, 2], size=n_samples, p=[0.35, 0.52, 0.08, 0.05])
    pay_3 = np.random.choice([-1, 0, 1, 2], size=n_samples, p=[0.35, 0.52, 0.08, 0.05])
    pay_4 = np.random.choice([-1, 0, 1, 2], size=n_samples, p=[0.35, 0.52, 0.08, 0.05])
    pay_5 = np.random.choice([-1, 0, 1, 2], size=n_samples, p=[0.35, 0.52, 0.08, 0.05])
    pay_6 = np.random.choice([-1, 0, 1, 2], size=n_samples, p=[0.35, 0.52, 0.08, 0.05])
    
    # Bill amounts
    bill_amt1 = np.random.uniform(500, 150000, size=n_samples)
    bill_amt2 = bill_amt1 * np.random.uniform(0.8, 1.1, size=n_samples)
    bill_amt3 = bill_amt2 * np.random.uniform(0.8, 1.1, size=n_samples)
    bill_amt4 = bill_amt3 * np.random.uniform(0.8, 1.1, size=n_samples)
    bill_amt5 = bill_amt4 * np.random.uniform(0.8, 1.1, size=n_samples)
    bill_amt6 = bill_amt5 * np.random.uniform(0.8, 1.1, size=n_samples)
    
    # Payment amounts
    pay_amt1 = bill_amt1 * np.random.uniform(0.05, 0.60, size=n_samples)
    pay_amt2 = bill_amt2 * np.random.uniform(0.05, 0.60, size=n_samples)
    pay_amt3 = bill_amt3 * np.random.uniform(0.05, 0.60, size=n_samples)
    pay_amt4 = bill_amt4 * np.random.uniform(0.05, 0.60, size=n_samples)
    pay_amt5 = bill_amt5 * np.random.uniform(0.05, 0.60, size=n_samples)
    pay_amt6 = bill_amt6 * np.random.uniform(0.05, 0.60, size=n_samples)
    
    # Latent probability of default
    z = -1.8 + 0.65 * pay_0 + 0.40 * (bill_amt1 / limit_bal) - 0.50 * (pay_amt1 / np.maximum(bill_amt1, 1))
    prob = 1.0 / (1.0 + np.exp(-z))
    default_label = (np.random.rand(n_samples) < prob).astype(int)

    data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": np.round(bill_amt1, 2),
        "BILL_AMT2": np.round(bill_amt2, 2),
        "BILL_AMT3": np.round(bill_amt3, 2),
        "BILL_AMT4": np.round(bill_amt4, 2),
        "BILL_AMT5": np.round(bill_amt5, 2),
        "BILL_AMT6": np.round(bill_amt6, 2),
        "PAY_AMT1": np.round(pay_amt1, 2),
        "PAY_AMT2": np.round(pay_amt2, 2),
        "PAY_AMT3": np.round(pay_amt3, 2),
        "PAY_AMT4": np.round(pay_amt4, 2),
        "PAY_AMT5": np.round(pay_amt5, 2),
        "PAY_AMT6": np.round(pay_amt6, 2),
        "default": default_label
    }
    return pd.DataFrame(data)
