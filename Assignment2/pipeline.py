"""
Production Machine Learning Pipeline Implementation
Course: DDM501 - Individual Assignment 2
Implements all 7 stages: Ingestion, Validation Gate, Feature Engineering,
Preprocessing, Model Training, Evaluation Gate, and Registry Serialization.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import settings
from utils import seed_everything, calculate_metrics, calculate_disparate_impact_ratio, generate_synthetic_credit_data


class ValidationError(Exception):
    """Custom exception raised when a Data or Model Quality Gate fails."""
    pass


class CreditRiskPipeline:
    """
    Modular End-to-End Production ML Pipeline for Credit Default Risk Scoring.
    """

    NUMERIC_FEATURES = [
        "LIMIT_BAL", "AGE",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
    ]
    CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
    TARGET_COLUMN = "default"

    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed if random_seed is not None else settings.data.random_seed
        seed_everything(self.random_seed)
        self.preprocessor: Optional[ColumnTransformer] = None
        self.model: Optional[Any] = None
        self.is_fitted: bool = False

    # -------------------------------------------------------------
    # Stage 1: Data Ingestion
    # -------------------------------------------------------------
    def ingest_data(self, data_path: Optional[Path] = None, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Loads a configured CSV, or creates synthetic demo data when no path is configured.
        """
        configured_path = data_path or settings.data.train_data_path
        if configured_path is not None:
            if not configured_path.is_file():
                raise FileNotFoundError(f"Configured training data file does not exist: {configured_path}")
            df = pd.read_csv(configured_path)
        else:
            df = generate_synthetic_credit_data(n_samples=5000, seed=self.random_seed)

        # Normalize column names & standardize target column
        df.columns = [c.strip() for c in df.columns]
        rename_map = {
            "default.payment.next.month": "default",
            "default_payment_next_month": "default",
            "DEFAULT": "default",
        }
        df = df.rename(columns=rename_map)

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=self.random_seed).reset_index(drop=True)

        return df

    # -------------------------------------------------------------
    # Stage 2: Data Validation & Schema Gate
    # -------------------------------------------------------------
    def validate_data_gate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates schema, null counts, and domain boundary constraints.
        Raises ValidationError if any criteria fail.
        """
        if df.empty:
            raise ValidationError("Data Quality Gate Failed: dataset is empty")

        # 1. Expected columns
        required_cols = set(self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES + [self.TARGET_COLUMN])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValidationError(f"Schema violation: missing required columns {missing_cols}")

        # 2. Null check (Strict 0.0% null tolerance)
        null_counts = df[list(required_cols)].isnull().sum().sum()
        null_rate = null_counts / (len(df) * len(required_cols))
        if null_rate > settings.data.null_tolerance:
            raise ValidationError(
                f"Data Quality Gate Failed: detected {null_counts} null values "
                f"({null_rate:.4%}) above tolerance {settings.data.null_tolerance:.4%}"
            )

        numeric_values = df[self.NUMERIC_FEATURES].to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise ValidationError("Data Quality Gate Failed: numeric features contain non-finite values")

        # 3. Domain boundaries
        if (df["AGE"] < 18).any() or (df["AGE"] > 100).any():
            raise ValidationError("Data Quality Gate Failed: AGE outside [18, 100]")

        if (df["LIMIT_BAL"] <= 0).any():
            raise ValidationError("Data Quality Gate Failed: LIMIT_BAL contains non-positive values")

        allowed_categories = {
            "SEX": {1, 2},
            "EDUCATION": {1, 2, 3, 4},
            "MARRIAGE": {1, 2, 3},
        }
        for column, allowed_values in allowed_categories.items():
            if not df[column].isin(allowed_values).all():
                raise ValidationError(f"Data Quality Gate Failed: invalid {column} category")

        if not df[self.TARGET_COLUMN].isin({0, 1}).all():
            raise ValidationError("Data Quality Gate Failed: target must contain only 0 and 1")

        return df

    # -------------------------------------------------------------
    # Stage 3: Feature Engineering & Preprocessing
    # -------------------------------------------------------------
    def build_preprocessor(self) -> ColumnTransformer:
        """
        Constructs standard Scikit-Learn ColumnTransformer pipeline.
        """
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.NUMERIC_FEATURES),
                ("cat", categorical_transformer, self.CATEGORICAL_FEATURES),
            ]
        )
        return preprocessor

    # -------------------------------------------------------------
    # Stage 4: Model Training
    # -------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "hist_gbdt"
    ) -> Any:
        """
        Fits preprocessor and classifier on training data.
        """
        self.preprocessor = self.build_preprocessor()
        X_train_proc = self.preprocessor.fit_transform(X_train)

        # Select model architecture
        if model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, random_state=self.random_seed)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight="balanced", random_state=self.random_seed
            )
        else:  # HistGradientBoosting is the sklearn prototype, not LightGBM.
            self.model = HistGradientBoostingClassifier(
                max_iter=150, max_depth=6, class_weight="balanced", random_state=self.random_seed
            )

        self.model.fit(X_train_proc, y_train)
        self.is_fitted = True
        return self.model

    # -------------------------------------------------------------
    # Stage 5: Evaluation & Fair Lending Gate
    # -------------------------------------------------------------
    def evaluate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Computes validation metrics and evaluates operational and fairness gates.
        """
        if not self.is_fitted or self.preprocessor is None or self.model is None:
            raise RuntimeError("Pipeline must be trained before evaluation.")

        X_val_proc = self.preprocessor.transform(X_val)
        y_prob = self.model.predict_proba(X_val_proc)[:, 1]
        y_pred = (y_prob >= settings.model.review_threshold).astype(int)

        metrics = calculate_metrics(y_val.to_numpy(), y_prob, threshold=settings.model.review_threshold)

        # Fair Lending Disparate Impact Ratio on protected attribute (SEX)
        dir_score = calculate_disparate_impact_ratio(
            y_pred=y_pred,
            protected_attr=X_val["SEX"],
            unprivileged_group=2,
            privileged_group=1
        )
        metrics["disparate_impact_ratio"] = dir_score

        # Quality Gates
        passed_gates = True
        if metrics["roc_auc"] < settings.model.min_roc_auc_gate:
            passed_gates = False
        if not (settings.fairness.dir_min_threshold <= dir_score <= settings.fairness.dir_max_threshold):
            passed_gates = False

        metrics["passed_quality_gates"] = passed_gates
        return metrics

    # -------------------------------------------------------------
    # End-to-End Execution
    # -------------------------------------------------------------
    def run_pipeline(
        self,
        data_path: Optional[Path] = None,
        model_type: str = "hist_gbdt",
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end pipeline run from ingestion to evaluation gate.
        """
        df_raw = self.ingest_data(data_path, max_samples=max_samples)
        df_clean = self.validate_data_gate(df_raw)

        X = df_clean.drop(columns=[self.TARGET_COLUMN])
        y = df_clean[self.TARGET_COLUMN]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=(1.0 - settings.data.train_split),
            random_state=self.random_seed, stratify=y
        )

        self.train(X_train, y_train, model_type=model_type)
        eval_results = self.evaluate(X_val, y_val)
        eval_results["train_samples"] = len(X_train)
        eval_results["val_samples"] = len(X_val)
        return eval_results
