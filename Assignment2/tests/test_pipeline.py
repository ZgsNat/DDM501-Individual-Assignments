"""
Automated Pytest Suite for Assignment 2 ML Pipeline & MLOps Infrastructure
"""

from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from config import PipelineConfig
from pipeline import CreditRiskPipeline, ValidationError
from utils import seed_everything, calculate_metrics, calculate_disparate_impact_ratio, generate_synthetic_credit_data
import airflow_dag


@pytest.fixture
def clean_dataset():
    seed_everything(42)
    return generate_synthetic_credit_data(n_samples=1000, seed=42)


def test_data_ingestion(clean_dataset):
    pipeline = CreditRiskPipeline()
    df = pipeline.ingest_data()
    assert len(df) > 0
    assert "default" in df.columns
    assert "LIMIT_BAL" in df.columns


def test_data_validation_gate_success(clean_dataset):
    pipeline = CreditRiskPipeline()
    validated_df = pipeline.validate_data_gate(clean_dataset)
    assert len(validated_df) == len(clean_dataset)


def test_data_validation_gate_null_failure(clean_dataset):
    pipeline = CreditRiskPipeline()
    corrupt_df = clean_dataset.copy()
    corrupt_df.loc[0, "AGE"] = np.nan
    with pytest.raises(ValidationError, match="null values"):
        pipeline.validate_data_gate(corrupt_df)


def test_data_validation_gate_age_boundary(clean_dataset):
    pipeline = CreditRiskPipeline()
    corrupt_df = clean_dataset.copy()
    corrupt_df.loc[0, "AGE"] = 12  # Underage borrower
    with pytest.raises(ValidationError, match="AGE outside"):
        pipeline.validate_data_gate(corrupt_df)


def test_data_validation_gate_negative_limit(clean_dataset):
    pipeline = CreditRiskPipeline()
    corrupt_df = clean_dataset.copy()
    corrupt_df.loc[0, "LIMIT_BAL"] = -1000.0
    with pytest.raises(ValidationError, match="non-positive"):
        pipeline.validate_data_gate(corrupt_df)


def test_data_validation_gate_invalid_target(clean_dataset):
    pipeline = CreditRiskPipeline()
    corrupt_df = clean_dataset.copy()
    corrupt_df.loc[0, "default"] = 2
    with pytest.raises(ValidationError, match="target"):
        pipeline.validate_data_gate(corrupt_df)


def test_explicit_missing_dataset_fails_instead_of_using_synthetic_data():
    pipeline = CreditRiskPipeline()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        pipeline.ingest_data(data_path=Path("missing-training-data.csv"))


def test_config_loads_nested_yaml_and_applies_environment_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "environment: test\nstorage:\n  mlflow_tracking_uri: http://yaml:5000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_ENV", "test-override")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
    monkeypatch.setenv("DATA_PATH", "/tmp/training.csv")

    settings = PipelineConfig.load(config_path)

    assert settings.environment == "test-override"
    assert settings.storage.mlflow_tracking_uri == "http://env:5000"
    assert settings.data.train_data_path == Path("/tmp/training.csv")


def test_preprocessor_dimensionality(clean_dataset):
    pipeline = CreditRiskPipeline()
    preprocessor = pipeline.build_preprocessor()
    X = clean_dataset.drop(columns=["default"])
    X_proc = preprocessor.fit_transform(X)
    assert X_proc.shape[0] == len(clean_dataset)
    # 20 numerics + one-hot categories (SEX: 2, EDU: 4, MAR: 3) = 29-30 cols
    assert X_proc.shape[1] >= 29


def test_pipeline_end_to_end():
    pipeline = CreditRiskPipeline(random_seed=42)
    results = pipeline.run_pipeline(model_type="hist_gbdt", max_samples=1000)
    assert "roc_auc" in results
    assert "f1_score" in results
    assert "disparate_impact_ratio" in results
    assert 0.0 <= results["roc_auc"] <= 1.0
    assert np.isfinite(results["disparate_impact_ratio"])
    assert isinstance(results["passed_quality_gates"], bool)


def test_seed_reproducibility():
    p1 = CreditRiskPipeline(random_seed=42)
    r1 = p1.run_pipeline(model_type="hist_gbdt", max_samples=1000)

    p2 = CreditRiskPipeline(random_seed=42)
    r2 = p2.run_pipeline(model_type="hist_gbdt", max_samples=1000)

    assert r1["roc_auc"] == r2["roc_auc"]
    assert r1["brier_score"] == r2["brier_score"]


def test_disparate_impact_ratio_computation():
    y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 0])
    # SEX: 2 is female (unprivileged), 1 is male (privileged)
    sex_attr = pd.Series([1, 1, 1, 1, 2, 2, 2, 2])
    dir_val = calculate_disparate_impact_ratio(y_pred, sex_attr)
    assert dir_val > 0.0


def test_disparate_impact_ratio_fails_when_group_is_missing():
    y_pred = np.array([0, 1, 0])
    sex_attr = pd.Series([1, 1, 1])
    with pytest.raises(ValueError, match="both comparison groups"):
        calculate_disparate_impact_ratio(y_pred, sex_attr)


def test_disparate_impact_ratio_fails_for_zero_reference_approval_rate():
    y_pred = np.array([1, 1, 0, 0])
    sex_attr = pd.Series([1, 1, 2, 2])
    with pytest.raises(ValueError, match="approval rate is zero"):
        calculate_disparate_impact_ratio(y_pred, sex_attr)


def test_airflow_dag_does_not_claim_execution_when_airflow_is_unavailable():
    if not airflow_dag.AIRFLOW_AVAILABLE:
        assert airflow_dag.dag is None
    else:
        assert airflow_dag.dag.dag_id == "credit_risk_candidate_evaluation"
