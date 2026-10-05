"""Typed YAML configuration with a small, explicit environment override set."""

from pathlib import Path
from typing import Any, Optional
import os

import yaml
from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    name: str = "credit-risk-model"
    alias: str = "champion"
    min_roc_auc_gate: float = 0.7500
    target_roc_auc: float = 0.7700
    review_threshold: float = 0.30
    decline_threshold: float = 0.60
    algorithm: str = "HistGradientBoostingClassifier"
    random_state: int = 42


class DataSettings(BaseModel):
    train_split: float = 0.80
    sliding_window_size: int = 20000
    random_seed: int = 42
    null_tolerance: float = 0.00
    train_data_path: Optional[Path] = None


class FairnessSettings(BaseModel):
    protected_attribute: str = "SEX"
    dir_min_threshold: float = 0.80
    dir_max_threshold: float = 1.25


class StorageSettings(BaseModel):
    mlflow_tracking_uri: str = "http://localhost:15040"
    model_dir: Path = Path("models")
    artifact_bucket: Optional[str] = None


class PipelineConfig(BaseModel):
    environment: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    model: ModelSettings = Field(default_factory=ModelSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    fairness: FairnessSettings = Field(default_factory=FairnessSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)

    @classmethod
    def load(cls, yaml_path: Optional[Path] = None) -> "PipelineConfig":
        if yaml_path is None:
            yaml_path = Path(__file__).parent / "config.yaml"

        data: dict[str, Any] = {}
        if yaml_path.exists():
            with yaml_path.open("r", encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Configuration in {yaml_path} must be a YAML mapping.")
            data = loaded

        if "APP_ENV" in os.environ:
            data["environment"] = os.environ["APP_ENV"]
        if "DATA_PATH" in os.environ:
            data.setdefault("data", {})["train_data_path"] = os.environ["DATA_PATH"]
        if "MLFLOW_TRACKING_URI" in os.environ:
            data.setdefault("storage", {})["mlflow_tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

        return cls.model_validate(data)


settings = PipelineConfig.load()
