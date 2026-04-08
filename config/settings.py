"""
JeevanSync AI — Central Configuration
=====================================
All application settings managed via environment variables with sensible defaults.
Uses pydantic-settings for validation and .env file support.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


# ── Project root detection ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


class DataSettings(BaseSettings):
    """Paths to raw and processed data."""

    dataset_dir: Path = DATASET_DIR
    patients_csv: Path = DATASET_DIR / "patients.csv"
    vitals_csv: Path = DATASET_DIR / "vitals_timeseries.csv"
    labs_csv: Path = DATASET_DIR / "labs_timeseries.csv"
    hourly_panel_csv: Path = DATASET_DIR / "hospital_deterioration_hourly_panel.csv"
    ml_ready_csv: Path = DATASET_DIR / "hospital_deterioration_ml_ready.csv"
    mimic_dir: Path = DATASET_DIR / "mimic-iv-clinical-database-demo-2.2"

    # Output directories
    data_lake_dir: Path = PROJECT_ROOT / "data" / "datalake"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    features_dir: Path = PROJECT_ROOT / "data" / "features"

    model_config = ConfigDict(env_prefix="JEEVAN_DATA_")


class KafkaSettings(BaseSettings):
    """Redpanda / Kafka broker configuration."""

    bootstrap_servers: str = "localhost:9092"
    vitals_topic: str = "icu-vitals"
    labs_topic: str = "icu-labs"
    batch_topic: str = "icu-batch"
    consumer_group: str = "jeevansync-preprocessor"
    auto_offset_reset: str = "earliest"

    # Fallback to local file queue when broker unavailable
    use_local_fallback: bool = True
    local_queue_dir: Path = PROJECT_ROOT / "data" / "local_queue"

    model_config = ConfigDict(env_prefix="JEEVAN_KAFKA_")


class FeatureSettings(BaseSettings):
    """Feature engineering parameters."""

    # Sliding window
    window_sizes: list[int] = Field(default=[6, 12])
    default_window_size: int = 12
    prediction_horizon: int = 12  # hours ahead

    # Vital sign columns
    vital_columns: list[str] = Field(default=[
        "heart_rate", "respiratory_rate", "spo2_pct",
        "temperature_c", "systolic_bp", "diastolic_bp",
    ])

    # Lab columns
    lab_columns: list[str] = Field(default=[
        "wbc_count", "lactate", "creatinine",
        "crp_level", "hemoglobin",
    ])

    # All numeric columns for feature extraction
    numeric_columns: list[str] = Field(default=[
        "heart_rate", "respiratory_rate", "spo2_pct",
        "temperature_c", "systolic_bp", "diastolic_bp",
        "oxygen_flow", "mobility_score", "nurse_alert",
        "wbc_count", "lactate", "creatinine",
        "crp_level", "hemoglobin", "sepsis_risk_score",
    ])

    # Categorical columns
    categorical_columns: list[str] = Field(default=[
        "gender", "admission_type", "oxygen_device",
    ])

    # Static patient features
    static_columns: list[str] = Field(default=[
        "age", "gender", "comorbidity_index", "admission_type",
    ])

    # Target column
    target_column: str = "deterioration_next_12h"

    # Data split ratios (patient-level)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    model_config = ConfigDict(env_prefix="JEEVAN_FEATURE_")


class APISettings(BaseSettings):
    """FastAPI application settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    title: str = "JeevanSync AI — ICU Data Ingestion API"
    version: str = "1.0.0"
    cors_origins: list[str] = Field(default=["*"])

    model_config = ConfigDict(env_prefix="JEEVAN_API_")


class ModelSettings(BaseSettings):
    """Model training and serving settings."""

    artifacts_dir: Path = PROJECT_ROOT / "models" / "artifacts"
    mlflow_tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow_experiment_name: str = "jeevansync-deterioration"

    # XGBoost defaults
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05

    # LSTM / Transformer
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3

    # GPU (RTX 3050 4GB)
    use_gpu: bool = True
    gpu_device: str = "cuda:0"

    # Optuna
    optuna_n_trials: int = 100

    model_config = ConfigDict(env_prefix="JEEVAN_MODEL_")


class ServingSettings(BaseSettings):
    """Model serving configuration."""

    serving_host: str = "0.0.0.0"
    serving_port: int = 8001
    model_cache_ttl: int = 3600  # seconds

    model_config = ConfigDict(env_prefix="JEEVAN_SERVING_")


# ── Singleton instances ──────────────────────────────────────────────────
data_settings = DataSettings()
kafka_settings = KafkaSettings()
feature_settings = FeatureSettings()
api_settings = APISettings()
model_settings = ModelSettings()
serving_settings = ServingSettings()
