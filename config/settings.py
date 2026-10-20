"""
JeevanSync AI — Central Configuration
=====================================
All application settings managed via environment variables with sensible defaults.
Uses pydantic-settings for validation and .env file support.
"""

from pathlib import Path

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

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


class EMRSettings(BaseSettings):
    """EMR/EHR integration settings (HL7v2 + FHIR R4)."""

    # Master toggle — off by default
    emr_enabled: bool = False

    # FHIR R4 Configuration
    fhir_base_url: str = "https://ehr.hospital.org/fhir/R4"
    fhir_client_id: str = ""
    fhir_client_secret: str = ""
    fhir_auth_token_url: str = ""
    fhir_timeout_s: int = 30
    fhir_polling_interval_s: int = 60
    fhir_subscription_enabled: bool = True

    # HL7v2 MLLP Configuration
    hl7_mllp_host: str = "0.0.0.0"
    hl7_mllp_port: int = 2575
    hl7_enabled: bool = False

    # Audit trail
    audit_log_dir: Path = PROJECT_ROOT / "data" / "audit"

    # Data source tagging (for provenance tracking)
    emr_source_tag: str = "emr-fhir"

    model_config = ConfigDict(env_prefix="JEEVAN_EMR_")


class FederatedSettings(BaseSettings):
    """Federated learning configuration (Phase 6).

    Controls multi-site federated training with differential privacy.
    When fl_role == "standalone" (default), the entire FL subsystem is
    inert and the existing training pipeline runs exactly as before.
    """

    # Master toggle — off by default
    fl_enabled: bool = False
    fl_role: str = "standalone"  # "coordinator", "participant", "standalone"

    # Coordinator settings
    coordinator_host: str = "0.0.0.0"
    coordinator_port: int = 50051
    coordinator_url: str = "localhost:50051"

    # Site identification
    site_id: str = "site-001"
    site_name: str = "Default Hospital"

    # Training rounds
    fl_rounds: int = 50
    fl_min_participants: int = 2
    fl_epochs_per_round: int = 3
    fl_learning_rate: float = 1e-3

    # Aggregation strategy
    aggregation_strategy: str = "fedavg"  # "fedavg", "fedprox", "fedbn"
    fedprox_mu: float = 0.01  # Proximal term weight for FedProx

    # Differential privacy
    dp_enabled: bool = True
    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.1

    # Secure aggregation
    secure_aggregation: bool = True

    # Model versioning
    fl_artifacts_dir: Path = PROJECT_ROOT / "models" / "artifacts" / "federated"

    # gRPC settings
    grpc_max_message_size: int = 100 * 1024 * 1024  # 100 MB for model weights

    model_config = ConfigDict(env_prefix="JEEVAN_FL_")


class ContinuousSettings(BaseSettings):
    """Continuous learning configuration (Phase 7).

    Controls drift-triggered retraining, champion/challenger model
    promotion, and model versioning. When continuous_enabled is False
    (default), the entire continuous learning subsystem is inert.

    IMPORTANT: auto_promote defaults to False. In a clinical system,
    model promotion should require human approval by default. Set to
    True only after FDA clearance for adaptive AI (Phase 9).
    """

    # Master toggle — off by default
    continuous_enabled: bool = False

    # Drift check schedule
    drift_check_interval_s: int = 3600  # Check drift hourly

    # Retraining buffer
    retraining_buffer_dir: Path = PROJECT_ROOT / "data" / "retraining_buffer"
    min_retrain_samples: int = 5000
    max_buffer_size_mb: int = 500

    # Champion/Challenger evaluation
    shadow_evaluation_hours: int = 24
    champion_auroc_threshold: float = 0.005  # 0.5% AUROC improvement required
    champion_auprc_threshold: float = 0.0  # AUPRC must be >= champion
    min_evaluation_samples: int = 1000

    # Model versioning
    model_versions_dir: Path = PROJECT_ROOT / "models" / "versions"
    max_model_versions: int = 10

    # Promotion policy
    auto_promote: bool = False  # Require human approval by default

    # Fine-tuning for online updates
    finetune_lr_multiplier: float = 0.1  # 10% of base LR for fine-tuning
    finetune_weight_decay: float = 1e-4  # Prevent catastrophic forgetting

    model_config = ConfigDict(env_prefix="JEEVAN_CL_")


class MobileSettings(BaseSettings):
    """Mobile companion app configuration (Phase 8).

    Controls JWT authentication, Firebase Cloud Messaging push
    notifications, and mobile-optimized API behavior. Both auth
    and FCM are feature-flagged OFF by default so the existing
    unauthenticated serving API remains unchanged.
    """

    # ── JWT Authentication ───────────────────────────────────────
    auth_enabled: bool = False  # Feature flag — off by default
    jwt_secret: str = "jeevansync-mobile-jwt-secret-key-32b!"  # Overridable via JEEVAN_MOBILE_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    jwt_refresh_expiry_hours: int = 168  # 7 days

    # ── Firebase Cloud Messaging ─────────────────────────────────
    fcm_enabled: bool = False  # Feature flag — off by default
    fcm_server_key: str = ""  # Set via JEEVAN_MOBILE_FCM_SERVER_KEY
    fcm_project_id: str = ""

    # ── Push notification thresholds ─────────────────────────────
    push_critical_threshold: float = 0.75  # Push for CRITICAL risk
    push_high_threshold: float = 0.50  # Push for HIGH risk

    # ── Device registry persistence ──────────────────────────────
    device_registry_path: Path = PROJECT_ROOT / "data" / "device_registry.json"

    # ── Demo physician credentials (dev only) ────────────────────
    demo_username: str = "dr.sharma"
    demo_password_hash: str = ""  # bcrypt hash; empty = accept any password in dev

    model_config = ConfigDict(env_prefix="JEEVAN_MOBILE_")


class MultiOutcomeSettings(BaseSettings):
    """Multi-outcome prediction configuration (Phase 10).

    Extends the binary deterioration model to simultaneously predict
    sepsis, cardiac arrest, and respiratory failure. When
    multi_outcome_enabled is False (default), the entire multi-outcome
    subsystem is inert and the existing binary prediction pipeline
    runs exactly as before.
    """

    # Master toggle — off by default
    multi_outcome_enabled: bool = False

    # Target outcome definitions
    outcome_targets: list[str] = Field(default=[
        "deterioration_next_12h",
        "sepsis_next_6h",
        "cardiac_arrest_next_12h",
        "respiratory_failure_next_12h",
    ])

    # Per-outcome task weights for multi-task loss (higher = rarer outcome)
    task_weights: list[float] = Field(default=[1.0, 1.5, 2.0, 1.5])

    # Multi-head model architecture
    shared_d_model: int = 64
    shared_nhead: int = 4
    shared_num_layers: int = 2
    head_hidden_dim: int = 64
    head_dropout: float = 0.3

    # Training
    multi_epochs: int = 60
    multi_learning_rate: float = 5e-4
    multi_batch_size: int = 32

    # Model artifacts
    multi_artifacts_dir: Path = PROJECT_ROOT / "models" / "artifacts" / "multi_outcome"

    model_config = ConfigDict(env_prefix="JEEVAN_MULTI_")


# ── Singleton instances ──────────────────────────────────────────────────
data_settings = DataSettings()
kafka_settings = KafkaSettings()
feature_settings = FeatureSettings()
api_settings = APISettings()
model_settings = ModelSettings()
serving_settings = ServingSettings()
emr_settings = EMRSettings()
federated_settings = FederatedSettings()
continuous_settings = ContinuousSettings()
mobile_settings = MobileSettings()
multi_outcome_settings = MultiOutcomeSettings()
