# 🧠 Universal AI/ML Project — Agent Instructions Template

> **Drop this file into any AI/ML project root as `AGENTS.md`.**
> Agents will auto-discover it and follow these conventions.
> Override any section with project-specific rules as needed.

---

## Table of Contents

- [Project Identity](#-project-identity)
- [Repository Layout](#-repository-layout)
- [Python & Code Quality](#-python--code-quality)
- [Data Management](#-data-management)
- [Feature Engineering](#-feature-engineering)
- [Model Development](#-model-development)
- [Experiment Tracking](#-experiment-tracking)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Model Serving & APIs](#-model-serving--apis)
- [Monitoring & Observability](#-monitoring--observability)
- [Testing](#-testing)
- [Security & Compliance](#-security--compliance)
- [Infrastructure & Deployment](#-infrastructure--deployment)
- [Commit & PR Conventions](#-commit--pr-conventions)
- [Common Pitfalls](#-common-pitfalls)
- [Quick Reference Template](#-quick-reference-template)

---

## 🪪 Project Identity

> **Fill this section in for your specific project.**

```
Project Name:       <your-project-name>
Domain:             <e.g., healthcare, finance, NLP, CV, recommendation>
Task Type:          <classification | regression | generation | ranking | anomaly-detection>
Primary Framework:  <PyTorch | TensorFlow | JAX | scikit-learn | XGBoost>
Python Version:     >=3.10
License:            <MIT | Apache-2.0 | Proprietary>
```

---

## 🏗️ Repository Layout

Use this canonical structure. Adapt as needed but **never flatten everything into root**.

```
project-root/
│
├── config/                  # All configuration (settings, logging, constants)
│   ├── settings.py          # Centralized config (pydantic-settings or dataclasses)
│   ├── logging_config.py    # Structured logging setup
│   └── constants.py         # Project-wide constants (column names, paths, enums)
│
├── src/                     # All source code lives here
│   ├── data/                # Data loading, validation, versioning
│   │   ├── loaders.py       # Dataset readers (CSV, Parquet, DB, API)
│   │   ├── schemas.py       # Pydantic/dataclass validation schemas
│   │   └── splitters.py     # Train/val/test splitting logic
│   │
│   ├── preprocessing/       # Feature engineering & transformation
│   │   ├── pipeline.py      # Orchestrates all preprocessing steps
│   │   ├── transforms.py    # Individual transform functions
│   │   ├── imputation.py    # Missing value strategies
│   │   └── encoding.py      # Categorical encoding, normalization
│   │
│   ├── modeling/            # Model definitions, training, evaluation
│   │   ├── base.py          # Abstract base class for all models
│   │   ├── architectures/   # Model definitions (nets, configs)
│   │   ├── trainers/        # Training loops (one per model type)
│   │   ├── losses.py        # Custom loss functions
│   │   ├── ensemble.py      # Model combination strategies
│   │   ├── calibration.py   # Probability calibration
│   │   └── hpo.py           # Hyperparameter optimization (Optuna, Ray Tune)
│   │
│   ├── evaluation/          # Metrics, analysis, reporting
│   │   ├── metrics.py       # Custom metric implementations
│   │   ├── analysis.py      # Error analysis, slice-based evaluation
│   │   └── reports.py       # Automated evaluation report generation
│   │
│   ├── explainability/      # Model interpretability
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   └── attention_viz.py
│   │
│   ├── serving/             # Model serving & inference
│   │   ├── api.py           # FastAPI/Flask endpoints
│   │   ├── inference.py     # Inference logic (batched, streaming)
│   │   └── middleware.py    # Auth, rate limiting, request validation
│   │
│   ├── monitoring/          # Production monitoring
│   │   ├── drift.py         # Data/concept drift detection
│   │   ├── metrics.py       # Prometheus/custom metrics
│   │   └── alerts.py        # Alerting logic
│   │
│   └── utils/               # Shared utilities
│       ├── io.py            # File I/O helpers
│       ├── timing.py        # Profiling decorators
│       └── reproducibility.py  # Seed setting, determinism
│
├── notebooks/               # Jupyter notebooks (EDA, prototyping ONLY)
├── tests/                   # Mirror src/ structure
├── scripts/                 # One-off scripts (data download, migration)
├── docker/                  # Dockerfile + docker-compose
├── data/                    # Processed/intermediate data (gitignored)
├── models/                  # Trained artifacts (gitignored)
├── logs/                    # Log files (gitignored)
├── pyproject.toml           # Project metadata + tool configs
├── requirements.txt         # Pinned dependencies
├── .env.example             # Template for env vars (never .env itself)
├── .gitignore
├── AGENTS.md                # ← This file
└── README.md
```

### Rules
- **Every directory** with Python code must have an `__init__.py` (can be minimal — a module docstring is fine).
- **Notebooks are for exploration only** — never import from notebooks in production code. Extract any reusable logic into `src/`.
- **Scripts** are one-off entry points (e.g., `scripts/download_data.py`). They import from `src/`, never the reverse.
- Keep the `data/`, `models/`, and `logs/` directories **gitignored**. Use DVC, MLflow, or cloud storage for versioning.

---

## 🐍 Python & Code Quality

### Language Standards
- **Python ≥ 3.10** — Use modern syntax:
  - `X | None` instead of `Optional[X]`
  - `list[str]` instead of `List[str]`
  - `match/case` for complex branching
  - f-strings everywhere (never `%` formatting or `.format()`)
- All files start with `from __future__ import annotations` for forward references.
- **Type hints are mandatory** on all function signatures, return types, and class attributes.

### Type Hinting Cheat Sheet
```python
from __future__ import annotations

# Function signatures — always typed
def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, float]:
    ...

# Class attributes — always typed
class ModelConfig:
    hidden_size: int = 128
    dropout: float = 0.1
    num_layers: int = 2

# Complex types
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

Batch: TypeAlias = dict[str, torch.Tensor]
MetricFn: TypeAlias = Callable[[np.ndarray, np.ndarray], float]
```

### Linting & Formatting
- **Ruff** is the sole linter/formatter. Configure in `pyproject.toml`:
  ```toml
  [tool.ruff]
  line-length = 100
  target-version = "py310"

  [tool.ruff.lint]
  select = ["E", "F", "I", "W", "UP", "B", "SIM", "RUF"]
  ignore = ["E501"]  # line length handled by formatter
  ```
- Do **NOT** introduce `black`, `flake8`, `isort`, or `pylint` — Ruff replaces all of them.
- Run: `ruff check src/ config/ tests/` and `ruff format src/ config/ tests/`

### Docstrings
- Every public module, class, and function gets a docstring.
- Use **Google style** docstrings:
  ```python
  def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
      """Compute Area Under the ROC Curve.

      Args:
          y_true: Ground truth binary labels (0 or 1).
          y_prob: Predicted probabilities in [0, 1].

      Returns:
          AUROC score as a float in [0, 1].

      Raises:
          ValueError: If arrays have different lengths.
      """
  ```
- Module-level banner docstrings for visual structure:
  ```python
  """
  ProjectName — Module Name
  ==============================
  Brief description of what this module does.
  """
  ```

### Import Ordering
Enforce via Ruff `I` rules. Order:
1. `__future__`
2. Standard library (`os`, `sys`, `pathlib`, `json`)
3. Third-party packages (`numpy`, `torch`, `fastapi`)
4. Local imports (`config.settings`, `src.modeling.base`)

### Naming Conventions

| Element          | Convention        | Example                           |
| ---------------- | ----------------- | --------------------------------- |
| Classes          | `PascalCase`      | `TransformerTrainer`, `DataConfig` |
| Functions        | `snake_case`      | `compute_shap_values()`           |
| Constants        | `UPPER_SNAKE`     | `PROJECT_ROOT`, `MAX_SEQ_LEN`     |
| Private methods  | `_single_prefix`  | `_validate_input()`               |
| File names       | `snake_case.py`   | `feature_extractor.py`            |
| Test files       | `test_*.py`       | `test_preprocessing.py`           |
| Config env vars  | `PREFIX_GROUP_KEY` | `MYPROJECT_MODEL_LR`             |

---

## 📦 Data Management

### Principles
1. **Raw data is immutable.** Never modify source data files. Transform into new files.
2. **All data paths in config**, never hardcoded in code.
3. **Validate early.** Use Pydantic schemas or pandera to validate data at ingestion boundaries.
4. **Version your data.** Use DVC, Delta Lake, or at minimum timestamp-named output directories.
5. **Document your schema.** Every dataset should have column definitions, types, and valid ranges.

### Data Validation Pattern
```python
from pydantic import BaseModel, Field, field_validator

class SampleRecord(BaseModel):
    """Validated record for a single observation."""

    entity_id: int = Field(..., ge=1)
    feature_a: float = Field(..., ge=0, le=1000, description="Feature A value")
    feature_b: float = Field(..., ge=-100, le=100, description="Feature B value")
    timestamp: datetime

    @field_validator("feature_a")
    @classmethod
    def validate_feature_a(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Feature A cannot be negative: {v}")
        return v
```

### Data Splitting Rules
- Always split **at the entity level** (patient, user, session) — not at the row level.
- Prevent **data leakage**: temporal features must not use future information.
- Standard splits: **70/15/15** (train/val/test) unless domain-specific requirements dictate otherwise.
- Use a **fixed random seed** for reproducibility (default: `42`).

### Handling Class Imbalance
- Document the class distribution in your config or README.
- Strategies (choose based on context):
  - **Weighted loss functions** (preferred for deep learning)
  - **SMOTE / ADASYN** (tabular only, apply to train set only)
  - **Stratified sampling** in data splits
  - **Focal loss** for severe imbalance
- **Never** apply oversampling to validation or test sets.

---

## ⚙️ Feature Engineering

### Principles
1. **Pipeline, not scripts.** Feature engineering must be a reproducible pipeline (`sklearn.Pipeline`, custom pipeline class, or orchestrator script).
2. **Fit on train only.** All fitted transforms (scalers, encoders, imputers) are fitted on the training set and applied to val/test.
3. **Feature names travel with data.** Always maintain column names through the pipeline. Never lose track of what each feature represents.
4. **Document derived features.** If you create `heart_rate_rolling_mean_6h`, document the formula and rationale.

### Feature Engineering Checklist
- [ ] Missing value imputation strategy defined and documented
- [ ] Numerical features: scaling/normalization approach chosen
- [ ] Categorical features: encoding strategy (one-hot, target, ordinal)
- [ ] Temporal features: lag features, rolling windows, time-since events
- [ ] Interaction features: if domain-relevant
- [ ] Feature selection: importance-based filtering or recursive elimination
- [ ] Feature names and descriptions in a registry/config

### Reproducibility
- Save fitted preprocessors alongside model artifacts (e.g., `scaler.joblib`, `encoder.joblib`).
- The serving pipeline must use the **exact same transforms** as training. Never re-fit in production.

---

## 🧠 Model Development

### Base Class Pattern
Always define an abstract base class for model trainers:

```python
import abc
from typing import Any

class BaseTrainer(abc.ABC):
    """Abstract interface for all model trainers."""

    @abc.abstractmethod
    def train(self, X_train, y_train, X_val, y_val) -> dict[str, float]:
        """Train the model. Return validation metrics."""

    @abc.abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Return predicted probabilities."""

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Save model artifacts to disk."""

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Load model artifacts from disk."""

    def evaluate(self, X_test, y_test) -> dict[str, float]:
        """Evaluate on test set. Override for custom metrics."""
        y_prob = self.predict(X_test)
        return compute_standard_metrics(y_test, y_prob)
```

### Training Rules
1. **Always log experiments** — MLflow, Weights & Biases, or TensorBoard. No untracked experiments.
2. **Always set seeds** for reproducibility:
   ```python
   import random, numpy as np, torch
   def set_seed(seed: int = 42) -> None:
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
   ```
3. **Early stopping** on the validation metric, not training loss.
4. **Checkpoint the best model**, not just the last epoch.
5. **Log everything**: hyperparameters, metrics per epoch, dataset versions, git commit hash.

### GPU Management
- Always check GPU availability:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- Be mindful of VRAM. Document the expected memory footprint.
- Use **mixed precision** (`torch.amp` / `tf.keras.mixed_precision`) for large models.
- Use **gradient accumulation** when batch sizes don't fit in VRAM.
- For multi-GPU: prefer `DistributedDataParallel` over `DataParallel`.

### Deep Learning Specifics (PyTorch)
- Models inherit from `nn.Module`. Keep `forward()` clean.
- Use `torch.no_grad()` for inference.
- Move tensors to device **once** at batch creation, not inside the model.
- Prefer `torch.utils.data.DataLoader` with `num_workers > 0` and `pin_memory=True` for GPU.

### Ensemble Strategies
When combining multiple models:
- **Learned weights** > simple averaging (use validation set to learn weights).
- Ensemble members should be **diverse** (different architectures or feature subsets).
- Always compare ensemble vs. best single model — ensemble must earn its complexity.

### Calibration
- **Raw model outputs are rarely well-calibrated.** Always check calibration.
- Use **isotonic regression** or **Platt scaling** as a post-hoc calibration step.
- Report **Expected Calibration Error (ECE)** alongside discrimination metrics.
- Calibrate on the validation set, evaluate on the test set.
- For clinical/financial applications, calibration is **non-negotiable**.

---

## 📊 Experiment Tracking

### Mandatory Logging
Every training run must log:

| What to log          | Example                                          |
| -------------------- | ------------------------------------------------ |
| Hyperparameters      | `lr=0.001, batch_size=64, epochs=50`             |
| Dataset version      | `dataset_v2.3, split_seed=42, n_train=10000`    |
| Metrics per epoch    | `train_loss, val_loss, val_auroc, val_auprc`     |
| Final test metrics   | `test_auroc=0.95, test_auprc=0.72`              |
| Model artifacts      | `model.pt, scaler.joblib, config.json`           |
| Environment          | `python=3.11, torch=2.3.0, cuda=12.4`           |
| Git commit           | `abc1234`                                        |
| Training time        | `duration_seconds=3600`                          |

### MLflow Pattern
```python
import mlflow

with mlflow.start_run(run_name="xgboost-v3"):
    mlflow.log_params({"lr": 0.01, "max_depth": 8})
    mlflow.log_metrics({"auroc": 0.95, "auprc": 0.72})
    mlflow.log_artifact("models/xgboost_model.joblib")
    mlflow.set_tag("model_type", "xgboost")
```

---

## 📈 Evaluation & Metrics

### Metric Selection by Task Type

| Task                | Primary Metrics                      | Avoid              |
| ------------------- | ------------------------------------ | ------------------- |
| Binary classification | **AUROC, AUPRC**, F1, Recall@K     | Accuracy alone      |
| Multi-class         | Macro F1, Weighted F1, Confusion Matrix | Accuracy alone   |
| Regression          | **RMSE, MAE**, R², MAPE             | R² alone            |
| Ranking             | **NDCG, MRR**, MAP@K                | —                   |
| Anomaly detection   | **AUPRC**, Recall@FPR, F1           | Accuracy            |
| Generation (NLP)    | **BLEU, ROUGE, BERTScore**          | Perplexity alone    |
| Computer Vision     | **mAP, IoU**, Precision@K           | Accuracy alone      |
| Time Series         | **RMSE, MAE, MASE**, SMAPE          | R² alone            |

### Imbalanced Data Warning
- **Never report accuracy as the primary metric** when classes are imbalanced.
- Always report **AUPRC** (area under precision-recall curve) — it is more informative than AUROC for rare events.
- Report metrics at **multiple thresholds** (e.g., sensitivity at 95% specificity).

### Evaluation Checklist
- [ ] Metrics computed on held-out test set (never on train/val)
- [ ] Confidence intervals or bootstrap estimates reported
- [ ] Per-class/subgroup breakdown (fairness slicing)
- [ ] Comparison against a meaningful baseline (not just random)
- [ ] Calibration plot + ECE if outputting probabilities
- [ ] Error analysis: what does the model get wrong and why?

---

## 🚀 Model Serving & APIs

### API Framework
- Use **FastAPI** for Python ML APIs. It provides:
  - Automatic OpenAPI docs
  - Pydantic request/response validation
  - Native async support
  - Dependency injection

### API Design Pattern
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="ML Prediction API", version="1.0.0")

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    model_type: str = "ensemble"

class PredictResponse(BaseModel):
    prediction: float = Field(..., ge=0, le=1)
    model_used: str
    latency_ms: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    ...

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": [...], "gpu_available": bool}
```

### API Rules
1. **All request/response models use Pydantic `BaseModel`** with `Field()` validators.
2. **Health check endpoint is mandatory**: `GET /health`.
3. **Validate input ranges** — never pass raw user input to the model.
4. **Return prediction latency** in the response for monitoring.
5. **Use `async def`** for endpoints — never block the event loop.
6. **Return meaningful HTTP errors**: 400 (bad input), 422 (validation), 500 (model error).
7. **Never expose stack traces** in production error responses.
8. **Batch endpoint** for high-throughput scenarios: `POST /predict/batch`.

### Model Loading
- Load models **once at startup** (via `lifespan` or `@app.on_event("startup")`).
- Cache loaded models in memory. Do not reload per-request.
- Log model metadata at startup (version, training date, metrics).

---

## 📡 Monitoring & Observability

### What to Monitor in Production

| Signal              | Tool/Approach                        | Alert Threshold            |
| ------------------- | ------------------------------------ | -------------------------- |
| Prediction latency  | Prometheus histogram                 | p99 > 500ms               |
| Prediction volume   | Prometheus counter                   | < 50% of baseline         |
| Input distribution  | KS-test, PSI                         | PSI > 0.2                 |
| Output distribution | Mean prediction drift                | > 2σ from baseline        |
| Error rate          | HTTP 5xx counter                     | > 1%                      |
| Model staleness     | Age of loaded model                  | > 30 days                 |
| GPU utilization     | `nvidia-smi` / DCGM                  | Sustained > 95%           |
| Memory usage        | Process RSS                          | > 80% of available        |

### Data & Concept Drift
- Implement **input drift detection** using Population Stability Index (PSI) or Kolmogorov-Smirnov tests.
- Compare incoming feature distributions against a **reference window** (training data statistics).
- Trigger **retraining alerts** (not automatic retraining) when drift exceeds thresholds.
- Log all drift metrics for post-hoc analysis.

---

## 🧪 Testing

### Framework & Setup
- **pytest** as the test runner. Configure in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  python_functions = ["test_*"]
  addopts = "-v --tb=short"
  asyncio_mode = "auto"
  ```
- Use `pytest-asyncio` for async tests, `pytest-cov` for coverage.

### Test Structure
Mirror `src/` in `tests/`:
```
tests/
├── test_data/             # Tests for src/data/
├── test_preprocessing/    # Tests for src/preprocessing/
├── test_modeling/         # Tests for src/modeling/
├── test_serving/          # Tests for src/serving/
├── test_monitoring/       # Tests for src/monitoring/
├── conftest.py            # Shared fixtures
└── fixtures/              # Test data files
```

### What to Test

| Layer              | What to Test                                      | How                        |
| ------------------ | ------------------------------------------------- | -------------------------- |
| Data validation    | Schema rejects invalid inputs                     | Pydantic validation tests  |
| Preprocessing      | Transforms produce expected shapes/types          | Unit tests with fixtures   |
| Model              | Forward pass runs, output shape correct            | Smoke tests, no training   |
| Training           | 1-epoch overfit on tiny data                       | Integration test           |
| API endpoints      | Status codes, response schemas                     | `TestClient` from FastAPI  |
| Metrics            | Custom metrics match expected values               | Unit tests with known data |

### What NOT to Test
- ❌ Third-party library internals (PyTorch, scikit-learn)
- ❌ Tests that require GPU or large datasets
- ❌ Tests that hit external services without mocking
- ❌ Model accuracy (this belongs in evaluation, not unit tests)

### Test Pattern
```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from src.serving.api import app
    return TestClient(app)

class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def _make_request(self) -> dict:
        return {"features": [1.0, 2.0, 3.0], "model_type": "ensemble"}

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=self._make_request())
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        response = client.post("/predict", json=self._make_request())
        data = response.json()
        assert 0 <= data["prediction"] <= 1
        assert "latency_ms" in data

    def test_predict_rejects_empty_features(self, client):
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 422
```

---

## 🔒 Security & Compliance

### Data Security
- **Never log PII/PHI** (names, emails, SSNs, medical records, financial data).
- Entity IDs in code must be **synthetic or anonymized**.
- Datasets must be **de-identified** before committing or sharing.
- Comply with domain regulations: HIPAA (healthcare), GDPR (EU data), SOC2 (enterprise).

### Secrets Management
- `.env` files are **always gitignored**. Provide `.env.example` as a template.
- All secrets flow through **environment variables → config system** (pydantic-settings, python-dotenv).
- Never commit: API keys, database credentials, cloud tokens, model registry passwords.

### Model Artifacts
- Trained models (`*.pt`, `*.joblib`, `*.h5`, `*.onnx`) are **gitignored**.
- Store in: MLflow Model Registry, cloud storage (S3/GCS), or DVC.
- Never commit model weights to version control (files are too large, leak training data patterns).

### Dependency Security
- Pin **minimum versions** in `requirements.txt` (e.g., `torch>=2.3`).
- Run `pip audit` or Dependabot for vulnerability scanning.
- Review and vet any new dependency before adding.

---

## 🐳 Infrastructure & Deployment

### Docker
- Use **multi-stage builds** to keep images small:
  ```dockerfile
  # Stage 1: Build
  FROM python:3.11-slim AS builder
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  # Stage 2: Runtime
  FROM python:3.11-slim
  COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
  COPY src/ /app/src/
  COPY config/ /app/config/
  CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8001"]
  ```
- Health checks in `docker-compose.yml` for every service.
- Mount model artifacts as **read-only volumes** in serving containers.

### GPU in Docker
- Use `nvidia/cuda` base images or `deploy.resources.reservations.devices` in compose.
- Always set `NVIDIA_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` explicitly.

### Configuration
- All settings configurable via **environment variables**.
- Use `pydantic-settings` or `python-dotenv` for config management.
- Never hardcode: hostnames, ports, paths, credentials, hyperparameters.
- Use a `ConfigDict(env_prefix="MYPROJECT_")` pattern to namespace env vars.

---

## 📝 Commit & PR Conventions

### Commit Messages
Use **conventional commits**:
```
feat(modeling):      add new attention-based model architecture
fix(serving):        handle NaN inputs in prediction endpoint
refactor(pipeline):  extract feature transforms into separate module
perf(training):      enable mixed precision for 2x speedup
docs(readme):        update model performance benchmarks
test(preprocessing): add edge case tests for missing value imputation
chore(deps):         upgrade torch to 2.4.0
data(features):      add rolling window features for vitals
```

### ML-Specific Scope Tags
- `data` — data loading, schemas, validation
- `features` — feature engineering changes
- `modeling` — model architecture, training logic
- `eval` — evaluation metrics, analysis
- `serving` — API endpoints, inference
- `monitoring` — drift detection, alerting
- `pipeline` — end-to-end pipeline orchestration
- `hpo` — hyperparameter optimization

### PR Rules
- One logical change per PR.
- Include **test coverage** for new features.
- Include **evaluation results** for model changes (metrics table in PR description).
- Update docstrings if public interfaces change.
- Breaking changes to API schemas require versioning discussion.
- Model architecture changes require ensemble integration review.

---

## 📊 Logging

### Rules
- Use a **structured logging library** (loguru, structlog, or stdlib logging with JSON formatter).
- **Never use `print()` in production code.** Always use the logger.
- Log levels:
  - `DEBUG` — Detailed diagnostic info (batch sizes, tensor shapes, per-sample processing)
  - `INFO` — Normal operation events (training started, epoch completed, model loaded)
  - `WARNING` — Potential issues (missing optional config, fallback used, degraded performance)
  - `ERROR` — Failures that prevent normal operation (model load failed, API error)
  - `CRITICAL` — System-level failures (out of memory, GPU unavailable when required)
- Rotating file logs with **daily rotation** and **30-day retention**.
- JSON-structured logs in production for machine parsing.

### Loguru Pattern (Recommended)
```python
from loguru import logger

logger.info("Training started | model={} epochs={} lr={}", model_name, epochs, lr)
logger.warning("GPU not available, falling back to CPU")
logger.error("Model file not found | path={}", model_path)
```

---

## ⚠️ Common Pitfalls

### Data Pitfalls
1. **Data leakage** — Using future information in features. Always respect temporal ordering.
2. **Train/test contamination** — Fitting transformers on test data. Fit on train only.
3. **Ignoring class imbalance** — Reporting accuracy on 95/5 splits. Use AUPRC.
4. **Hardcoded column names** — Put all column names in config, not scattered in code.

### Model Pitfalls
5. **No calibration** — Raw probabilities are rarely calibrated. Always calibrate.
6. **No baseline comparison** — Always compare against a simple baseline (logistic regression, majority class, last value).
7. **Overfitting to validation set** — Limit hyperparameter tuning rounds. Use separate test set.
8. **Silent NaN propagation** — Add NaN checks at model input and output boundaries.

### Engineering Pitfalls
9. **Training/serving skew** — Different preprocessing in training vs. serving. Use the same pipeline object.
10. **Untracked experiments** — Every training run must be logged. No exceptions.
11. **Blocking API calls** — Use `async def` in FastAPI. Never run model inference synchronously on the main thread.
12. **Missing health checks** — Every service needs a `/health` endpoint and Docker healthcheck.
13. **No reproducibility** — Always set seeds, log library versions, pin dependencies.
14. **Giant monolithic files** — If a file exceeds 500 lines, split it into focused modules.

---

## 🗺️ Quick Reference Template

> **Copy this table and fill it in for your project.**

| What you need                  | Where to find it                     |
| ------------------------------ | ------------------------------------ |
| All configuration              | `config/settings.py`                 |
| Logging setup                  | `config/logging_config.py`           |
| Data validation schemas        | `src/data/schemas.py`                |
| Data loaders                   | `src/data/loaders.py`                |
| Feature pipeline               | `src/preprocessing/pipeline.py`      |
| Base model class               | `src/modeling/base.py`               |
| Model architectures            | `src/modeling/architectures/`        |
| Training loops                 | `src/modeling/trainers/`             |
| Loss functions                 | `src/modeling/losses.py`             |
| Ensemble logic                 | `src/modeling/ensemble.py`           |
| Calibration                    | `src/modeling/calibration.py`        |
| Hyperparameter optimization    | `src/modeling/hpo.py`                |
| Evaluation metrics             | `src/evaluation/metrics.py`          |
| Prediction API                 | `src/serving/api.py`                 |
| Explainability                 | `src/explainability/`                |
| Drift detection                | `src/monitoring/drift.py`            |
| Docker deployment              | `docker/docker-compose.yml`          |
| CI pipeline                    | `.github/workflows/ci.yml`           |

---

## 🔄 Customization Guide

This template is meant to be **forked and modified**. Here's how to adapt it:

1. **Fill in Project Identity** — Name, domain, task type, framework.
2. **Adjust the directory layout** — Remove modules you don't need, add domain-specific ones.
3. **Set your metrics** — Pick primary/secondary metrics for your task type.
4. **Add domain-specific rules** — Clinical validation ranges, financial compliance requirements, NLP tokenization standards, etc.
5. **Update the Quick Reference table** — Map your actual file paths.
6. **Add common pitfalls** — Document project-specific gotchas that waste time.

> **The best AGENTS.md is one that prevents an agent from making the same mistake twice.**
