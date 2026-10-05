# G_One_Sync AI — Agent Instructions

> **AI-Powered Early Clinical Deterioration Prediction System for ICU Patients**
> Built with Python 3.10+, PyTorch, XGBoost, FastAPI, and MLflow.

---

## 🏗️ Architecture Overview

This is a **production-grade clinical ML system** with a modular pipeline architecture.
Every module lives under `src/` and follows a strict separation of concerns:

```
src/
├── ingestion/       # FastAPI ingest API, Kafka producer/consumer, schemas
├── preprocessing/   # Feature extraction, imputation, sliding windows, validation
├── modeling/        # XGBoost, BiLSTM, Transformer trainers + ensemble + calibration
├── explainability/  # SHAP waterfall explanations per patient
├── serving/         # FastAPI prediction server (single, batch, explain endpoints)
├── monitoring/      # Prometheus metrics, drift detection, alerting
└── dashboard/       # Real-time clinical dashboard (Plotly + WebSocket)
```

**Supporting directories:**

| Directory    | Purpose                                              |
| ------------ | ---------------------------------------------------- |
| `config/`    | Pydantic-settings based configuration (`settings.py`, `logging_config.py`) |
| `tests/`     | Pytest test suite (mirrors `src/` module structure)  |
| `docker/`    | Dockerfile + docker-compose (serving, ingest, Redpanda, MLflow) |
| `dataset/`   | Raw CSV data (patients, vitals, labs, MIMIC-IV)      |
| `data/`      | Processed outputs: `datalake/`, `processed/`, `features/` |
| `models/`    | Trained model artifacts                              |
| `.github/`   | CI/CD workflows (`ci.yml`, `cd.yml`)                 |

---

## 🐍 Python & Code Style

### Language & Runtime
- **Python ≥ 3.10** — Use modern syntax: `match`, `type | None`, `list[str]` (not `List[str]`).
- All files use `from __future__ import annotations` for forward references.
- Type hints are **mandatory** on all function signatures and class attributes.

### Formatting & Linting
- **Ruff** is the sole linter/formatter. Config lives in `pyproject.toml`:
  - Line length: **100 characters**
  - Rules: `E`, `F`, `I`, `W` (ignoring `E501`)
  - Target: `py310`
- Run lint: `ruff check src/ config/ --ignore E501,F401`
- Do **NOT** introduce `black`, `flake8`, `isort`, or `pylint` — Ruff handles everything.

### Docstrings & Comments
- Every module starts with a banner docstring:
  ```python
  """
  G_One_Sync AI — Module Name
  ==============================
  Brief description of what this module does.
  """
  ```
- Use `═` section separators in docstrings for consistency with the existing style.
- Inline comments use `# ── Section Name ──────` separator style for visual structure.
- Preserve all existing comments and docstrings unless explicitly asked to change them.

### Import Ordering
Imports follow this order (enforced by Ruff `I` rules):
1. `__future__`
2. Standard library
3. Third-party packages
4. Local imports (`config.settings`, `src.*`)

### Naming Conventions
- Classes: `PascalCase` (e.g., `EvaluationMetrics`, `VitalSignsPayload`)
- Functions/methods: `snake_case` (e.g., `compute_shap_values`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `PROJECT_ROOT`, `DATASET_DIR`)
- Private methods: single underscore prefix `_compute_specificity()`
- File names: `snake_case.py` always

---

## ⚙️ Configuration System

**All configuration is centralized in `config/settings.py`** using `pydantic-settings`.

### Rules
- Every config group is a separate `BaseSettings` subclass with its own `env_prefix`.
- Singleton instances are created at module level (e.g., `model_settings = ModelSettings()`).
- **Never hardcode** paths, ports, model hyperparameters, or feature column names — always reference `config.settings`.
- Environment variable overrides follow the pattern: `JEEVAN_{GROUP}_{FIELD}` (e.g., `JEEVAN_MODEL_USE_GPU`).

### Config Groups

| Class              | Env Prefix          | Purpose                        |
| ------------------ | ------------------- | ------------------------------ |
| `DataSettings`     | `JEEVAN_DATA_`      | Dataset paths, output dirs     |
| `KafkaSettings`    | `JEEVAN_KAFKA_`     | Redpanda/Kafka broker config   |
| `FeatureSettings`  | `JEEVAN_FEATURE_`   | Feature engineering params, columns, splits |
| `APISettings`      | `JEEVAN_API_`       | Ingestion API host/port/CORS   |
| `ModelSettings`    | `JEEVAN_MODEL_`     | Training hyperparams, MLflow, GPU |
| `ServingSettings`  | `JEEVAN_SERVING_`   | Prediction server config       |

### Adding New Settings
1. Add a new `BaseSettings` subclass in `config/settings.py`.
2. Set `model_config = ConfigDict(env_prefix="JEEVAN_NEWGROUP_")`.
3. Create a singleton instance at the bottom of the file.
4. Import via `from config.settings import new_settings`.

---

## 🧠 Domain Context (Critical)

> **This is a clinical safety system.** Code changes can directly impact patient care decisions.
> Treat every change with the rigor of a medical device.

### Key Clinical Concepts
- **Deterioration prediction**: Binary classification — will this patient deteriorate within the next **12 hours**?
- **Target column**: `deterioration_next_12h` (binary: 0 = stable, 1 = deteriorating)
- **Class imbalance**: Deterioration events are rare (~5-10%). Always use class-weighted losses, AUPRC as primary metric.
- **Calibration matters**: Predicted probabilities must be well-calibrated (isotonic calibration achieves ECE → 10⁻⁹). Never skip calibration.

### Vital Signs & Valid Ranges
All payloads enforce clinical range validation via Pydantic:

| Vital             | Min  | Max  | Unit       |
| ----------------- | ---- | ---- | ---------- |
| Heart Rate        | 20   | 300  | BPM        |
| Respiratory Rate  | 4    | 60   | breaths/min|
| SpO2              | 50   | 100  | %          |
| Temperature       | 30.0 | 45.0 | °C         |
| Systolic BP       | 40   | 300  | mmHg       |
| Diastolic BP      | 20   | 200  | mmHg       |

**Do NOT relax these ranges** without clinical justification.

### Model Ensemble
The system uses a **3-model ensemble** with learned weights:
1. **XGBoost** — Tabular features (gradient boosted trees)
2. **BiLSTM** — Sequential temporal patterns
3. **Transformer** — Attention-based temporal analysis

All trainers inherit from `BaseTrainer` (abstract base class in `src/modeling/base_trainer.py`).
When adding a new model, subclass `BaseTrainer` and implement all abstract methods.

### Explainability is Non-Negotiable
- Every prediction must be explainable via **SHAP values**.
- Never remove or bypass the `/predict/explain` endpoint.
- SHAP waterfall plots are clinically required for regulatory compliance.

---

## 🧪 Testing

### Framework & Config
- **pytest** with `pytest-asyncio` for async tests.
- Config in `pyproject.toml`: `testpaths = ["tests"]`, `addopts = "-v --tb=short"`.
- Test files: `test_*.py`, test functions: `test_*`.

### Running Tests
```bash
# Full suite
pytest tests/ -v --cov=src --cov-report=term-missing --tb=short

# Single module
pytest tests/test_serving.py -v

# CI equivalent
ruff check src/ config/ --ignore E501,F401 && pytest tests/ -v
```

### Test Conventions
- Group tests into **classes by feature/endpoint** (e.g., `class TestHealthEndpoint`, `class TestPredictEndpoint`).
- Use `@pytest.fixture` for shared setup (test clients, mock data).
- FastAPI tests use `TestClient` from `fastapi.testclient`.
- Helper methods use `_make_request()` pattern for constructing test payloads.
- Tests must **mirror the `src/` structure**: `tests/test_ingestion.py` tests `src/ingestion/`, etc.

### What to Test
- ✅ API endpoint status codes and response schemas
- ✅ Pydantic validation rejects out-of-range clinical values
- ✅ Feature extraction produces correct column counts
- ✅ Model predictions return valid probability ranges `[0, 1]`
- ✅ SHAP explanations are non-empty for every feature
- ❌ Do NOT test third-party library internals (PyTorch, XGBoost)
- ❌ Do NOT write tests that require GPU or MIMIC-IV data

---

## 📡 API Conventions

### Two FastAPI Servers
1. **Ingestion API** — `src/ingestion/ingest_api.py` → port `8000`
2. **Serving API** — `src/serving/model_server.py` → port `8001`

### Endpoint Standards
- Use Pydantic `BaseModel` for all request/response schemas.
- All endpoints return JSON with a consistent structure.
- Use `str(Enum)` types for categorical fields (e.g., `ModelType`, `Gender`, `AdmissionType`).
- Include `Field(...)` with `ge`, `le`, `description` for all numeric fields.
- Health checks: `GET /health` returns `{"status": "healthy", "models_loaded": [...], "gpu_available": bool}`.

### Error Handling
- Use `HTTPException` with specific status codes (400, 404, 422, 500).
- Never expose stack traces or internal paths in error responses.
- Log errors with `loguru.logger.error()` before raising.

---

## 📊 Logging

- **loguru** is the sole logging library. Import: `from loguru import logger`.
- Auto-configured in `config/logging_config.py` (console + rotating file sink).
- **Never** use `print()` or stdlib `logging`. Always use `logger.info()`, `logger.warning()`, `logger.error()`.
- Log files: `logs/jeevansync_YYYY-MM-DD.log` (JSON structured, rotated daily, 30-day retention).

---

## 🔒 Security & Sensitive Data

### Patient Data
- **Never log or expose PHI** (Protected Health Information) — patient names, MRNs, SSNs, dates of birth.
- `patient_id` in code is a synthetic integer identifier, not real patient data.
- Dataset is from **MIMIC-IV demo** (de-identified). Do not introduce real patient data.

### Secrets & Environment
- `.env` files are gitignored. Never commit API keys, database credentials, or tokens.
- All sensitive config flows through environment variables → `pydantic-settings`.

### Model Artifacts
- Trained models are stored in `models/artifacts/` (gitignored).
- Never commit model weights (`.pt`, `.joblib`, `.json` model files) to version control.

---

## 🚀 CI/CD Pipeline

### CI (`.github/workflows/ci.yml`)
Runs on push to `main`/`develop` and PRs to `main`:
1. Matrix: Python 3.11, 3.12
2. Lint with Ruff
3. Validate all module imports
4. Run full test suite with coverage

### CD (`.github/workflows/cd.yml`)
Handles deployment after merge to `main`.

### Pre-Commit Checklist
Before any commit, ensure:
- [ ] `ruff check src/ config/` passes
- [ ] `pytest tests/ -v` passes
- [ ] No hardcoded paths or credentials
- [ ] New settings added to `config/settings.py` (not inline)
- [ ] New modules have corresponding test files
- [ ] Docstrings on all public classes and functions

---

## 🐳 Docker & Deployment

### Services (via `docker/docker-compose.yml`)

| Service          | Container Name         | Port  | Purpose            |
| ---------------- | ---------------------- | ----- | ------------------ |
| `prediction-api` | `g-one-sync-serving`   | 8001  | Model serving      |
| `ingest-api`     | `g-one-sync-ingest`    | 8000  | Data ingestion     |
| `redpanda`       | `g-one-sync-redpanda`  | 9092  | Kafka-compatible broker |
| `mlflow`         | `g-one-sync-mlflow`    | 5000  | Experiment tracking|

### GPU Access
- Serving container uses NVIDIA GPU passthrough (`deploy.resources.reservations.devices`).
- Set `JEEVAN_MODEL_USE_GPU=true` to enable CUDA inference.
- The development machine has an **RTX 3050 (4GB VRAM)** — be mindful of batch sizes and model memory.

---

## 📝 Commit & PR Conventions

### Commit Messages
Use conventional commits:
```
feat(modeling): add gradient accumulation for large batch training
fix(serving): handle NaN values in lab results payload
refactor(preprocessing): extract sliding window into separate module
docs(readme): update architecture diagram
test(ingestion): add edge case tests for vital sign validation
chore(ci): upgrade Python matrix to 3.12
```

### PR Rules
- One logical change per PR.
- Include test coverage for new features.
- Update docstrings if public interfaces change.
- If adding a new model architecture, update the ensemble integration and serving endpoints.
- Breaking changes to API schemas require versioning discussion.

---

## ⚠️ Common Pitfalls

1. **Don't skip calibration** — Raw model probabilities are not clinically usable. Always run isotonic calibration.
2. **Don't use accuracy as primary metric** — Class imbalance makes accuracy misleading. Use AUROC and AUPRC.
3. **Don't modify `FeatureSettings.vital_columns` or `lab_columns`** without updating all downstream consumers (preprocessing, modeling, serving, SHAP).
4. **Don't add synchronous blocking calls** in the FastAPI endpoints — use `async def` and background tasks.
5. **Don't train models without MLflow tracking** — every experiment must be logged for reproducibility.
6. **Don't ignore Kafka fallback** — The system has a local file queue fallback (`use_local_fallback`) for when Redpanda is unavailable. Preserve this.
7. **Don't remove the `__init__.py` files** — They contain module-level imports and docstrings.
8. **Don't introduce new dependencies** without adding them to `requirements.txt` with minimum version pins.

---

## 🗺️ Quick Reference: Key Files

| What you need                  | Where to find it                            |
| ------------------------------ | ------------------------------------------- |
| All configuration              | `config/settings.py`                        |
| Logging setup                  | `config/logging_config.py`                  |
| Data validation schemas        | `src/ingestion/schemas.py`                  |
| Feature engineering pipeline   | `src/preprocessing/pipeline.py`             |
| Training orchestration         | `src/modeling/train_pipeline.py`            |
| Base class for new models      | `src/modeling/base_trainer.py`              |
| XGBoost trainer                | `src/modeling/xgboost_trainer.py`           |
| BiLSTM trainer                 | `src/modeling/lstm_trainer.py`              |
| Transformer trainer            | `src/modeling/transformer_trainer.py`       |
| Ensemble logic                 | `src/modeling/ensemble.py`                  |
| Probability calibration        | `src/modeling/calibration.py`               |
| Hyperparameter optimization    | `src/modeling/hpo.py`                       |
| Prediction API                 | `src/serving/model_server.py`               |
| SHAP explanations              | `src/explainability/shap_explainer.py`      |
| Clinical dashboard             | `src/dashboard/app.py`                      |
| Drift detection                | `src/monitoring/drift_detector.py`          |
| Prometheus metrics             | `src/monitoring/metrics_collector.py`       |
| Docker deployment              | `docker/docker-compose.yml`                 |
| CI pipeline                    | `.github/workflows/ci.yml`                  |
