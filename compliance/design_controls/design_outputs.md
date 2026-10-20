# Design Outputs

> **G_One_Sync AI — Design Controls**
> Design Output Documentation
> Per FDA 21 CFR 820.30(d) & IEC 62304:2006 §5.4

---

## 1. Software Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    G_One_Sync AI System                          │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│  Ingestion  │ Preprocessing│   Modeling   │     Serving        │
│  Layer      │  Pipeline    │   Ensemble   │     API            │
│             │              │              │                    │
│ • FHIR R4   │ • Feature    │ • XGBoost    │ • /predict         │
│ • HL7v2     │   extraction │ • BiLSTM     │ • /predict/explain │
│ • Kafka     │ • Imputation │ • Transformer│ • /predict/batch   │
│ • REST API  │ • Windowing  │ • Calibration│ • /health          │
│ • Schemas   │ • Validation │ • Ensemble   │ • /mobile/*        │
└──────┬──────┴──────┬───────┴──────┬───────┴────────┬───────────┘
       │             │              │                │
┌──────▼──────┬──────▼───────┬──────▼───────┬────────▼───────────┐
│ Monitoring  │Explainability│  Dashboard   │   Mobile App       │
│             │              │              │                    │
│ • Prometheus│ • SHAP       │ • Plotly     │ • React Native     │
│ • Drift     │ • Waterfall  │ • WebSocket  │ • Push Alerts      │
│ • Alerting  │ • Global     │ • Real-time  │ • Offline Cache    │
└─────────────┴──────────────┴──────────────┴────────────────────┘
```

### Module Design Outputs

| Design Input | Design Output Module | Key Files | Implementation Summary |
|:------------:|---------------------|-----------|----------------------|
| DI-001 | `src/serving/model_server.py` | `POST /predict` endpoint | Accepts patient features, returns calibrated probability |
| DI-002 | `src/modeling/ensemble.py` | `EnsemblePredictor` class | Weighted average of 3 model predictions |
| DI-003 | `src/serving/model_server.py` | `ModelRegistry` class | Preloaded models with GPU inference, <500ms p99 |
| DI-004 | `src/explainability/shap_explainer.py` | `SHAPExplainer` class | SHAP TreeExplainer (XGBoost) + DeepExplainer (PyTorch) |
| DI-005 | `src/modeling/train_pipeline.py` | Training orchestration | Cross-validated training with MLflow tracking |
| DI-006 | `src/modeling/calibration.py` | `CalibratedModel` class | Isotonic calibration with ECE measurement |
| DI-007 | `config/settings.py` | `ModelSettings` class | Configurable thresholds and operating points |
| DI-008 | `src/monitoring/alerting.py` | `AlertingEngine` class | Real-time alert delivery with priority queue |
| DI-009 | `src/serving/push_notifications.py` | `NotificationService` | FCM push + WebSocket + dashboard alerts |
| DI-010 | `config/settings.py` | `ServingSettings` class | Per-unit configurable thresholds |
| DI-011 | `src/dashboard/app.py` | Plotly Dash application | Real-time dashboard with risk trends |
| DI-012 | `mobile/` | React Native Expo app | iOS/Android companion app |
| DI-013 | `src/ingestion/schemas.py` | Pydantic BaseModel schemas | Clinical range validation |
| DI-014 | `src/ingestion/emr/fhir_mapper.py` | `FHIRMapper` class | FHIR R4 → internal schema mapping |
| DI-015 | `src/ingestion/emr/hl7_mapper.py` | `HL7Mapper` class | HL7v2 → internal schema mapping |
| DI-016 | `docker/docker-compose.yml` | Container orchestration | Auto-restart, health checks |
| DI-017 | Multiple modules | Fallback implementations | Kafka→local, GPU→CPU, ensemble→partial |
| DI-018 | `src/serving/auth.py` | JWT middleware | Token-based authentication |
| DI-019 | `config/logging_config.py` | Loguru configuration | PHI-free structured logging |

## 2. API Contract Specifications

### Prediction API (Port 8001)

| Endpoint | Method | Request Schema | Response Schema | Design Input |
|----------|--------|---------------|----------------|:------------:|
| `/predict` | POST | `PredictionRequest` | `PredictionResponse` | DI-001 |
| `/predict/batch` | POST | `BatchPredictionRequest` | `BatchPredictionResponse` | DI-001 |
| `/predict/explain` | POST | `PredictionRequest` | `ExplainResponse` | DI-004 |
| `/health` | GET | — | `HealthResponse` | DI-016 |
| `/mobile/patient/{id}/summary` | GET | Path param | `PatientSummary` | DI-012 |
| `/mobile/patients/watchlist` | GET | Query params | `WatchlistResponse` | DI-012 |

### Ingestion API (Port 8000)

| Endpoint | Method | Request Schema | Response Schema | Design Input |
|----------|--------|---------------|----------------|:------------:|
| `/ingest/vitals` | POST | `VitalSignsPayload` | `IngestResponse` | DI-013 |
| `/ingest/labs` | POST | `LabResultsPayload` | `IngestResponse` | DI-013 |
| `/ingest/batch` | POST | `BatchPayload` | `IngestResponse` | DI-013 |
| `/emr/fhir/webhook` | POST | FHIR Bundle | `IngestResponse` | DI-014 |

## 3. Model Artifacts

| Artifact | Format | Location | Size (typical) |
|----------|--------|----------|:--------------:|
| XGBoost model | `.json` (XGBoost native) | `models/artifacts/xgboost_model.json` | ~5 MB |
| BiLSTM model | `.pt` (PyTorch state dict) | `models/artifacts/bilstm_model.pt` | ~2 MB |
| Transformer model | `.pt` (PyTorch state dict) | `models/artifacts/transformer_model.pt` | ~8 MB |
| Ensemble weights | `.json` | `models/artifacts/ensemble_weights.json` | <1 KB |
| Isotonic calibrator | `.joblib` | `models/artifacts/calibrator.joblib` | <100 KB |
| Feature scaler | `.joblib` | `models/artifacts/scaler.joblib` | <50 KB |

## 4. Configuration Schema

All configuration is centralized in `config/settings.py` using `pydantic-settings`:

| Settings Class | Env Prefix | Key Parameters |
|---------------|-----------|---------------|
| `DataSettings` | `JEEVAN_DATA_` | Dataset paths, output directories |
| `KafkaSettings` | `JEEVAN_KAFKA_` | Broker config, topics, fallback |
| `FeatureSettings` | `JEEVAN_FEATURE_` | Feature columns, window sizes, splits |
| `APISettings` | `JEEVAN_API_` | Ingestion API host/port/CORS |
| `ModelSettings` | `JEEVAN_MODEL_` | Training hyperparams, MLflow, GPU |
| `ServingSettings` | `JEEVAN_SERVING_` | Prediction server, auth, thresholds |
| `EMRSettings` | `JEEVAN_EMR_` | FHIR/HL7 integration config |
| `FederatedSettings` | `JEEVAN_FL_` | Federated learning config |
| `ContinuousSettings` | `JEEVAN_CL_` | Continuous learning config |

## 5. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial design outputs |
