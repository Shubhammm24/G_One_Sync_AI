# Design Inputs

> **G_One_Sync AI — Design Controls**
> Design Input Requirements Specification
> Per FDA 21 CFR 820.30(c) & IEC 62304:2006 §5.2

---

## 1. Functional Requirements

### DI-001: Deterioration Prediction
- **Source**: UN-001
- **Requirement**: The system SHALL predict clinical deterioration within the next 12 hours for ICU patients using vital signs, lab results, and demographic data
- **Acceptance Criteria**: Binary classification output with probability score ∈ [0.0, 1.0]
- **Verification**: Unit test — prediction endpoint returns valid probability

### DI-002: Ensemble Prediction
- **Source**: UN-001
- **Requirement**: The system SHALL use an ensemble of at least 3 diverse model architectures to generate predictions
- **Rationale**: Ensemble diversity reduces single-model failure modes
- **Acceptance Criteria**: Ensemble combines XGBoost (tabular), BiLSTM (sequential), and Transformer (attention-based) predictions with learned weights
- **Verification**: Model loading health check confirms 3 models loaded

### DI-003: Continuous Real-Time Scoring
- **Source**: UN-001
- **Requirement**: The system SHALL generate a new prediction within 60 seconds of receiving updated patient observations
- **Acceptance Criteria**: p99 prediction latency < 500ms (single patient)
- **Verification**: Load test with concurrent predictions

## 2. Performance Requirements

### DI-004: SHAP Explainability
- **Source**: UN-002
- **Requirement**: The system SHALL provide SHAP-based feature importance explanations for every prediction
- **Acceptance Criteria**: Top 10 contributing features displayed with direction (positive/negative contribution) and magnitude
- **Verification**: API `/predict/explain` returns non-empty SHAP values for all features

### DI-005: Model Discrimination
- **Source**: UN-001
- **Requirement**: The ensemble model SHALL achieve AUROC ≥ 0.85 and AUPRC ≥ 0.50 on a held-out test set
- **Acceptance Criteria**: Metrics calculated on ≥1000 test samples with 95% CI
- **Verification**: Cross-validated evaluation report

### DI-006: Model Calibration
- **Source**: UN-012
- **Requirement**: Predicted probabilities SHALL be well-calibrated with Expected Calibration Error (ECE) < 0.01
- **Acceptance Criteria**: Isotonic calibration applied; calibration curve within ±2% of perfect diagonal
- **Verification**: Calibration analysis with reliability diagram

### DI-007: Sensitivity/Specificity Targets
- **Source**: UN-001
- **Requirement**: At the default operating threshold, the system SHALL achieve:
  - Sensitivity (Recall) ≥ 80%
  - Specificity ≥ 75%
  - Negative Predictive Value (NPV) ≥ 95%
- **Rationale**: High NPV is clinically critical — a "low risk" prediction must be reliable
- **Verification**: Holdout test set evaluation

## 3. Alerting Requirements

### DI-008: Real-Time Alert Delivery
- **Source**: UN-003
- **Requirement**: The system SHALL deliver alerts within 30 seconds of prediction crossing the configured threshold
- **Acceptance Criteria**: Alert timestamp minus prediction timestamp < 30s
- **Verification**: End-to-end alerting test

### DI-009: Multi-Channel Alerting
- **Source**: UN-003
- **Requirement**: Alerts SHALL be delivered via: (1) Dashboard visual alert, (2) WebSocket notification, (3) Push notification to mobile app
- **Acceptance Criteria**: All 3 channels receive alert for critical threshold crossing
- **Verification**: Integration test per channel

### DI-010: Configurable Alert Thresholds
- **Source**: UN-004
- **Requirement**: Alert thresholds SHALL be configurable per ICU unit with at least 3 risk levels:
  - LOW: probability < configurable threshold (default 0.3)
  - MODERATE: probability 0.3–0.6 (default)
  - HIGH: probability 0.6–0.8 (default)
  - CRITICAL: probability > 0.8 (default)
- **Acceptance Criteria**: Thresholds modifiable via API or configuration
- **Verification**: API configuration test

## 4. User Interface Requirements

### DI-011: Patient Dashboard
- **Source**: UN-006
- **Requirement**: The clinical dashboard SHALL display:
  - Real-time patient risk scores with color-coded severity
  - Risk trend chart (24-hour history)
  - Current vital signs
  - SHAP explanation summary (top 5 features)
  - Patient demographics
- **Acceptance Criteria**: All elements render within 3 seconds of page load
- **Verification**: Dashboard functional test

### DI-012: Mobile Application
- **Source**: UN-005
- **Requirement**: The mobile companion app SHALL provide:
  - Patient watchlist with current risk scores
  - Individual patient detail view with SHAP chart
  - Push notification alerts
  - Secure authentication (JWT + biometric)
- **Acceptance Criteria**: App functional on iOS 15+ and Android 12+
- **Verification**: Mobile E2E tests

## 5. Data & Integration Requirements

### DI-013: Input Data Validation
- **Source**: H-005 (Risk Control)
- **Requirement**: The system SHALL validate all incoming clinical data against physiological ranges:
  - Heart Rate: 20–300 BPM
  - Respiratory Rate: 4–60 breaths/min
  - SpO2: 50–100%
  - Temperature: 30.0–45.0 °C
  - Systolic BP: 40–300 mmHg
  - Diastolic BP: 20–200 mmHg
- **Acceptance Criteria**: Out-of-range values rejected with HTTP 422 and descriptive error message
- **Verification**: Pydantic schema validation tests

### DI-014: FHIR R4 Integration
- **Source**: UN-007
- **Requirement**: The system SHALL accept patient data via FHIR R4 Observation resources
- **Acceptance Criteria**: FHIR Bundle containing Observation resources mapped to internal schemas
- **Verification**: FHIR integration tests with mock bundles

### DI-015: HL7v2 Integration
- **Source**: UN-007
- **Requirement**: The system SHALL accept patient data via HL7v2 ORU^R01 messages
- **Acceptance Criteria**: HL7v2 messages parsed and mapped to internal schemas
- **Verification**: HL7 integration tests with mock messages

## 6. Reliability Requirements

### DI-016: System Availability
- **Source**: UN-009
- **Requirement**: The prediction service SHALL maintain ≥99.9% uptime (maximum 8.76 hours downtime per year)
- **Acceptance Criteria**: Health check monitoring with auto-restart on failure
- **Verification**: Docker restart policy + health check endpoint

### DI-017: Graceful Degradation
- **Source**: UN-009, H-004
- **Requirement**: When individual components fail, the system SHALL degrade gracefully:
  - Kafka unavailable → local file queue fallback
  - GPU unavailable → CPU inference fallback
  - One ensemble model fails → remaining models generate prediction with reduced confidence
- **Verification**: Fault injection tests

## 7. Security Requirements

### DI-018: Authentication
- **Source**: UN-010
- **Requirement**: External API access SHALL require JWT authentication with configurable token expiration
- **Acceptance Criteria**: Unauthenticated requests return HTTP 401
- **Verification**: Auth middleware tests

### DI-019: Data Protection
- **Source**: UN-010
- **Requirement**: The system SHALL NOT log, store, or transmit Protected Health Information (PHI)
- **Acceptance Criteria**: Patient identifiers are synthetic integers; no names, MRNs, SSNs, or DOBs in logs
- **Verification**: Log audit + code review

## 8. Traceability Matrix

| Design Input | User Need | Hazard | Design Output | Test |
|:------------:|:---------:|:------:|:-------------:|:----:|
| DI-001 | UN-001 | — | `model_server.py` `/predict` | `test_serving.py` |
| DI-002 | UN-001 | H-001 | `ensemble.py` | `test_modeling.py` |
| DI-003 | UN-001 | — | `model_server.py` | Load test |
| DI-004 | UN-002 | — | `shap_explainer.py` | `test_serving.py` |
| DI-005 | UN-001 | H-001 | Training pipeline | `test_modeling.py` |
| DI-006 | UN-012 | H-001 | `calibration.py` | Calibration report |
| DI-007 | UN-001 | H-001 | Threshold config | Performance report |
| DI-008 | UN-003 | H-020 | Alert engine | Integration test |
| DI-009 | UN-003 | H-020 | Multi-channel alerts | `test_mobile_api.py` |
| DI-010 | UN-004 | H-021 | Configuration API | `test_serving.py` |
| DI-011 | UN-006 | — | `dashboard/app.py` | Usability test |
| DI-012 | UN-005 | — | `mobile/` | Mobile E2E |
| DI-013 | — | H-005 | `schemas.py` | `test_ingestion.py` |
| DI-014 | UN-007 | — | `fhir_mapper.py` | `test_emr_integration.py` |
| DI-015 | UN-007 | — | `hl7_mapper.py` | `test_emr_integration.py` |
| DI-016 | UN-009 | H-004 | Docker config | Uptime monitoring |
| DI-017 | UN-009 | H-004 | Fallback logic | Fault injection |
| DI-018 | UN-010 | H-006 | `auth.py` | `test_mobile_api.py` |
| DI-019 | UN-010 | H-006 | Logging config | Log audit |

## 9. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial — 19 design inputs |
