# Failure Mode and Effects Analysis (FMEA)

> **G_One_Sync AI — Risk Management File**
> FMEA for Clinical Deterioration Prediction System
> Per ISO 14971:2019 Annex H

---

## 1. FMEA Methodology

### Scoring Scales

**Severity (S)**: 1–10

| Score | Level | Description |
|:-----:|-------|-------------|
| 1 | None | No effect |
| 2–3 | Minor | Slight inconvenience, no clinical impact |
| 4–5 | Moderate | Delay in notification, minor clinical impact |
| 6–7 | High | Potential for delayed treatment |
| 8–9 | Very High | Significant risk of patient harm |
| 10 | Hazardous | Death or life-threatening injury |

**Occurrence (O)**: 1–10

| Score | Level | Frequency |
|:-----:|-------|-----------|
| 1 | Almost Impossible | < 1 in 10⁶ |
| 2–3 | Low | 1 in 10⁵ to 1 in 10⁴ |
| 4–5 | Moderate | 1 in 10³ to 1 in 10² |
| 6–7 | High | 1 in 10¹ to 1 in 10² |
| 8–9 | Very High | > 1 in 10 |
| 10 | Almost Certain | Every occurrence |

**Detection (D)**: 1–10

| Score | Level | Description |
|:-----:|-------|-------------|
| 1 | Almost Certain | Automated detection with immediate alert |
| 2–3 | High | Automated detection during routine monitoring |
| 4–5 | Moderate | Detected during periodic review or testing |
| 6–7 | Low | Detected only through investigation |
| 8–9 | Very Low | Unlikely to be detected |
| 10 | Undetectable | Cannot be detected |

**Risk Priority Number (RPN)** = S × O × D — Maximum 1000, threshold for action: **RPN > 100**

---

## 2. FMEA Worksheet

### 2.1 Data Ingestion Failures

| ID | Failure Mode | Effect | S | Cause | O | Current Controls | D | RPN | Action Required |
|----|-------------|--------|:-:|-------|:-:|-----------------|:-:|:---:|:-:|
| FM-001 | Missing vital signs in payload | Incomplete feature vector → degraded prediction accuracy | 6 | Sensor disconnect, EMR integration gap | 5 | Pydantic schema validation rejects incomplete records | 2 | **60** | ✅ Acceptable |
| FM-002 | Out-of-range vital values | Physiologically impossible input → model confusion | 7 | Sensor malfunction, data entry error | 4 | Clinical range validation (HR 20-300, SpO2 50-100, etc.) | 1 | **28** | ✅ Acceptable |
| FM-003 | Kafka broker unavailable | Data not ingested in real-time | 4 | Network failure, broker crash | 3 | Local file fallback queue (`use_local_fallback=True`) | 2 | **24** | ✅ Acceptable |
| FM-004 | Duplicate data submission | Over-counting observations → skewed temporal features | 5 | EMR retry logic, network timeout + retry | 4 | Deduplication by (patient_id, timestamp) | 3 | **60** | ✅ Acceptable |
| FM-005 | Patient ID mismatch | Data attributed to wrong patient | 9 | EMR mapping error, ID collision | 2 | Patient ID validation in schema, audit logging | 3 | **54** | ✅ Acceptable |

### 2.2 Preprocessing & Feature Engineering Failures

| ID | Failure Mode | Effect | S | Cause | O | Current Controls | D | RPN | Action Required |
|----|-------------|--------|:-:|-------|:-:|-----------------|:-:|:---:|:-:|
| FM-006 | NaN propagation in features | Model receives NaN → prediction failure or garbage output | 7 | Missing imputation, division by zero | 3 | KNN imputation, NaN checks in pipeline | 2 | **42** | ✅ Acceptable |
| FM-007 | Feature scaling mismatch | Features normalized differently than training → shifted predictions | 6 | Scaler not loaded, scaler version mismatch | 2 | Scaler artifact versioned with model | 3 | **36** | ✅ Acceptable |
| FM-008 | Incorrect sliding window | Temporal features computed over wrong time span | 6 | Clock skew, timestamp parsing error | 2 | Unit tests for window calculations | 4 | **48** | ✅ Acceptable |
| FM-009 | New lab test codes not mapped | Unknown lab tests ignored → missing features | 5 | EMR vocabulary expansion | 3 | Strict mapping with logged warnings for unmapped codes | 4 | **60** | ✅ Acceptable |

### 2.3 Model Prediction Failures

| ID | Failure Mode | Effect | S | Cause | O | Current Controls | D | RPN | Action Required |
|----|-------------|--------|:-:|-------|:-:|-----------------|:-:|:---:|:-:|
| FM-010 | XGBoost model file corrupted | Model fails to load → no predictions | 5 | Disk failure, incomplete write | 1 | Checksum validation on load, health check endpoint | 1 | **5** | ✅ Acceptable |
| FM-011 | BiLSTM/Transformer GPU OOM | CUDA out-of-memory during inference | 5 | Large batch size, GPU memory pressure | 3 | Batch size limiting, CPU fallback inference | 2 | **30** | ✅ Acceptable |
| FM-012 | Ensemble weight miscalibration | One model dominates incorrectly | 7 | Learned weights not updated after model retrain | 2 | Ensemble weights stored with model artifacts, validated on load | 3 | **42** | ✅ Acceptable |
| FM-013 | Calibration model missing | Raw probabilities served without isotonic calibration | 8 | Calibrator file not deployed | 2 | Startup health check verifies calibrator loaded | 1 | **16** | ✅ Acceptable |
| FM-014 | Model version mismatch | Feature names/order incompatible with model | 7 | Deployment error, partial model update | 2 | Feature name validation in ModelRegistry | 2 | **28** | ✅ Acceptable |
| FM-015 | Prediction exceeds [0,1] range | Clinically meaningless probability | 6 | Numerical instability in ensemble | 1 | `np.clip(prediction, 0.0, 1.0)` post-processing | 1 | **6** | ✅ Acceptable |

### 2.4 Serving API Failures

| ID | Failure Mode | Effect | S | Cause | O | Current Controls | D | RPN | Action Required |
|----|-------------|--------|:-:|-------|:-:|-----------------|:-:|:---:|:-:|
| FM-016 | API server crash | No predictions available | 4 | Unhandled exception, memory leak | 2 | Docker auto-restart, structured error handling | 1 | **8** | ✅ Acceptable |
| FM-017 | High latency response | Clinician waits too long for prediction | 4 | GPU contention, large batch, network | 3 | Response timeout, async processing, load monitoring | 2 | **24** | ✅ Acceptable |
| FM-018 | SHAP explanation missing | Prediction served without explanation | 3 | SHAP computation timeout | 3 | SHAP timeout with fallback (serve prediction without SHAP) | 2 | **18** | ✅ Acceptable |
| FM-019 | Authentication bypass | Unauthorized API access | 8 | JWT vulnerability, misconfigured auth | 2 | JWT validation middleware, rate limiting | 3 | **48** | ✅ Acceptable |

### 2.5 Alerting & Notification Failures

| ID | Failure Mode | Effect | S | Cause | O | Current Controls | D | RPN | Action Required |
|----|-------------|--------|:-:|-------|:-:|-----------------|:-:|:---:|:-:|
| FM-020 | Critical alert not delivered | Clinician unaware of high-risk patient | 8 | Push notification failure, phone silent | 3 | Multi-channel alerting (push + dashboard + WebSocket), escalation chain | 3 | **72** | ⚠️ Monitor closely |
| FM-021 | Alert fatigue from false positives | Clinicians ignore alerts | 7 | Low specificity threshold | 5 | Configurable thresholds, SHAP context, trend display | 4 | **140** | 🔴 **Action Required** |
| FM-022 | Delayed alert delivery | Alert arrives after clinical window | 7 | Queue backlog, network latency | 2 | Priority queue for CRITICAL alerts, WebSocket for real-time | 2 | **28** | ✅ Acceptable |

---

## 3. High-RPN Items — Action Plans

### FM-021: Alert Fatigue (RPN = 140)

**Root Cause**: Default alert threshold may generate excessive false positives, especially in high-acuity ICUs where baseline vital sign abnormalities are common.

**Mitigation Plan**:
1. Implement unit-specific alert threshold customization (not one-size-fits-all)
2. Add "alert suppression" for acknowledged alerts with configurable cooldown period
3. Display prediction trend (slope) — only alert when risk is rising, not when it's stably elevated
4. Track alert-to-action ratio as a quality metric
5. Conduct usability testing with ICU nurses to optimize alert presentation

**Target RPN**: ≤ 63 (S=7 × O=3 × D=3)

---

## 4. FMEA Summary

| Risk Level | Count | Percentage |
|-----------|:-----:|:---------:|
| 🟢 Acceptable (RPN ≤ 100) | 21 | 95.5% |
| ⚠️ Monitor (RPN 70–100) | 0 | 0% |
| 🔴 Action Required (RPN > 100) | 1 | 4.5% |

**Next Review Date**: Quarterly, or upon any design change per SOP-004.

## 5. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial FMEA — 22 failure modes identified |
