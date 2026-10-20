# Risk Analysis — ISO 14971

> **G_One_Sync AI — Risk Management File**
> Systematic Risk Analysis for Clinical Deterioration Prediction SaMD
> Per ISO 14971:2019 §5

---

## 1. Intended Use

G_One_Sync AI is a Software as a Medical Device (SaMD) intended to:
- **Predict clinical deterioration** in ICU patients within the next 12 hours
- **Provide decision support** to attending physicians and ICU nurses
- **Display explainable risk factors** via SHAP values to support clinical reasoning
- **Alert clinical staff** when patients exceed configurable risk thresholds

**The system does NOT:**
- Make autonomous treatment decisions
- Replace physician clinical judgment
- Serve as a primary patient monitoring device
- Provide diagnostic conclusions

## 2. Characteristics Affecting Safety

### 2.1 Patient Population

| Characteristic | Specification |
|---------------|--------------|
| Setting | Intensive Care Unit (ICU) |
| Patient type | Adult (≥18 years), critically ill |
| Conditions | Medical, surgical, cardiac, neurological ICU |
| Exclusions | Pediatric, neonatal, hospice/comfort care patients |

### 2.2 System Characteristics

| Characteristic | Value | Safety Relevance |
|---------------|-------|-----------------|
| Prediction horizon | 12 hours | Sufficient time for clinical intervention |
| Input data | Vital signs, lab results, demographics | Data quality directly affects prediction accuracy |
| Update frequency | Per new observation (real-time capable) | Stale predictions may miss rapid deterioration |
| Model ensemble | XGBoost + BiLSTM + Transformer | Redundancy reduces single-model failure |
| Calibration | Isotonic (ECE → 10⁻⁹) | Well-calibrated probabilities for clinical trust |

### 2.3 Use Environment

| Factor | Impact |
|--------|--------|
| ICU noise/distractions | May delay response to alerts |
| Shift changes | Risk information must persist across handoffs |
| Alert fatigue | Excessive false positives reduce trust |
| Network connectivity | System must degrade gracefully if network fails |
| Multiple data sources | EMR, bedside monitors, lab systems — integration complexity |

## 3. Hazard Identification

### H-001: False Negative — Failure to Predict Deterioration

| Attribute | Detail |
|-----------|--------|
| **Hazard** | System fails to identify a patient who will deteriorate |
| **Sequence of Events** | (1) Patient's condition worsening, (2) System predicts LOW risk, (3) Clinician does not escalate care, (4) Patient deteriorates |
| **Harm** | Delayed treatment, increased morbidity, potential death |
| **Severity** | S4 (Critical) to S5 (Catastrophic) |
| **Probability** | P3 (Occasional) — Based on target false negative rate <15% |
| **Pre-Control Risk** | 🔴 Unacceptable |
| **Risk Controls** | (1) Ensemble of 3 diverse models reduces individual model blindspots, (2) Class-weighted training emphasizes recall, (3) Confidence flagging when input data is incomplete, (4) Existing bedside monitors and nursing assessments as independent safety net |
| **Post-Control Probability** | P2 (Remote) — Ensemble + class weighting + external controls |
| **Residual Risk** | 🟡 ALARP |
| **Residual Risk Acceptable?** | Yes — Clinical benefit of catching 85%+ deterioration events outweighs residual risk, especially with independent external monitoring |

### H-002: False Positive — Incorrect Deterioration Alert

| Attribute | Detail |
|-----------|--------|
| **Hazard** | System predicts HIGH risk for a stable patient |
| **Sequence of Events** | (1) Stable patient, (2) System predicts HIGH risk, (3) Unnecessary clinical interventions ordered |
| **Harm** | Unnecessary procedures, alert fatigue, resource waste |
| **Severity** | S2 (Minor) |
| **Probability** | P4 (Probable) — ~20% FPR at standard operating threshold |
| **Pre-Control Risk** | 🟡 ALARP |
| **Risk Controls** | (1) SHAP explanations help clinicians evaluate plausibility, (2) Configurable alert thresholds per unit, (3) Trend display shows score trajectory (single spike vs sustained elevation), (4) Probability calibration provides accurate confidence |
| **Post-Control Probability** | P3 (Occasional) |
| **Residual Risk** | 🟢 Acceptable |

### H-003: Model Degradation Due to Data Drift

| Attribute | Detail |
|-----------|--------|
| **Hazard** | Model performance silently degrades as patient population changes |
| **Sequence of Events** | (1) New patient demographics differ from training data, (2) Model accuracy decreases, (3) Increased false negatives/positives undetected |
| **Harm** | Systematic prediction errors across multiple patients |
| **Severity** | S3 (Serious) to S4 (Critical) |
| **Probability** | P3 (Occasional) |
| **Pre-Control Risk** | 🔴 Unacceptable |
| **Risk Controls** | (1) Automated drift detection (`src/monitoring/drift_detector.py`) checks distributions hourly, (2) Performance monitoring tracks AUROC/AUPRC on incoming data, (3) Alert engine notifies when metrics drop below thresholds, (4) Continuous learning pipeline triggers retraining when drift detected |
| **Post-Control Probability** | P2 (Remote) |
| **Residual Risk** | 🟡 ALARP — Acceptable with ongoing monitoring |

### H-004: System Unavailability

| Attribute | Detail |
|-----------|--------|
| **Hazard** | Prediction service is offline or unresponsive |
| **Sequence of Events** | (1) API server crash/network failure, (2) No predictions available, (3) Clinicians lose supplementary decision support |
| **Harm** | Loss of supplementary monitoring tool |
| **Severity** | S2 (Minor) — System is supplementary, not primary monitoring |
| **Probability** | P3 (Occasional) |
| **Pre-Control Risk** | 🟡 ALARP |
| **Risk Controls** | (1) Docker container auto-restart (`restart: unless-stopped`), (2) Health check endpoint for monitoring, (3) Kafka local fallback queue preserves data during outages, (4) Dashboard shows clear "SYSTEM OFFLINE" status |
| **Post-Control Probability** | P2 (Remote) |
| **Residual Risk** | 🟢 Acceptable |

### H-005: Data Integrity Compromise

| Attribute | Detail |
|-----------|--------|
| **Hazard** | Corrupted or tampered input data produces incorrect predictions |
| **Sequence of Events** | (1) Malformed vital signs or lab values enter the system, (2) Feature engineering produces anomalous features, (3) Model generates misleading predictions |
| **Harm** | Clinical decisions based on corrupted data |
| **Severity** | S3 (Serious) |
| **Probability** | P2 (Remote) |
| **Pre-Control Risk** | 🟡 ALARP |
| **Risk Controls** | (1) Pydantic schema validation with clinical range checks (HR 20-300, SpO2 50-100, etc.), (2) Outlier detection in feature pipeline, (3) Input completeness thresholds, (4) Audit logging for all data ingestion |
| **Post-Control Probability** | P1 (Improbable) |
| **Residual Risk** | 🟢 Acceptable |

### H-006: Cybersecurity Breach

| Attribute | Detail |
|-----------|--------|
| **Hazard** | Unauthorized access to system or patient data |
| **Sequence of Events** | (1) Attacker gains API access, (2) Patient data exfiltrated or predictions manipulated |
| **Harm** | PHI breach, manipulated clinical decisions |
| **Severity** | S3 (Serious) to S4 (Critical) |
| **Probability** | P2 (Remote) |
| **Pre-Control Risk** | 🟡 ALARP to 🔴 Unacceptable |
| **Risk Controls** | (1) JWT authentication on mobile/external endpoints, (2) Network segmentation (ICU VLAN), (3) No PHI in logs (synthetic patient IDs only), (4) SBOM vulnerability scanning in CI, (5) TLS encryption for all API traffic |
| **Post-Control Probability** | P1 (Improbable) |
| **Residual Risk** | 🟢 Acceptable |

### H-007: Algorithmic Bias

| Attribute | Detail |
|-----------|--------|
| **Hazard** | Model performs differently across demographic subgroups |
| **Sequence of Events** | (1) Training data underrepresents certain demographics, (2) Model has lower sensitivity for underrepresented groups, (3) Disparate clinical outcomes |
| **Harm** | Health equity violations, missed deterioration in vulnerable populations |
| **Severity** | S3 (Serious) |
| **Probability** | P3 (Occasional) |
| **Pre-Control Risk** | 🔴 Unacceptable |
| **Risk Controls** | (1) Subgroup performance analysis by age, sex, admission type, (2) Stratified evaluation in validation study, (3) Continuous monitoring of per-subgroup metrics, (4) Fairness constraints in model training if disparities detected |
| **Post-Control Probability** | P2 (Remote) |
| **Residual Risk** | 🟡 ALARP — Acceptable with ongoing monitoring |

## 4. Risk Summary

| Hazard ID | Description | Pre-Control | Post-Control | Status |
|-----------|-------------|:-----------:|:------------:|:------:|
| H-001 | False Negative | 🔴 | 🟡 ALARP | ✅ Acceptable |
| H-002 | False Positive | 🟡 | 🟢 | ✅ Acceptable |
| H-003 | Data Drift | 🔴 | 🟡 ALARP | ✅ Acceptable |
| H-004 | Unavailability | 🟡 | 🟢 | ✅ Acceptable |
| H-005 | Data Integrity | 🟡 | 🟢 | ✅ Acceptable |
| H-006 | Cybersecurity | 🟡/🔴 | 🟢 | ✅ Acceptable |
| H-007 | Algorithmic Bias | 🔴 | 🟡 ALARP | ✅ Acceptable |

**Overall Residual Risk Assessment**: Acceptable — the clinical benefit of early deterioration detection substantially outweighs the residual risks, particularly given the system's supplementary role alongside existing clinical monitoring.

## 5. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial release |
