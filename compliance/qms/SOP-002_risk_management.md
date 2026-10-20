# SOP-002: Risk Management

> **G_One_Sync AI — Quality Management System**
> Standard Operating Procedure for Risk Management
> Per ISO 14971:2019 — Medical Devices — Application of Risk Management

---

## 1. Purpose

Establish a systematic process for identifying, evaluating, controlling, and monitoring risks associated with G_One_Sync AI throughout its lifecycle. This SOP ensures that residual risks are acceptable and that the overall risk/benefit ratio favors clinical use.

## 2. Scope

This SOP covers all hazards and hazardous situations arising from:
- AI/ML model predictions (false positives, false negatives)
- Data quality issues (missing values, out-of-range inputs, sensor failures)
- Software defects (API failures, model loading errors, incorrect feature engineering)
- Cybersecurity vulnerabilities (unauthorized access, data tampering)
- Human factors (alert fatigue, misinterpretation of risk scores)
- Integration failures (EMR connectivity, Kafka broker unavailability)

## 3. Definitions

| Term | Definition |
|------|-----------|
| **Hazard** | Potential source of harm |
| **Hazardous Situation** | Circumstance in which people, property, or environment are exposed to a hazard |
| **Harm** | Physical injury or damage to health |
| **Risk** | Combination of probability of occurrence and severity of harm |
| **Residual Risk** | Risk remaining after risk control measures have been applied |
| **Risk Acceptability** | Determination that residual risk is acceptable per defined criteria |

## 4. Risk Management Process

### 4.1 Risk Management Plan

The risk management plan for G_One_Sync AI defines:

| Element | Specification |
|---------|--------------|
| **Scope** | All SaMD functions contributing to clinical decision support |
| **Life Cycle Phases** | Design, Development, Verification, Validation, Post-Market |
| **Risk Acceptability Criteria** | See §4.5 |
| **Verification Activities** | See §4.6 |
| **Post-Market Monitoring** | Drift detection, performance monitoring, complaint tracking |

### 4.2 Risk Analysis

Risk analysis is performed using two complementary methods:

1. **Failure Mode and Effects Analysis (FMEA)** — Bottom-up analysis of component failures
   - Documented in `compliance/risk_management/fmea.md`
2. **Hazard Analysis** — Top-down identification of clinical hazards
   - Documented in `compliance/risk_management/risk_analysis.md`

### 4.3 Severity Classification

| Level | Category | Description | Clinical Example |
|-------|----------|-------------|-----------------|
| **S1** | Negligible | Inconvenience or temporary discomfort | Delayed non-urgent alert |
| **S2** | Minor | Temporary injury, not requiring intervention | Unnecessary lab order from false positive |
| **S3** | Serious | Injury requiring medical intervention | Delayed treatment due to missed deterioration |
| **S4** | Critical | Life-threatening injury | Cardiac arrest missed by false negative |
| **S5** | Catastrophic | Death | Death due to undetected sepsis |

### 4.4 Probability Classification

| Level | Category | Frequency |
|-------|----------|-----------|
| **P1** | Improbable | < 1 in 1,000,000 predictions |
| **P2** | Remote | 1 in 100,000 to 1 in 1,000,000 |
| **P3** | Occasional | 1 in 10,000 to 1 in 100,000 |
| **P4** | Probable | 1 in 1,000 to 1 in 10,000 |
| **P5** | Frequent | > 1 in 1,000 predictions |

### 4.5 Risk Acceptability Matrix

|  | **S1** Negligible | **S2** Minor | **S3** Serious | **S4** Critical | **S5** Catastrophic |
|--|:-:|:-:|:-:|:-:|:-:|
| **P5** Frequent | 🟡 ALARP | 🔴 Unacceptable | 🔴 Unacceptable | 🔴 Unacceptable | 🔴 Unacceptable |
| **P4** Probable | 🟢 Acceptable | 🟡 ALARP | 🔴 Unacceptable | 🔴 Unacceptable | 🔴 Unacceptable |
| **P3** Occasional | 🟢 Acceptable | 🟡 ALARP | 🟡 ALARP | 🔴 Unacceptable | 🔴 Unacceptable |
| **P2** Remote | 🟢 Acceptable | 🟢 Acceptable | 🟡 ALARP | 🟡 ALARP | 🔴 Unacceptable |
| **P1** Improbable | 🟢 Acceptable | 🟢 Acceptable | 🟢 Acceptable | 🟡 ALARP | 🟡 ALARP |

- 🟢 **Acceptable**: Risk is broadly acceptable without further controls
- 🟡 **ALARP**: Risk must be reduced As Low As Reasonably Practicable
- 🔴 **Unacceptable**: Risk must be eliminated or reduced before release

### 4.6 Risk Control Measures

Risk control measures shall be applied in the following priority order (ISO 14971 §7):

1. **Inherent Safety by Design**
   - Clinical range validation in Pydantic schemas (e.g., HR 20–300 BPM)
   - Mandatory calibration of predicted probabilities
   - Ensemble modeling to reduce single-model failure modes
   - Class-weighted loss functions to minimize false negatives

2. **Protective Measures in the Device**
   - Confidence thresholds: predictions below confidence threshold flagged as "Insufficient Data"
   - Input completeness checks: minimum vital signs required before prediction
   - Model drift detection with automatic alerting
   - Fallback to rule-based scoring (NEWS2) when ML models are unavailable

3. **Information for Safety (Labeling)**
   - Intended use statement: "Clinical decision support — physician review required"
   - Model performance characteristics displayed with every prediction
   - SHAP explanations for every prediction to support clinical reasoning
   - Training and onboarding materials for clinical staff

### 4.7 Residual Risk Evaluation

After all risk controls are applied:
1. Individual residual risks shall be re-assessed using the acceptability matrix
2. Overall residual risk shall be evaluated against clinical benefit
3. Risk/benefit analysis documented in `compliance/risk_management/risk_benefit_analysis.md`

### 4.8 Risk Management Report

The risk management report shall confirm:
- [ ] All identified hazards have been analyzed
- [ ] Risk control measures have been implemented and verified
- [ ] All residual risks are acceptable or ALARP
- [ ] Overall residual risk is acceptable considering clinical benefit
- [ ] Post-market risk monitoring is established

## 5. Post-Market Risk Monitoring

Ongoing risk monitoring includes:

| Activity | Frequency | Owner | Source |
|----------|-----------|-------|--------|
| Model performance monitoring | Continuous | ML Engineering | `src/monitoring/metrics_collector.py` |
| Data drift detection | Hourly | ML Engineering | `src/monitoring/drift_detector.py` |
| Complaint analysis | As received | QA/RA | `SOP-005_complaint_handling.md` |
| Literature review | Quarterly | Clinical | Published clinical AI studies |
| Regulatory updates | Quarterly | QA/RA | FDA, EU MDR guidance documents |

## 6. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial release |
