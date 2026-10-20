# Study Endpoints

> **G_One_Sync AI — Regulatory Documentation**
> Primary and Secondary Endpoint Definitions

---

## 1. Primary Endpoint

### EP-001: Area Under the Receiver Operating Characteristic Curve (AUROC)

| Attribute | Specification |
|-----------|--------------|
| **Definition** | AUROC for binary classification of clinical deterioration within 12 hours |
| **Success Criterion** | AUROC ≥ 0.85 (lower bound of 95% CI > 0.80) |
| **Analysis Method** | DeLong method for 95% confidence interval |
| **Outcome Definition** | See §3 — Deterioration Event Definition |
| **Timepoint** | End of Phase 1 (silent mode, 3 months) |

## 2. Secondary Endpoints

### EP-002: Area Under the Precision-Recall Curve (AUPRC)

| Attribute | Specification |
|-----------|--------------|
| **Definition** | AUPRC for binary deterioration prediction |
| **Success Criterion** | AUPRC ≥ 0.50 |
| **Rationale** | AUPRC is the preferred metric for imbalanced classification; more informative than AUROC when prevalence is low |

### EP-003: Sensitivity (Recall)

| Attribute | Specification |
|-----------|--------------|
| **Definition** | True Positive Rate at the default operating threshold |
| **Success Criterion** | Sensitivity ≥ 80% |
| **Rationale** | Minimizing false negatives is clinically critical — missing a deterioration event is more harmful than a false alarm |

### EP-004: Specificity

| Attribute | Specification |
|-----------|--------------|
| **Definition** | True Negative Rate at the default operating threshold |
| **Success Criterion** | Specificity ≥ 75% |
| **Rationale** | Sufficient specificity to limit alert fatigue while maintaining high sensitivity |

### EP-005: Negative Predictive Value (NPV)

| Attribute | Specification |
|-----------|--------------|
| **Definition** | Probability that a LOW-risk prediction is truly stable |
| **Success Criterion** | NPV ≥ 95% |
| **Rationale** | Clinicians must be confident that a low-risk prediction reliably rules out deterioration |

### EP-006: Expected Calibration Error (ECE)

| Attribute | Specification |
|-----------|--------------|
| **Definition** | Mean absolute difference between predicted probability and observed frequency |
| **Success Criterion** | ECE < 0.05 (prospective data) |
| **Rationale** | Calibrated probabilities enable clinically meaningful risk stratification |

### EP-007: Positive Predictive Value (PPV)

| Attribute | Specification |
|-----------|--------------|
| **Definition** | Probability that a HIGH/CRITICAL-risk prediction corresponds to actual deterioration |
| **Success Criterion** | PPV ≥ 30% (given ~10% prevalence, this represents significant enrichment) |

### EP-008: Lead Time to Detection (Phase 2 only)

| Attribute | Specification |
|-----------|--------------|
| **Definition** | Time difference between G_One_Sync AI alert and clinical recognition of deterioration |
| **Success Criterion** | Median lead time ≥ 4 hours earlier than clinical recognition |
| **Measurement** | Chart review: AI alert timestamp minus clinician-initiated escalation timestamp |

### EP-009: Comparative Performance vs NEWS2

| Attribute | Specification |
|-----------|--------------|
| **Definition** | Difference in AUROC between G_One_Sync AI and NEWS2 computed on the same patient observations |
| **Success Criterion** | G_One_Sync AUROC > NEWS2 AUROC (paired DeLong test, p < 0.05) |

### EP-010: Clinical Impact (Phase 2 only)

| Attribute | Specification |
|-----------|--------------|
| **Definition** | Time from deterioration prediction to clinical intervention (labs ordered, medication started, or escalation initiated) |
| **Success Criterion** | Median time-to-intervention < 2 hours from alert |

## 3. Outcome Definitions

### 3.1 Deterioration Event (Primary Outcome)

A deterioration event is defined as ANY of the following occurring within the prediction horizon (12 hours):

| Event | Definition |
|-------|-----------|
| **ICU Transfer** | Transfer to higher level of care |
| **Rapid Response activation** | RRT or code team activated |
| **Unplanned intubation** | Endotracheal intubation not scheduled in advance |
| **Vasopressor initiation** | New start of any vasopressor agent |
| **Cardiac arrest** | Code Blue event |
| **Death** | In-ICU death |
| **Significant vital sign deterioration** | Sustained (≥30 min): HR <40 or >150, SBP <80, SpO2 <88%, RR >35 |

### 3.2 Adjudication

- All deterioration events identified by automated EMR extraction
- Borderline cases adjudicated by 2 board-certified intensivists
- Inter-rater disagreement resolved by a third reviewer

## 4. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial endpoints |
