# Statistical Analysis Plan (SAP)

> **G_One_Sync AI — Regulatory Documentation**
> Statistical Analysis Plan for Clinical Validation Study

---

## 1. General Principles

### 1.1 Analysis Populations

| Population | Definition | Use |
|-----------|-----------|-----|
| **Full Analysis Set (FAS)** | All enrolled patients meeting inclusion criteria | Primary analysis |
| **Per-Protocol Set (PPS)** | FAS excluding protocol deviations (e.g., <6h of data) | Sensitivity analysis |
| **Safety Set** | All patients exposed to predictions (Phase 2) | Safety analysis |

### 1.2 Statistical Software
- Python 3.10+ with `scipy`, `scikit-learn`, `statsmodels`
- Significance level: α = 0.05 (two-sided) unless otherwise specified
- All confidence intervals: 95%
- No adjustment for multiple comparisons in secondary endpoints (exploratory)

## 2. Primary Analysis

### 2.1 AUROC Estimation (EP-001)

**Method**: Non-parametric AUROC estimation with DeLong standard error

```python
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

# Point estimate
auroc = roc_auc_score(y_true, y_pred_proba)

# DeLong 95% CI
se_auroc = delong_se(y_true, y_pred_proba)
ci_lower = auroc - 1.96 * se_auroc
ci_upper = auroc + 1.96 * se_auroc
```

**Success Criterion**: Lower bound of 95% CI > 0.80 AND point estimate ≥ 0.85

**Hypothesis Test**:
- H₀: AUROC ≤ 0.75 (no better than standard EWS)
- H₁: AUROC > 0.75
- One-sided test at α = 0.025

## 3. Secondary Analyses

### 3.1 Operating Characteristics (EP-003, EP-004, EP-005, EP-007)

At the default operating threshold:

| Metric | Formula | 95% CI Method |
|--------|---------|---------------|
| Sensitivity | TP / (TP + FN) | Wilson interval |
| Specificity | TN / (TN + FP) | Wilson interval |
| PPV | TP / (TP + FP) | Wilson interval |
| NPV | TN / (TN + FN) | Wilson interval |

Report confusion matrix with exact counts.

### 3.2 Calibration Analysis (EP-006)

**Expected Calibration Error (ECE)**:
```
ECE = Σ (|Bm| / N) × |accuracy(Bm) - confidence(Bm)|
```
where Bm is the m-th bin, using M = 10 equal-width bins over [0, 1].

**Calibration Plot**: Observed event rate vs. predicted probability with LOESS smoother and 95% pointwise CI.

**Hosmer-Lemeshow Test**: Goodness-of-fit test for calibration (H₀: model is well-calibrated).

### 3.3 AUPRC Analysis (EP-002)

**Method**: Trapezoidal approximation of precision-recall curve.

**Baseline**: Random classifier AUPRC = prevalence (~0.10).

**Bootstrapped 95% CI**: 2000 bootstrap resamples.

### 3.4 Comparison with NEWS2 (EP-009)

**Paired DeLong Test**:
- Compute NEWS2 score for each patient-time observation
- Compute G_One_Sync prediction for the same observation
- DeLong test for correlated AUROCs

**Net Reclassification Improvement (NRI)**:
```
NRI = P(up | event) - P(up | non-event) + P(down | non-event) - P(down | event)
```

**Integrated Discrimination Improvement (IDI)**:
```
IDI = (mean_pred_events_new - mean_pred_events_old) - (mean_pred_nonevents_new - mean_pred_nonevents_old)
```

### 3.5 Lead Time Analysis (EP-008)

**Method**: Kaplan-Meier analysis of time from AI alert to clinical recognition.

**Paired analysis**: For each deterioration event, compute:
- T_AI = time of first HIGH/CRITICAL alert
- T_clinical = time of clinician-initiated escalation
- Lead time = T_clinical - T_AI (positive = AI was earlier)

**Report**: Median lead time with IQR.

## 4. Subgroup Analyses

### 4.1 Pre-Specified Subgroups

| Subgroup | Categories | Minimum N per Group |
|----------|-----------|:-------------------:|
| Age | <50, 50–70, >70 | 100 |
| Sex | Male, Female | 200 |
| Admission Type | Medical, Surgical, Cardiac | 100 |
| ICU LOS quartile | Q1–Q4 | 125 |

### 4.2 Subgroup Analysis Method

For each subgroup:
1. AUROC with 95% CI
2. Sensitivity and specificity at global threshold
3. Interaction test: logistic regression with subgroup × prediction interaction term

**Fairness criterion**: No subgroup with AUROC < 0.80 (point estimate).

## 5. Sensitivity Analyses

| Analysis | Purpose |
|----------|---------|
| Per-protocol vs. FAS | Impact of protocol deviations |
| Varying prediction threshold (0.2–0.9) | Full ROC curve characterization |
| Excluding first 6 hours of ICU stay | Cold-start period effect |
| Multiple imputation for missing labs | Missing data impact |
| Time-stratified analysis (day vs. night) | Temporal performance variation |

## 6. Interim Analysis

An interim analysis will be conducted at 50% enrollment (N ≈ 250):
- **Purpose**: Futility assessment only (not efficacy stopping)
- **Futility criterion**: If AUROC < 0.78 at interim, study may be paused for model recalibration
- **Alpha spending**: Not applicable (futility only)

## 7. Missing Data

| Data Type | Handling |
|-----------|---------|
| Missing vital signs | KNN imputation (existing pipeline) — note proportion in report |
| Missing lab results | Carried forward from last available (12-hour window) |
| Missing outcome data | Excluded from FAS, documented in CONSORT diagram |
| Missing demographics | Complete case analysis; sensitivity analysis with imputation |

## 8. Reporting

Results reported per:
- **STARD 2015** (Standards for Reporting Diagnostic Accuracy Studies)
- **TRIPOD** (Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis)

## 9. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial SAP |
