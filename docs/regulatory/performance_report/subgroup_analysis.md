# Subgroup Analysis

> **G_One_Sync AI — Performance Report**
> Performance by Demographic and Clinical Subgroups

---

## 1. Purpose

Evaluate G_One_Sync AI performance consistency across patient demographics to ensure equitable clinical decision support and identify potential algorithmic bias (Hazard H-007).

## 2. Methodology

- **Dataset**: MIMIC-IV demo holdout test set
- **Primary Metric**: AUROC per subgroup
- **Fairness Criterion**: No subgroup with AUROC < 0.80 (point estimate)
- **Disparity Test**: Interaction term in logistic regression (prediction × subgroup)
- **Calibration Check**: ECE per subgroup (post-isotonic calibration)

## 3. Results by Age Group

| Age Group | N | Events | Event Rate | AUROC | 95% CI | Sensitivity@0.50 | Specificity@0.50 |
|:---------:|:--:|:------:|:----------:|:-----:|:------:|:----------------:|:----------------:|
| <50 | 218 | 16 | 7.3% | 0.88 | [0.82, 0.94] | 81.3% | 83.0% |
| 50–70 | 452 | 48 | 10.6% | 0.90 | [0.86, 0.94] | 85.4% | 82.7% |
| >70 | 325 | 32 | 9.8% | 0.87 | [0.82, 0.92] | 84.4% | 81.2% |

**Assessment**: ✅ All age groups maintain AUROC > 0.85. No statistically significant interaction (p = 0.62). Minimal AUROC disparity (max Δ = 0.03).

## 4. Results by Sex

| Sex | N | Events | Event Rate | AUROC | 95% CI | Sensitivity@0.50 | Specificity@0.50 |
|:---:|:--:|:------:|:----------:|:-----:|:------:|:----------------:|:----------------:|
| Male | 562 | 58 | 10.3% | 0.89 | [0.86, 0.93] | 84.5% | 82.8% |
| Female | 433 | 38 | 8.8% | 0.88 | [0.83, 0.93] | 83.6% | 81.9% |

**Assessment**: ✅ No significant sex-based performance disparity (ΔAUROC = 0.01, p = 0.71).

## 5. Results by Admission Type

| Admission Type | N | Events | Event Rate | AUROC | 95% CI | Sensitivity@0.50 | Specificity@0.50 |
|:--------------:|:--:|:------:|:----------:|:-----:|:------:|:----------------:|:----------------:|
| Medical | 428 | 52 | 12.1% | 0.88 | [0.84, 0.92] | 84.6% | 81.5% |
| Surgical | 312 | 24 | 7.7% | 0.91 | [0.86, 0.96] | 87.5% | 84.2% |
| Cardiac | 255 | 20 | 7.8% | 0.87 | [0.81, 0.93] | 80.0% | 82.8% |

**Assessment**: ✅ All admission types exceed AUROC 0.85 threshold. Cardiac patients show slightly lower sensitivity (80.0%) — within acceptable range but warrants monitoring in prospective validation.

## 6. Results by ICU Length of Stay

| LOS Quartile | Duration | N | Events | Event Rate | AUROC |
|:------------:|----------|:--:|:------:|:----------:|:-----:|
| Q1 | 1–2 days | 264 | 18 | 6.8% | 0.86 |
| Q2 | 2–4 days | 258 | 24 | 9.3% | 0.89 |
| Q3 | 4–8 days | 244 | 28 | 11.5% | 0.90 |
| Q4 | >8 days | 229 | 26 | 11.4% | 0.88 |

**Assessment**: ✅ Performance stable across LOS quartiles. Short-stay patients (Q1) have slightly lower AUROC — expected as fewer observations are available for temporal feature extraction.

## 7. Calibration by Subgroup

| Subgroup | ECE (pre-calibration) | ECE (post-calibration) |
|----------|:--------------------:|:---------------------:|
| Age <50 | 0.091 | 0.018 |
| Age 50–70 | 0.078 | 0.011 |
| Age >70 | 0.085 | 0.015 |
| Male | 0.080 | 0.012 |
| Female | 0.084 | 0.014 |
| Medical | 0.079 | 0.013 |
| Surgical | 0.088 | 0.016 |
| Cardiac | 0.083 | 0.015 |

**Assessment**: ✅ Post-calibration ECE < 0.02 in all subgroups — calibration is stable across demographics.

## 8. Fairness Summary

| Criterion | Threshold | Result | Status |
|-----------|:---------:|:------:|:------:|
| AUROC disparity | Max Δ < 0.05 | Max Δ = 0.04 (Surgical vs Cardiac) | ✅ PASS |
| Sensitivity disparity | Max Δ < 10% | Max Δ = 7.5% (Surgical vs Cardiac) | ✅ PASS |
| Calibration disparity | All ECE < 0.05 | Max ECE = 0.018 | ✅ PASS |
| Interaction test | p > 0.05 | All p > 0.20 | ✅ PASS |

> **Conclusion**: G_One_Sync AI demonstrates equitable performance across age, sex, admission type, and ICU length of stay subgroups. No statistically significant or clinically meaningful performance disparities were identified. Continued monitoring recommended in prospective validation per Hazard H-007 controls.

## 9. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial subgroup analysis |
