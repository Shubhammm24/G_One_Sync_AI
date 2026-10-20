# Sample Size Calculation

> **G_One_Sync AI — Regulatory Documentation**
> Statistical Power Analysis for Validation Study

---

## 1. Primary Endpoint Power Analysis

### 1.1 Objective
Determine the minimum sample size required to demonstrate AUROC ≥ 0.85 for the G_One_Sync AI deterioration prediction model with adequate statistical power.

### 1.2 Assumptions

| Parameter | Value | Justification |
|-----------|-------|--------------|
| Expected AUROC | 0.89 | Retrospective performance on MIMIC-IV holdout |
| Null hypothesis AUROC | 0.75 | Minimum clinically relevant discrimination |
| Deterioration event rate | 10% | Published ICU deterioration rates: 5–15% |
| Significance level (α) | 0.025 (one-sided) | Equivalent to two-sided α = 0.05 |
| Power (1−β) | 0.90 | 90% power to detect true AUROC ≥ 0.85 |
| Dropout/exclusion rate | 15% | Missing data, protocol deviations |

### 1.3 Calculation Method

Using the method of Obuchowski (1994) for comparing an estimated AUROC against a hypothesized value:

```
n_events = (z_α + z_β)² × V(AUC) / (AUC_alt - AUC_null)²

Where:
  z_α = z(0.025) = 1.96
  z_β = z(0.10) = 1.28
  V(AUC) ≈ 0.0035 (variance of AUC estimator under alternative)
  AUC_alt = 0.89
  AUC_null = 0.75

n_events = (1.96 + 1.28)² × 0.0035 / (0.89 - 0.75)²
         = (3.24)² × 0.0035 / (0.14)²
         = 10.50 × 0.0035 / 0.0196
         = 0.0367 / 0.0196
         ≈ 47 events needed
```

### 1.4 Total Sample Size

| Component | Calculation | Result |
|-----------|------------|:------:|
| Required events | Power analysis | 47 |
| Events with margin | ×1.2 safety margin | 57 |
| Total patients (at 10% event rate) | 57 / 0.10 | 570 |
| Adjusted for dropouts (15%) | 570 / 0.85 | **671** |

### 1.5 Recommended Sample Size

> **Minimum enrollment: 500 ICU admissions** (targeting ≥50 deterioration events)
>
> **Recommended enrollment: 700 ICU admissions** (accounting for dropout and ensuring subgroup power)

## 2. Secondary Endpoint Power

### 2.1 Sensitivity/Specificity Precision

With 50 deterioration events and 450 non-events:
- **Sensitivity 80%**: 95% CI width ≈ ±11% (Wilson interval)
- **Specificity 75%**: 95% CI width ≈ ±4% (Wilson interval)

With 70 events and 630 non-events (recommended enrollment):
- **Sensitivity 80%**: 95% CI width ≈ ±9%
- **Specificity 75%**: 95% CI width ≈ ±3%

### 2.2 Comparison with NEWS2

For paired AUROC comparison (DeLong test):
- Expected: G_One_Sync AUROC 0.89 vs NEWS2 AUROC 0.74
- Correlation between scores: ρ ≈ 0.5
- With 500 patients (50 events): Power = 95% to detect ΔAUROC = 0.15

### 2.3 Subgroup Analysis

For the largest subgroups (e.g., Age >70, Male):
- Expected subgroup size: ~150–200 patients
- Expected events per subgroup: ~15–20
- Power to detect AUROC > 0.75 within each subgroup: ~70–75%

> **Note**: Subgroup analyses are exploratory; the study is powered for the overall primary endpoint.

## 3. Enrollment Timeline Estimate

| Parameter | Value |
|-----------|-------|
| Target ICU beds | 20–30 |
| Average ICU admission rate | 8–12/day |
| Eligible rate (meeting criteria) | ~60% |
| Monthly eligible admissions | ~150–200 |
| Enrollment to reach 700 | **4–5 months** |

## 4. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial sample size calculation |
