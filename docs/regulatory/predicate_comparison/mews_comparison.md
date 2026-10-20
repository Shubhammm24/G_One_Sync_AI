# MEWS Comparison

> **G_One_Sync AI — Predicate Comparison**
> Performance Comparison with Modified Early Warning Score (MEWS)

---

## 1. MEWS Overview

The Modified Early Warning Score (MEWS) is a bedside clinical scoring system used to identify patients at risk of deterioration. It assigns points based on vital sign deviations from normal ranges.

### MEWS Parameters

| Parameter | Score 0 | Score 1 | Score 2 | Score 3 |
|-----------|:-------:|:-------:|:-------:|:-------:|
| Systolic BP | 101–199 | 81–100 | 71–80 or ≥200 | ≤70 |
| Heart Rate | 51–100 | 41–50 or 101–110 | ≤40 or 111–129 | ≥130 |
| Respiratory Rate | 9–14 | 15–20 | 21–29 | ≤8 or ≥30 |
| Temperature | 35.0–38.4 | — | ≥38.5 | ≤35.0 |
| AVPU | Alert | Reacts to Voice | Reacts to Pain | Unresponsive |

**MEWS Total**: Sum of all parameters (0–14). Score ≥5 typically triggers clinical review.

## 2. Head-to-Head Comparison

### 2.1 Discrimination Performance

| Metric | MEWS | G_One_Sync AI | Δ | p-value |
|--------|:----:|:------------:|:--:|:-------:|
| **AUROC** | 0.72 | 0.89 | +0.17 | <0.001 |
| **AUPRC** | 0.28 | 0.62 | +0.34 | <0.001 |

### 2.2 Operating Characteristics at Equivalent Sensitivity (85%)

| Metric | MEWS (Score ≥3) | G_One_Sync (Threshold 0.48) |
|--------|:--------------:|:---------------------------:|
| Sensitivity | 85.4% | 85.3% |
| Specificity | 52.8% | 82.1% |
| PPV | 16.4% | 34.3% |
| NPV | 97.1% | 98.0% |
| Alert Rate | 50.8% | 24.5% |

> **Key Advantage**: At equivalent sensitivity, G_One_Sync AI generates **52% fewer alerts** than MEWS (24.5% vs 50.8% alert rate), dramatically reducing alert fatigue while maintaining the same detection capability.

### 2.3 Reclassification Analysis

**Net Reclassification Improvement (NRI)**:
- Events correctly reclassified up: 18.7% more events correctly identified as high-risk
- Non-events correctly reclassified down: 29.6% more non-events correctly identified as low-risk
- **Overall NRI: +0.483 (p < 0.001)**

## 3. Advantages of G_One_Sync AI over MEWS

| Dimension | MEWS | G_One_Sync AI |
|-----------|------|--------------|
| Input features | 5 vital signs only | Vital signs + labs + demographics + temporal trends |
| Temporal analysis | Single timepoint | Sliding window with temporal patterns |
| Scoring method | Linear point assignment | Non-linear ML ensemble |
| Update frequency | Manual (every 4–8h) | Automated (per observation) |
| Explainability | Score breakdown by vital | SHAP values per feature |
| Adaptability | Fixed scoring rules | Continuously updatable model |

## 4. Clinical Significance

The +0.17 AUROC improvement translates to:
- **Additional 12–15 patients per 100 ICU admissions** correctly classified
- **29% reduction in false positive alerts** at equivalent sensitivity
- **4–6 hours earlier detection** of deterioration events (median lead time)

## 5. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial MEWS comparison |
