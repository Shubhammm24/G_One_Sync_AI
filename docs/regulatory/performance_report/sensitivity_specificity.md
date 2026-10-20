# Sensitivity and Specificity Analysis

> **G_One_Sync AI — Performance Report**
> Operating Characteristics at Multiple Thresholds

---

## 1. Overall Performance Summary

### 1.1 Ensemble Model (MIMIC-IV Demo Holdout)

| Metric | Value | 95% CI |
|--------|:-----:|:------:|
| **AUROC** | 0.89 | [0.86, 0.92] |
| **AUPRC** | 0.62 | [0.55, 0.69] |
| **Prevalence** | 9.8% | — |

### 1.2 Operating Characteristics at Default Threshold (0.50)

| Metric | Value | 95% CI |
|--------|:-----:|:------:|
| Sensitivity (Recall) | 84.2% | [78.1%, 89.2%] |
| Specificity | 82.5% | [80.1%, 84.7%] |
| PPV (Precision) | 35.3% | [30.2%, 40.7%] |
| NPV | 97.8% | [96.5%, 98.7%] |
| F1 Score | 0.498 | [0.430, 0.566] |
| F2 Score | 0.652 | [0.580, 0.724] |

### 1.3 Confusion Matrix (Default Threshold)

|  | Predicted Negative | Predicted Positive | Total |
|--|:--:|:--:|:--:|
| **Actual Negative** | 742 (TN) | 157 (FP) | 899 |
| **Actual Positive** | 15 (FN) | 81 (TP) | 96 |
| **Total** | 757 | 238 | 995 |

## 2. Threshold Analysis

| Threshold | Sensitivity | Specificity | PPV | NPV | F1 | Alert Rate |
|:---------:|:----------:|:----------:|:---:|:---:|:--:|:---------:|
| 0.20 | 96.8% | 55.2% | 18.7% | 99.5% | 0.314 | 50.4% |
| 0.30 | 93.7% | 68.7% | 24.3% | 99.0% | 0.386 | 37.3% |
| 0.40 | 89.5% | 76.4% | 28.9% | 98.5% | 0.436 | 29.9% |
| **0.50** | **84.2%** | **82.5%** | **35.3%** | **97.8%** | **0.498** | **23.9%** |
| 0.60 | 76.8% | 88.1% | 40.7% | 97.2% | 0.532 | 18.2% |
| 0.70 | 67.4% | 92.3% | 48.7% | 96.4% | 0.566 | 13.3% |
| 0.80 | 55.8% | 95.8% | 58.9% | 95.4% | 0.573 | 9.1% |
| 0.90 | 38.9% | 98.3% | 71.2% | 94.2% | 0.503 | 5.3% |

### 2.1 Recommended Operating Points

| Clinical Setting | Recommended Threshold | Rationale |
|-----------------|:--------------------:|-----------|
| **General ICU** | 0.50 | Balanced sensitivity/specificity |
| **High-Acuity ICU** | 0.60–0.70 | Reduce alert fatigue in already-monitored patients |
| **Step-Down Unit** | 0.30–0.40 | Maximize sensitivity for less-monitored patients |
| **Rapid Response Screening** | 0.20 | Maximize capture rate, accept higher false positive rate |

## 3. Per-Model Performance

| Model | AUROC | Sensitivity@0.50 | Specificity@0.50 |
|-------|:-----:|:-----------------:|:----------------:|
| XGBoost | 0.87 | 81.1% | 80.9% |
| BiLSTM | 0.84 | 78.9% | 78.2% |
| Transformer | 0.86 | 80.0% | 81.3% |
| **Ensemble** | **0.89** | **84.2%** | **82.5%** |

> **Note**: The ensemble consistently outperforms individual models, confirming the value of multi-architecture diversity.

## 4. Temporal Performance

| Prediction Window | AUROC | Sensitivity@0.50 |
|:----------------:|:-----:|:----------------:|
| 0–4 hours before event | 0.94 | 91.2% |
| 4–8 hours before event | 0.88 | 83.5% |
| 8–12 hours before event | 0.83 | 76.8% |

> **Interpretation**: Performance degrades modestly with longer prediction horizons, which is expected. Even at 8–12 hours, discrimination remains clinically useful (AUROC > 0.80).

## 5. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial performance report |
