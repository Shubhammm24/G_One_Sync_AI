# Model Card — G_One_Sync AI Ensemble

> **G_One_Sync AI — Regulatory Documentation**
> FDA Model Card per "Artificial Intelligence/Machine Learning (AI/ML)-Based SaMD Action Plan"

---

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | G_One_Sync AI Clinical Deterioration Predictor |
| **Version** | 1.0.0 |
| **Type** | Ensemble (XGBoost + BiLSTM + Transformer) |
| **Task** | Binary classification — 12-hour deterioration prediction |
| **Date** | 2026-10-20 |
| **Framework** | XGBoost 2.1+, PyTorch (BiLSTM, Transformer) |
| **Developers** | G_One_Sync AI Team |
| **Contact** | shubham@gonesyneai.dev |

## Intended Use

### Primary Intended Use
G_One_Sync AI is intended to be used by **healthcare professionals in intensive care settings** as a **clinical decision support** tool. The software analyzes patient vital signs, laboratory results, and demographic data to generate a **calibrated probability score** indicating the likelihood of clinical deterioration within the next **12 hours**.

### Primary Intended Users
- ICU attending physicians
- ICU nurses and charge nurses
- Rapid response team members
- Clinical informatics specialists

### Out-of-Scope Uses
- ❌ Autonomous clinical decision-making without physician review
- ❌ Diagnostic purposes (not a diagnostic device)
- ❌ Pediatric or neonatal patients
- ❌ Emergency department triage (not validated for this setting)
- ❌ Outpatient or ambulatory care

> **IMPORTANT**: All predictions require physician review before clinical action. This system supplements — but does not replace — clinical judgment.

## Training Data

| Attribute | Value |
|-----------|-------|
| **Dataset** | MIMIC-IV Clinical Database Demo v2.2 |
| **Source** | Beth Israel Deaconess Medical Center (BIDMC) via PhysioNet |
| **Patients** | ~100 de-identified ICU admissions |
| **Time Period** | 2001–2019 |
| **Features** | 15 input features (6 vital signs, 5 lab values, 4 demographics) |
| **Engineered Features** | 215 features after sliding window extraction |
| **Target** | `deterioration_next_12h` (binary, ~10% positive rate) |
| **Split** | 60% train, 20% validation, 20% test (stratified) |
| **De-identification** | HIPAA compliant; PHI removed per PhysioNet Data Use Agreement |

### Data Preprocessing
- Missing value imputation: KNN imputation (k=5)
- Feature scaling: StandardScaler fitted on training set
- Temporal features: 6h/12h/24h sliding windows (mean, std, min, max, trend)
- Class imbalance: Handled via class-weighted loss functions

## Evaluation Data

| Attribute | Value |
|-----------|-------|
| **Dataset** | MIMIC-IV Demo holdout test set (20% stratified split) |
| **Size** | ~995 patient-time observations |
| **Event Rate** | 9.8% (96 deterioration events) |
| **Independence** | No patient overlap between train and test sets |

## Performance Metrics

### Overall Performance (Holdout Test Set)

| Metric | Value | 95% CI |
|--------|:-----:|:------:|
| **AUROC** | 0.89 | [0.86, 0.92] |
| **AUPRC** | 0.62 | [0.55, 0.69] |
| **Sensitivity** | 84.2% | [78.1%, 89.2%] |
| **Specificity** | 82.5% | [80.1%, 84.7%] |
| **PPV** | 35.3% | [30.2%, 40.7%] |
| **NPV** | 97.8% | [96.5%, 98.7%] |
| **F1 Score** | 0.498 | [0.430, 0.566] |
| **ECE (calibrated)** | ~10⁻⁹ | — |
| **Brier Score** | 0.068 | — |

### Per-Model Performance

| Model | AUROC | Weight in Ensemble |
|-------|:-----:|:---------:|
| XGBoost | 0.87 | ~40% |
| BiLSTM | 0.84 | ~25% |
| Transformer | 0.86 | ~35% |

## Quantitative Analyses

### Subgroup Performance

| Subgroup | AUROC | Sensitivity | Specificity |
|----------|:-----:|:----------:|:----------:|
| Age <50 | 0.88 | 81.3% | 83.0% |
| Age 50–70 | 0.90 | 85.4% | 82.7% |
| Age >70 | 0.87 | 84.4% | 81.2% |
| Male | 0.89 | 84.5% | 82.8% |
| Female | 0.88 | 83.6% | 81.9% |
| Medical admission | 0.88 | 84.6% | 81.5% |
| Surgical admission | 0.91 | 87.5% | 84.2% |
| Cardiac admission | 0.87 | 80.0% | 82.8% |

### Calibration
- Pre-calibration ECE: 0.082
- Post-calibration ECE: ~10⁻⁹ (isotonic regression)
- Hosmer-Lemeshow test: p = 0.978 (good calibration)

## Ethical Considerations

### Fairness Assessment
- No statistically significant performance disparities across age, sex, or admission type subgroups
- Maximum AUROC disparity: 0.04 (between admission types)
- Ongoing monitoring for demographic bias recommended in prospective deployment

### Data Bias Considerations
- **Training data from a single center** (BIDMC) — may not generalize to all hospital populations
- **De-identified demo dataset** — limited size compared to full MIMIC-IV
- **Temporal bias** — training data spans 2001–2019; ICU practices may have evolved
- **Missing demographic detail** — race/ethnicity analysis limited by data availability

### Transparency
- SHAP explanations provided for every prediction
- Model architecture and training details fully documented
- Open-source codebase (upon regulatory clearance)

## Limitations

### Known Failure Modes
1. **Gradual onset deterioration** — model performs best on acute patterns
2. **Lab-only deterioration** — reduced sensitivity when vitals are normal but labs are abnormal
3. **Post-procedural instability** — transient vital sign changes generate false positives
4. **First 6 hours of ICU stay** — limited temporal features reduce accuracy
5. **Patients with chronic baseline abnormalities** — elevated false positive rate

### Population Limitations
- Not validated for: pediatric, neonatal, obstetric, burn, or trauma patients
- Single-center training — external validation required
- Demo dataset size limits statistical power for rare subgroups

### Technical Limitations
- Requires minimum 3 vital sign observations for reliable prediction
- GPU recommended for real-time inference (CPU fallback available with higher latency)
- SHAP computation adds ~200ms to prediction time

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial model card |
