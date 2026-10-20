# Intended Use Statement

> **G_One_Sync AI — Regulatory Documentation**
> Intended Use and Indications for Use
> For FDA 510(k) Submission & EU MDR Technical Documentation

---

## 1. Intended Use

G_One_Sync AI is a **Software as a Medical Device (SaMD)** intended to be used by qualified healthcare professionals in **intensive care unit (ICU) settings** as a **clinical decision support tool** for the **early detection of patient clinical deterioration**.

The software processes patient vital signs (heart rate, respiratory rate, SpO2, temperature, blood pressure), laboratory results (lactate, WBC, creatinine, BUN, hemoglobin), and demographic information to generate:

1. A **calibrated probability score** (0–100%) indicating the likelihood of clinical deterioration within the next **12 hours**
2. **Explainable risk factors** identifying the top contributing features to the risk assessment
3. **Configurable clinical alerts** when patient risk exceeds defined thresholds

## 2. Indications for Use

G_One_Sync AI is indicated for use by:
- **ICU attending physicians** — for patient risk prioritization and proactive care planning
- **ICU nurses** — for continuous patient monitoring support and early alert notification
- **Rapid Response Teams** — for targeted patient assessment based on risk stratification
- **Clinical informatics specialists** — for quality improvement and outcomes monitoring

In the following settings:
- **Medical Intensive Care Units (MICU)**
- **Surgical Intensive Care Units (SICU)**
- **Cardiac Intensive Care Units (CCU)**
- **Mixed Medical-Surgical ICUs**

## 3. Patient Population

| Criterion | Specification |
|-----------|--------------|
| **Age** | Adult patients (≥ 18 years) |
| **Setting** | ICU admission with expected stay ≥ 24 hours |
| **Data requirement** | Minimum 3 vital sign observations |
| **Exclusions** | Comfort care/hospice, pediatric, neonatal |

## 4. Contraindications

G_One_Sync AI is **NOT intended** for:

1. **Autonomous clinical decision-making** — All predictions require physician review before any clinical action is taken
2. **Diagnostic purposes** — The system provides risk estimates, not diagnoses
3. **Primary patient monitoring** — The system supplements, but does not replace, standard physiological monitoring equipment
4. **Pediatric or neonatal patients** — Not validated for these populations
5. **Emergency department triage** — Not validated for pre-ICU settings
6. **Outpatient or ambulatory care** — Not validated for non-ICU settings

## 5. Essential Safety Statement

> **⚠️ CAUTION: Clinical Decision Support Only**
>
> G_One_Sync AI is a clinical decision support tool. It is intended to supplement — not replace — the clinical judgment of qualified healthcare professionals.
>
> **All predictions must be reviewed by a licensed physician before any clinical action is taken.**
>
> The system may produce false positive or false negative results. Clinical decisions should never be based solely on the output of this software. Clinicians must consider the complete clinical picture, including physical examination, patient history, and other diagnostic information.

## 6. Performance Claims

Based on retrospective validation (MIMIC-IV):

| Claim | Evidence |
|-------|---------|
| AUROC ≥ 0.85 for 12-hour deterioration prediction | Holdout test set AUROC = 0.89 [0.86, 0.92] |
| Sensitivity ≥ 80% at default threshold | Sensitivity = 84.2% [78.1%, 89.2%] |
| NPV ≥ 95% for low-risk predictions | NPV = 97.8% [96.5%, 98.7%] |
| Well-calibrated predicted probabilities | ECE < 0.001 (post-isotonic calibration) |
| Superior to NEWS2 early warning score | ΔAUROC = +0.13 (p < 0.001) |

*Note: Performance claims based on retrospective validation. Prospective validation study required for FDA clearance.*

## 7. Regulatory Classification

| Jurisdiction | Classification | Pathway |
|-------------|---------------|---------|
| **US (FDA)** | Class II SaMD | 510(k) or De Novo |
| **EU (MDR)** | Class IIa Medical Device | Notified Body Conformity Assessment |
| **IEC 62304** | Class B Software | Hazardous situation possible, external risk controls exist |
| **ISO 14971** | Medium Risk | Residual risks acceptable with controls |

## 8. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial intended use statement |
