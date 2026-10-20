# Inclusion and Exclusion Criteria

> **G_One_Sync AI — Regulatory Documentation**
> Patient Selection Criteria for Clinical Validation Study

---

## 1. Inclusion Criteria

| ID | Criterion | Rationale |
|:--:|-----------|-----------|
| IC-01 | Age ≥ 18 years at ICU admission | System validated for adult patients only |
| IC-02 | Admitted to participating ICU (Medical, Surgical, or Cardiac) | Study enrollment unit |
| IC-03 | Expected ICU length of stay ≥ 24 hours | Sufficient observation window for 12-hour prediction horizon |
| IC-04 | At least 1 complete set of vital signs recorded within 6 hours of ICU admission | Minimum data required for initial prediction |
| IC-05 | Heart rate, respiratory rate, SpO2, temperature, and blood pressure all available | Core vital signs required for feature extraction |

## 2. Exclusion Criteria

| ID | Criterion | Rationale |
|:--:|-----------|-----------|
| EC-01 | Comfort care, hospice, or comfort-measures-only (CMO) status | Deterioration prediction not clinically appropriate for end-of-life care |
| EC-02 | DNR/DNI with limitation of escalation | Clinical actions based on prediction would conflict with care goals |
| EC-03 | ICU readmission within the same hospitalization | Data leakage risk — prior ICU stay data may bias predictions |
| EC-04 | Brain death evaluation or organ donation candidate | Clinical trajectory not relevant to deterioration prediction |
| EC-05 | Age < 18 years | System not validated for pediatric physiology |
| EC-06 | Active participation in another interventional clinical trial that modifies ICU monitoring protocols | Potential confounding of study endpoints |
| EC-07 | Fewer than 3 vital sign observations in first 6 hours of ICU admission | Insufficient temporal data for sliding window features |
| EC-08 | Transfer from external facility with no available EMR history | Missing baseline data for feature computation |

## 3. Screening Log

All patients admitted to the study ICU during the enrollment period will be screened. A screening log shall document:

```
Screening ID: SCR-YYYY-NNNN
Date: YYYY-MM-DD
ICU Admission Time: HH:MM
Age: [years]
Sex: [M/F]

Eligibility:
├── IC-01 (Age ≥18): [Y/N]
├── IC-02 (Study ICU): [Y/N]
├── IC-03 (LOS ≥24h): [Y/N/TBD]
├── IC-04 (Vitals in 6h): [Y/N]
├── IC-05 (Core vitals): [Y/N]
├── EC-01 (Comfort care): [Y/N]
├── EC-02 (DNR/DNI): [Y/N]
├── EC-03 (Readmission): [Y/N]
├── EC-04 (Brain death): [Y/N]
├── EC-05 (Pediatric): [Y/N]
├── EC-06 (Other trial): [Y/N]
├── EC-07 (<3 observations): [Y/N]
└── EC-08 (External transfer): [Y/N]

Decision: [ENROLLED / EXCLUDED (reason)]
```

## 4. Expected Enrollment

| Parameter | Estimate |
|-----------|:--------:|
| Total ICU admissions (study period) | ~1,500 |
| Expected exclusion rate | ~35% |
| Primary exclusion reasons | Comfort care (10%), <24h stay (12%), data gaps (8%), readmissions (5%) |
| Expected enrolled | **~975** |
| Minimum required | **500** |

## 5. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial criteria |
