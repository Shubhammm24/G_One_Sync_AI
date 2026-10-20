# Clinical Validation Study Protocol

> **G_One_Sync AI — Regulatory Documentation**
> Prospective Validation Study Protocol
> For FDA 510(k) / CE Marking Submission

---

## 1. Study Title

**Prospective Validation of G_One_Sync AI for Early Detection of Clinical Deterioration in Intensive Care Unit Patients**

## 2. Study Objectives

### 2.1 Primary Objective
Validate the discriminative performance of G_One_Sync AI in predicting clinical deterioration within 12 hours in a prospective ICU patient cohort.

### 2.2 Secondary Objectives
1. Evaluate prediction calibration accuracy in prospective deployment
2. Compare G_One_Sync AI performance against NEWS2 and MEWS early warning scores
3. Assess performance consistency across demographic subgroups
4. Measure time-to-clinical-intervention with and without AI-assisted alerts
5. Evaluate clinician trust and usability of AI-generated explanations

## 3. Study Design

| Attribute | Specification |
|-----------|--------------|
| **Design** | Prospective, single-center, observational |
| **Phase** | Analytical validation → Clinical validation |
| **Blinding** | Phase 1: Silent mode (predictions computed, not displayed); Phase 2: Open mode (predictions displayed to clinicians) |
| **Duration** | Phase 1: 3 months; Phase 2: 6 months |
| **Setting** | Medical/Surgical ICU, ≥20 beds |

### 3.1 Study Phases

```
Phase 1: Silent Validation (3 months)
├── System runs in background
├── Predictions computed and stored, NOT displayed
├── Compare predictions against actual outcomes
└── Confirm AUROC ≥ 0.85 before Phase 2

Phase 2: Active Validation (6 months)
├── Predictions displayed to clinicians
├── Alert system active
├── Measure clinical impact
└── Usability assessment
```

## 4. Patient Population

### 4.1 Inclusion Criteria
- Adult patients (age ≥ 18 years)
- Admitted to study ICU
- Expected ICU stay ≥ 24 hours
- At least one set of vital signs recorded in EMR

### 4.2 Exclusion Criteria
- Comfort care / hospice / DNR-comfort patients
- ICU readmission within same hospitalization (to avoid data leakage)
- Incomplete vital sign data (< 3 observations in first 6 hours)

See `inclusion_exclusion.md` for detailed criteria.

## 5. Sample Size

See `sample_size_calculation.md` for full power analysis.

**Summary**: ≥500 ICU admissions (estimated ≥50 deterioration events at 10% event rate).

## 6. Endpoints

See `endpoints.md` for detailed endpoint definitions.

### 6.1 Primary Endpoint
- **AUROC** for 12-hour deterioration prediction ≥ 0.85

### 6.2 Key Secondary Endpoints
- AUPRC ≥ 0.50
- Sensitivity ≥ 80% at default operating threshold
- Specificity ≥ 75% at default operating threshold
- NPV ≥ 95%
- Expected Calibration Error (ECE) < 0.05

## 7. Data Collection

| Data Type | Source | Frequency |
|-----------|--------|-----------|
| Vital signs | EMR → FHIR/HL7 interface | Per observation (real-time) |
| Lab results | EMR → FHIR/HL7 interface | Per result availability |
| Demographics | EMR | At admission |
| G_One_Sync predictions | System logs | Per observation |
| Deterioration outcomes | Chart review | At discharge |
| Clinical interventions | Chart review | Daily |
| Clinician feedback | Survey | Monthly |

## 8. Statistical Analysis

See `statistical_analysis_plan.md` for detailed methods.

### 8.1 Primary Analysis
- AUROC with 95% DeLong confidence interval
- Non-inferiority margin: AUROC ≥ 0.85

### 8.2 Comparison with Predicate
- AUROC comparison: G_One_Sync AI vs. NEWS2 (paired DeLong test)
- Net Reclassification Improvement (NRI)
- Integrated Discrimination Improvement (IDI)

### 8.3 Subgroup Analyses
- Age groups: <50, 50–70, >70
- Sex: Male, Female
- Admission type: Medical, Surgical, Cardiac
- ICU length of stay quartiles

## 9. Ethical Considerations

| Item | Status |
|------|--------|
| IRB/Ethics approval | Required before study initiation |
| Informed consent | Waived (observational study, standard of care not modified in Phase 1) |
| Data privacy | De-identified data only; HIPAA compliant |
| Clinician consent | Informed consent for usability survey participants |

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| System affects clinical decisions inappropriately | Phase 1 is silent; Phase 2 includes training emphasizing supplementary nature |
| Alert fatigue during Phase 2 | Configurable thresholds, monitoring of alert-to-action ratio |
| Insufficient deterioration events | Extended enrollment if needed; power maintained with ≥50 events |

## 11. Timeline

| Milestone | Target |
|-----------|--------|
| IRB submission | T+0 |
| IRB approval | T+8 weeks |
| System deployment (silent mode) | T+10 weeks |
| Phase 1 completion (silent) | T+22 weeks |
| Phase 1 analysis and go/no-go | T+24 weeks |
| Phase 2 initiation (active) | T+26 weeks |
| Phase 2 completion | T+50 weeks |
| Final analysis and report | T+54 weeks |
| Regulatory submission | T+58 weeks |

## 12. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial protocol |
