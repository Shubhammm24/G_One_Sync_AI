# Design Validation

> **G_One_Sync AI — Design Controls**
> Design Validation Plan & Report
> Per FDA 21 CFR 820.30(g) & IEC 62366-1:2015

---

## 1. Purpose

Design validation confirms that G_One_Sync AI meets **user needs and intended uses** under actual or simulated use conditions. Validation answers: "Did we build the **right** system?"

## 2. Validation vs. Verification

| Aspect | Verification (820.30(f)) | Validation (820.30(g)) |
|--------|:-----------------------:|:---------------------:|
| Question | Does output meet input specs? | Does device meet user needs? |
| Method | Testing, inspection, analysis | Simulated/actual use studies |
| Performers | Engineering team | Clinical end-users |
| Environment | Development/test | Simulated ICU or actual ICU |

## 3. Validation Activities

### V-001: Clinical Validation Study (Primary)

| Attribute | Detail |
|-----------|--------|
| **User Need** | UN-001 (Early Deterioration Detection) |
| **Study Design** | Prospective observational validation |
| **Full Protocol** | `docs/regulatory/clinical_validation/study_protocol.md` |
| **Primary Endpoint** | AUROC ≥ 0.85 on prospective ICU data |
| **Secondary Endpoints** | Sensitivity ≥ 80%, NPV ≥ 95%, time-to-intervention improvement |
| **Sample Size** | ≥500 ICU admissions (see sample size calculation) |
| **Duration** | 6–12 months |
| **Status** | 📋 Planned |

### V-002: Usability Validation — Dashboard

| Attribute | Detail |
|-----------|--------|
| **User Need** | UN-002, UN-006, UN-008 |
| **Method** | Task-based usability testing with think-aloud protocol |
| **Participants** | 10–15 ICU physicians and nurses (IEC 62366-1 minimum) |
| **Tasks** | (1) Identify highest-risk patient, (2) Interpret SHAP explanation, (3) Acknowledge alert, (4) Review patient trend |
| **Success Criteria** | ≥90% task completion rate, SUS score ≥ 70, mean task time < 30s per task |
| **Status** | 📋 Planned |

### V-003: Usability Validation — Mobile App

| Attribute | Detail |
|-----------|--------|
| **User Need** | UN-005 |
| **Method** | Contextual inquiry during simulated on-call scenario |
| **Participants** | 8–12 ICU physicians (attendings, fellows, residents) |
| **Tasks** | (1) Login via biometric, (2) Check watchlist, (3) View patient detail, (4) Respond to push notification |
| **Success Criteria** | ≥85% task completion rate, response time to critical alert < 60s |
| **Status** | 📋 Planned |

### V-004: Clinical Workflow Integration

| Attribute | Detail |
|-----------|--------|
| **User Need** | UN-008, UN-003 |
| **Method** | Simulated ICU shift with G_One_Sync AI integrated into workflow |
| **Participants** | 2 attending physicians, 4 nurses, 1 rapid response team |
| **Scenario** | 8-hour simulated shift with 12 patients (3 deterioration events embedded) |
| **Success Criteria** | All 3 deterioration events detected before conventional methods, no workflow disruption reported |
| **Status** | 📋 Planned |

### V-005: Subgroup Performance Validation

| Attribute | Detail |
|-----------|--------|
| **User Need** | UN-001 (health equity) |
| **Hazard** | H-007 (Algorithmic Bias) |
| **Method** | Stratified performance analysis on validation dataset |
| **Subgroups** | Age (<50, 50-70, >70), Sex (M/F), Admission type (Medical, Surgical, Cardiac), Ethnicity |
| **Success Criteria** | AUROC within ±0.05 across all subgroups; no statistically significant performance disparity |
| **Status** | 📋 Planned |

### V-006: EMR Integration Validation

| Attribute | Detail |
|-----------|--------|
| **User Need** | UN-007 |
| **Method** | End-to-end testing with EMR sandbox (Epic/Cerner test environment) |
| **Tests** | (1) FHIR Observation → Prediction, (2) HL7 ORU^R01 → Prediction, (3) Data reconciliation audit |
| **Success Criteria** | 100% of test observations correctly mapped and processed |
| **Status** | 📋 Planned |

## 4. Validation Traceability

| User Need | Validation Activity | Acceptance Criteria | Status |
|:---------:|:-------------------:|--------------------:|:------:|
| UN-001 | V-001 | AUROC ≥ 0.85 prospective | 📋 |
| UN-002 | V-002 | 90% task completion (SHAP) | 📋 |
| UN-003 | V-002, V-004 | Alert response <60s | 📋 |
| UN-004 | V-004 | Unit-specific thresholds work | 📋 |
| UN-005 | V-003 | 85% mobile task completion | 📋 |
| UN-006 | V-002 | Trend chart usable | 📋 |
| UN-007 | V-006 | 100% EMR mapping | 📋 |
| UN-008 | V-004 | No workflow disruption | 📋 |
| UN-009 | V-001 (passive) | 99.9% uptime during study | 📋 |
| UN-010 | V-006 | No PHI exposure | 📋 |
| UN-011 | FDA/CE submission | Regulatory clearance | 📋 |
| UN-012 | V-001, V-005 | Calibration verified prospectively | 📋 |

## 5. Validation Completion Criteria

Design validation is complete when:
- [ ] Clinical validation study primary endpoint met (V-001)
- [ ] Dashboard usability testing passed (V-002)
- [ ] Mobile app usability testing passed (V-003)
- [ ] Clinical workflow integration successful (V-004)
- [ ] No statistically significant subgroup disparities (V-005)
- [ ] EMR integration validated in sandbox (V-006)
- [ ] All user needs have at least one validation activity with passing results

## 6. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial validation plan |
