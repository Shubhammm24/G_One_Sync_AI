# SOP-006: Corrective and Preventive Actions (CAPA)

> **G_One_Sync AI — Quality Management System**
> Standard Operating Procedure for Corrective and Preventive Actions
> Per FDA 21 CFR 820.90 & ISO 13485:2016 §8.5.2/§8.5.3

---

## 1. Purpose

Establish procedures for identifying, investigating, implementing, and verifying corrective and preventive actions to eliminate the causes of existing nonconformities and prevent their recurrence or occurrence.

## 2. Definitions

| Term | Definition |
|------|-----------|
| **Correction** | Action to eliminate a detected nonconformity (immediate fix) |
| **Corrective Action** | Action to eliminate the cause of a detected nonconformity (prevent recurrence) |
| **Preventive Action** | Action to eliminate the cause of a potential nonconformity (prevent occurrence) |
| **Nonconformity** | Non-fulfillment of a requirement |

## 3. CAPA Sources

CAPAs may be initiated from:
- Complaint investigations (SOP-005)
- Internal/external audit findings
- Model performance degradation (drift detection alerts)
- Failed verification or validation tests
- Post-market surveillance data
- Regulatory feedback
- Management review findings

## 4. CAPA Process

### 4.1 Identification and Documentation

```
CAPA ID: CAPA-YYYY-NNN
Date Opened: YYYY-MM-DD
Type: [ ] Corrective  [ ] Preventive
Source: [Complaint / Audit / Monitoring / Test Failure / Other]
Priority: [Critical / High / Medium / Low]

Problem Statement:
[Clear, factual description of the nonconformity or potential nonconformity]

Immediate Correction (if applicable):
[Actions taken to address the immediate problem]
```

### 4.2 Root Cause Analysis

| Method | When to Use |
|--------|------------|
| **5 Whys** | Simple cause-effect chains |
| **Fishbone (Ishikawa)** | Multiple potential contributing factors |
| **Fault Tree Analysis** | Complex system failures with multiple paths |
| **FMEA Update** | When new failure modes are discovered |

Root cause categories for AI/ML systems:

| Category | Examples |
|----------|---------|
| **Data Quality** | Training data bias, missing demographics, label noise |
| **Model Design** | Underfitting, overfitting, inappropriate architecture |
| **Feature Engineering** | Missing temporal features, incorrect normalization |
| **Infrastructure** | API timeout, memory exhaustion, GPU failure |
| **Integration** | EMR data format change, Kafka schema evolution |
| **Human Factors** | Alert fatigue, misinterpretation, inadequate training |
| **Process** | Inadequate code review, missing test coverage |

### 4.3 Action Plan

The CAPA action plan shall include:

1. **Specific actions** to address root cause(s)
2. **Owner** for each action item
3. **Timeline** with milestones
4. **Success criteria** — measurable evidence that the CAPA is effective
5. **Verification plan** — how effectiveness will be confirmed

### 4.4 Implementation

- All code changes follow SOP-004 (Change Management)
- Model changes require re-verification per SOP-003
- Risk management file updated per SOP-002
- Changes documented in Git with conventional commit referencing CAPA ID:
  ```
  fix(modeling): adjust class weights to reduce FNR [CAPA-2026-003]
  ```

### 4.5 Effectiveness Verification

CAPA is not closed until effectiveness is verified:

| Verification Method | Criteria |
|--------------------|----------|
| Statistical analysis | Problem recurrence rate decreased by ≥90% |
| Model metrics | AUROC/AUPRC maintained or improved |
| Test results | New regression tests pass |
| Monitoring | No recurrence in 30-day monitoring window |
| Audit | Follow-up audit confirms compliance |

### 4.6 Closure

CAPA closure requires:
- [ ] Root cause identified and documented
- [ ] Corrective/preventive actions implemented
- [ ] Effectiveness verified with objective evidence
- [ ] Risk management file updated (if applicable)
- [ ] Related SOPs updated (if applicable)
- [ ] Approval by QA/RA Lead

## 5. CAPA Metrics

Tracked quarterly in management review:
- Total CAPAs opened/closed
- Average time to closure by priority
- Recurrence rate (effectiveness measure)
- CAPAs by source and root cause category
- Overdue CAPAs

## 6. Records

CAPA records maintained in `compliance/capa/` with:
- CAPA form (problem statement, root cause, action plan)
- Investigation evidence
- Effectiveness verification data
- Closure documentation

Retention: Device lifetime + 2 years (minimum 7 years)

## 7. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial release |
