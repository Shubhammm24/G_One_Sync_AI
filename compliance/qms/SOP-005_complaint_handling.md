# SOP-005: Complaint Handling

> **G_One_Sync AI — Quality Management System**
> Standard Operating Procedure for Complaint Handling
> Per FDA 21 CFR 820.198 & ISO 13485:2016 §8.2.2

---

## 1. Purpose

Establish procedures for receiving, evaluating, investigating, and resolving complaints related to G_One_Sync AI, including determination of whether complaints constitute reportable events under FDA Medical Device Reporting (MDR) regulations.

## 2. Scope

A complaint is any written, electronic, or oral communication that alleges deficiencies related to the identity, quality, durability, reliability, safety, effectiveness, or performance of G_One_Sync AI after release.

## 3. Complaint Categories

| Category | Description | Response SLA | Examples |
|----------|-------------|-------------|----------|
| **Patient Safety** | Complaint involves actual or potential patient harm | 24 hours | False negative led to delayed intervention |
| **Performance** | System not performing to specifications | 72 hours | AUROC below validated threshold |
| **Usability** | Difficulty using the system effectively | 1 week | Dashboard confusing, alert fatigue |
| **Technical** | Software bugs not affecting patient safety | 2 weeks | UI rendering issue, slow response |
| **Documentation** | Labeling or documentation issues | 2 weeks | Missing feature documentation |

## 4. Complaint Handling Process

### 4.1 Receipt and Logging

All complaints shall be logged with:

```
Complaint ID: COMP-YYYY-NNN
Date Received: YYYY-MM-DD
Source: [Clinician / Hospital IT / Internal / Regulatory]
Reporter: [Name, Role, Institution]
Category: [Patient Safety / Performance / Usability / Technical / Documentation]

Description:
[Detailed description of the complaint]

Patient Impact:
[ ] No patient involved
[ ] Patient involved, no harm
[ ] Patient involved, potential harm
[ ] Patient involved, actual harm
[ ] Death or serious injury
```

### 4.2 MDR Determination

Within 24 hours of receipt, evaluate whether the complaint is an MDR-reportable event:

**FDA MDR Reporting Required If:**
- Device may have caused or contributed to a death → Report within **30 calendar days**
- Device may have caused or contributed to a serious injury → Report within **30 calendar days**
- Device malfunction that would likely cause death or serious injury if recurrence → Report within **30 calendar days**

**EU Vigilance Reporting Required If:**
- Serious incident as defined by EU MDR Article 87 → Report within **15 calendar days** (or 10 days for death, 2 days for serious public health threat)

### 4.3 Investigation

| Step | Activity | Timeline |
|------|----------|----------|
| 1 | Gather complaint details and patient context | Day 1 |
| 2 | Reproduce the issue in development/staging | Day 2–3 |
| 3 | Root cause analysis (5 Whys, Fishbone) | Day 3–5 |
| 4 | Determine if complaint reveals new hazard | Day 5 |
| 5 | Implement correction (if needed) | Day 5–10 |
| 6 | Update risk management file if applicable | Day 10 |
| 7 | Close complaint with documented resolution | Day 10–14 |

### 4.4 Trend Analysis

Complaints are reviewed quarterly for trends:
- Recurring failure modes → update FMEA
- Performance degradation patterns → trigger retraining evaluation
- Usability patterns → inform UI/UX improvements
- Geographic/demographic patterns → investigate subgroup performance

## 5. Records Retention

- Complaint records retained for device lifetime + 2 years (minimum 7 years)
- MDR reports retained per FDA requirements
- All records maintained in `compliance/complaints/` directory

## 6. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial release |
