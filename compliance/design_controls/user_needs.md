# User Needs

> **G_One_Sync AI — Design Controls**
> User Needs Documentation
> Per FDA 21 CFR 820.30(c) & IEC 62366-1:2015

---

## 1. Stakeholder Identification

| Stakeholder | Role | Needs Priority |
|-------------|------|:-:|
| **ICU Attending Physicians** | Primary clinical decision-makers | Critical |
| **ICU Nurses** | First responders to patient changes, continuous monitoring | Critical |
| **Rapid Response Teams** | Called for imminent deterioration | High |
| **Hospital IT** | System deployment, integration, maintenance | High |
| **Clinical Informatics** | EHR integration, data quality | Medium |
| **Hospital Administration** | Regulatory compliance, cost-effectiveness | Medium |
| **Patients** | Beneficiaries of improved care (indirect users) | Critical (indirect) |

## 2. User Needs

### UN-001: Early Detection of Clinical Deterioration
- **Stakeholder**: ICU Physicians, Nurses
- **Need**: "I need to know which patients are at risk of deteriorating in the next few hours, before obvious clinical signs appear"
- **Context**: Current practice relies on periodic manual assessments (NEWS2, MEWS) that may miss subtle trends between assessments
- **Priority**: Critical

### UN-002: Actionable Risk Information
- **Stakeholder**: ICU Physicians
- **Need**: "I need to understand WHY the system thinks a patient is at risk, not just a number"
- **Context**: Clinicians will not trust or act on opaque "black box" predictions. They need clinical reasoning to integrate with their own assessment
- **Priority**: Critical

### UN-003: Real-Time Alerting
- **Stakeholder**: ICU Nurses, Rapid Response Teams
- **Need**: "I need to be notified immediately when a patient's risk crosses a critical threshold"
- **Context**: ICU nurses manage multiple patients simultaneously and cannot continuously monitor each patient's dashboard
- **Priority**: Critical

### UN-004: Configurable Alert Thresholds
- **Stakeholder**: ICU Physicians, Nurses
- **Need**: "I need to adjust alert sensitivity for my specific unit — a cardiac ICU has different baselines than a medical ICU"
- **Context**: One-size-fits-all thresholds cause excessive false positives in specialized units
- **Priority**: High

### UN-005: Mobile Access
- **Stakeholder**: ICU Physicians (attending, fellows, residents)
- **Need**: "I need to check patient risk status from my phone when I'm away from the workstation"
- **Context**: Physicians rotate between multiple ICU areas and need remote access during rounds, consults, and after-hours
- **Priority**: High

### UN-006: Patient Trend Visualization
- **Stakeholder**: ICU Physicians, Nurses
- **Need**: "I need to see how the patient's risk has changed over time, not just the current score"
- **Context**: A patient at 65% risk but declining is very different from one at 65% and rising
- **Priority**: High

### UN-007: Seamless EMR Integration
- **Stakeholder**: Hospital IT, Clinical Informatics
- **Need**: "The system must pull data from our EMR automatically — no manual data entry"
- **Context**: Manual data entry introduces errors, delays, and workflow disruption
- **Priority**: High

### UN-008: Minimal Disruption to Workflow
- **Stakeholder**: ICU Nurses, Physicians
- **Need**: "The system should enhance my workflow, not add extra steps or screens to check"
- **Context**: ICU staff are already overloaded with monitoring tasks and documentation
- **Priority**: High

### UN-009: System Reliability
- **Stakeholder**: All clinical users
- **Need**: "The system must be available 24/7 — patient deterioration doesn't follow business hours"
- **Context**: Any downtime during critical hours is unacceptable for a clinical tool
- **Priority**: Critical

### UN-010: Data Privacy & Security
- **Stakeholder**: Hospital IT, Administration, Patients
- **Need**: "Patient data must be protected in compliance with HIPAA and institutional policies"
- **Context**: PHI breaches have severe regulatory, financial, and reputational consequences
- **Priority**: Critical

### UN-011: Regulatory Compliance
- **Stakeholder**: Hospital Administration, Clinical Informatics
- **Need**: "The system must be FDA-cleared or CE-marked before clinical deployment"
- **Context**: Hospitals require regulatory clearance for AI-based clinical decision support tools
- **Priority**: Critical

### UN-012: Calibrated Risk Scores
- **Stakeholder**: ICU Physicians
- **Need**: "When the system says 80% risk, I need to trust that approximately 80% of such patients actually deteriorate"
- **Context**: Poorly calibrated scores erode clinician trust and make threshold decisions unreliable
- **Priority**: High

## 3. User Needs Traceability

| User Need | Design Input(s) | Verification | Validation |
|-----------|-----------------|-------------|------------|
| UN-001 | DI-001, DI-002, DI-003 | Model AUROC ≥ 0.85 | Clinical validation study |
| UN-002 | DI-004 | SHAP values for all features | Clinician usability testing |
| UN-003 | DI-005, DI-006 | Alert delivery <30s | Workflow integration testing |
| UN-004 | DI-007 | Threshold configuration API | Unit-specific validation |
| UN-005 | DI-008 | Mobile app functional testing | Physician usability testing |
| UN-006 | DI-009 | Dashboard trend charts | Clinician feedback |
| UN-007 | DI-010, DI-011 | FHIR/HL7 integration tests | EMR connectivity validation |
| UN-008 | DI-012 | Workflow analysis | Time-motion study |
| UN-009 | DI-013 | Uptime monitoring | 99.9% availability |
| UN-010 | DI-014 | Security testing | Penetration test |
| UN-011 | DI-015 | Regulatory submission | FDA/CE clearance |
| UN-012 | DI-016 | Calibration ECE < 0.01 | Calibration analysis report |

## 4. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial user needs — 12 needs identified |
