# Hazard Log

> **G_One_Sync AI — Risk Management File**
> Active Hazard Tracking Register
> Per ISO 14971:2019 §7.4

---

## Purpose

This hazard log is a living document that tracks all identified hazards, their current risk status, associated controls, and verification evidence. It is updated whenever new hazards are identified or existing controls are modified.

## Active Hazards

| ID | Hazard Description | Severity | Probability | Risk Level | Control Measures | Control Verified | Residual Risk | Owner | Last Reviewed |
|:--:|-------------------|:--------:|:----------:|:----------:|-----------------|:----------------:|:-------------:|-------|:------------:|
| H-001 | False negative — failure to predict deterioration | S4–S5 | P3→P2 | 🟡 ALARP | Ensemble modeling, class-weighted training, confidence flagging, external monitoring | ✅ | 🟡 ALARP | ML Lead | 2026-10-20 |
| H-002 | False positive — unnecessary clinical response | S2 | P4→P3 | 🟢 | SHAP explanations, configurable thresholds, trend display, calibration | ✅ | 🟢 | ML Lead | 2026-10-20 |
| H-003 | Model degradation from data drift | S3–S4 | P3→P2 | 🟡 ALARP | Drift detector, performance monitoring, alerting, continuous learning | ✅ | 🟡 ALARP | ML Lead | 2026-10-20 |
| H-004 | System unavailability | S2 | P3→P2 | 🟢 | Docker auto-restart, health checks, Kafka fallback | ✅ | 🟢 | DevOps | 2026-10-20 |
| H-005 | Data integrity compromise | S3 | P2→P1 | 🟢 | Pydantic validation, clinical range checks, audit logging | ✅ | 🟢 | Eng Lead | 2026-10-20 |
| H-006 | Cybersecurity breach | S3–S4 | P2→P1 | 🟢 | JWT auth, network segmentation, SBOM scanning, TLS | ✅ | 🟢 | SecOps | 2026-10-20 |
| H-007 | Algorithmic bias across demographics | S3 | P3→P2 | 🟡 ALARP | Subgroup analysis, stratified evaluation, continuous monitoring | ⏳ Partial | 🟡 ALARP | ML Lead | 2026-10-20 |

## Closed Hazards

*No hazards have been closed. All identified hazards remain under active monitoring.*

## Hazard Identification Sources

| Source | Frequency | Last Review |
|--------|-----------|-------------|
| Design FMEA | Per design change | 2026-10-20 |
| Clinical literature review | Quarterly | 2026-10-20 |
| Complaint analysis | As received | — |
| Internal audit findings | Annually | — |
| Post-market surveillance | Continuous | — |

## Review Schedule

- **Quarterly review**: All active hazards re-assessed
- **Triggered review**: Upon any design change (per SOP-004), complaint (per SOP-005), or CAPA (per SOP-006)
- **Annual full review**: Comprehensive hazard identification exercise

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial hazard log — 7 active hazards |
