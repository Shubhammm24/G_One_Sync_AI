# SOP-004: Change Management

> **G_One_Sync AI — Quality Management System**
> Standard Operating Procedure for Change Management
> Per FDA 21 CFR 820.30(i) & ISO 13485:2016 §7.3.9

---

## 1. Purpose

Establish procedures for controlling changes to the G_One_Sync AI system after initial design verification, ensuring that changes do not introduce new risks or degrade clinical performance.

## 2. Scope

This SOP applies to changes in:
- Source code (`src/`, `config/`)
- ML model architectures, hyperparameters, or training data
- API contracts and endpoint specifications
- Dependencies (`requirements.txt`)
- Deployment configuration (`docker/`)
- Clinical labeling or intended use statements

## 3. Change Classification

### 3.1 Change Categories

| Category | Description | Examples | Approval Level |
|----------|-------------|----------|---------------|
| **Critical** | Affects clinical safety or regulatory status | Model architecture change, prediction threshold modification, intended use change | QA/RA Lead + Clinical Advisor + Project Lead |
| **Major** | Significant functional change | New API endpoint, new feature engineering, dependency upgrade | Project Lead + 1 Reviewer |
| **Minor** | Low-risk improvement | Bug fix, documentation update, logging improvement | 1 Reviewer |
| **Administrative** | No functional impact | Comment changes, formatting, CI config tweaks | Self-approval with PR |

### 3.2 Change Impact Assessment

Every change request shall document:

```
Change Request ID: CR-YYYY-NNN
Date: YYYY-MM-DD
Requestor: [Name]
Category: [Critical / Major / Minor / Administrative]

1. Description of Change
   [What is being changed and why]

2. Affected Components
   [List of files, modules, APIs, models impacted]

3. Clinical Safety Impact
   [ ] No impact on clinical predictions
   [ ] Impacts model performance (re-verification required)
   [ ] Impacts prediction interpretation (re-validation required)
   [ ] Impacts intended use (regulatory re-submission required)

4. Risk Assessment
   [ ] No new hazards introduced
   [ ] New hazards identified → update FMEA (compliance/risk_management/fmea.md)
   [ ] Existing risk controls affected → update risk analysis

5. Verification Plan
   [Tests to be run, acceptance criteria]

6. Rollback Plan
   [How to revert if the change causes issues]
```

## 4. Change Control Process

### 4.1 Process Flow

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Request    │───▶│  Impact    │───▶│  Approval  │───▶│ Implement  │
│  Submitted  │    │ Assessment │    │  Decision  │    │  & Verify  │
└────────────┘    └────────────┘    └─────┬──────┘    └─────┬──────┘
                                          │                  │
                                    ┌─────▼──────┐    ┌─────▼──────┐
                                    │  Rejected  │    │  Release   │
                                    │  (w/ reason)│    │  & Close   │
                                    └────────────┘    └────────────┘
```

### 4.2 Model-Specific Changes

Changes to ML models require additional scrutiny:

| Change Type | Required Activities |
|-------------|-------------------|
| Training data modification | Re-train, re-evaluate on holdout set, compare AUROC/AUPRC |
| Hyperparameter tuning | Re-train with cross-validation, verify no overfitting |
| Architecture change | Full re-training, re-verification, re-calibration |
| Feature engineering | Verify feature distributions, re-train all models |
| Prediction threshold | Clinical review, sensitivity/specificity re-evaluation |

### 4.3 Emergency Changes

For urgent patient-safety issues:
1. Implement fix immediately with minimal review
2. Document change request retroactively within 24 hours
3. Conduct full impact assessment within 72 hours
4. Update risk management documentation within 1 week

## 5. Version Management

### 5.1 Semantic Versioning

G_One_Sync AI follows semantic versioning: `MAJOR.MINOR.PATCH`

| Component | Versioning Rule |
|-----------|----------------|
| **MAJOR** | Breaking API changes or regulatory re-submission required |
| **MINOR** | New features, model improvements (backwards-compatible) |
| **PATCH** | Bug fixes, documentation updates |

### 5.2 Model Versioning

Model artifacts are versioned independently:
- Format: `{model_type}_v{version}_{date}` (e.g., `xgboost_v2.1_20261020`)
- All versions retained in `models/versions/` for rollback
- MLflow experiment tracking for all training runs
- Active model version recorded in deployment configuration

## 6. Records

All change records are retained in:
- Git commit history (permanent)
- GitHub Pull Request discussions
- `compliance/change_log/` directory (formal change requests)

## 7. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial release |
