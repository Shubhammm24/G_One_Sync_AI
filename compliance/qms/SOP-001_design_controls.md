# SOP-001: Design Controls

> **G_One_Sync AI — Quality Management System**
> Standard Operating Procedure for Design Controls
> Per FDA 21 CFR 820.30 & ISO 13485:2016 §7.3

---

## 1. Purpose

Establish procedures for controlling the design and development of G_One_Sync AI, a Software as a Medical Device (SaMD) for clinical deterioration prediction. This SOP ensures that design outputs meet design inputs, user needs are validated, and all design activities are traceable.

## 2. Scope

This SOP applies to:
- All software modules within `src/` that contribute to clinical predictions
- Machine learning model architectures (XGBoost, BiLSTM, Transformer, Ensemble)
- Prediction serving API (`src/serving/model_server.py`)
- Data preprocessing pipeline (`src/preprocessing/pipeline.py`)
- Explainability module (`src/explainability/shap_explainer.py`)
- Clinical dashboard (`src/dashboard/app.py`)
- Mobile companion application (`mobile/`)

## 3. Definitions

| Term | Definition |
|------|-----------|
| **Design Input** | Physical and performance requirements used as a basis for device design |
| **Design Output** | Results of the design effort at each phase, including the final device |
| **Design Review** | Documented, comprehensive examination of a design at defined stages |
| **Design Verification** | Confirmation by examination and objective evidence that design outputs meet design input requirements |
| **Design Validation** | Establishing by objective evidence that the device meets user needs and intended uses |
| **Design Transfer** | Ensuring the device design is correctly translated into production |
| **DHF** | Design History File — collection of all design control records |

## 4. Responsibilities

| Role | Responsibility |
|------|---------------|
| **Project Lead** | Overall design control compliance; DHF maintenance |
| **ML Engineers** | Model design, implementation, verification testing |
| **Clinical Advisors** | User needs definition, validation study participation |
| **QA/RA Specialist** | SOP enforcement, audit readiness, regulatory submissions |
| **Software Engineers** | API design, integration testing, deployment procedures |

## 5. Design Control Process

### 5.1 Design Planning

Every design activity begins with a documented plan that includes:

1. **Objective Statement** — What clinical problem is being addressed
2. **Design Phase** — Which stage of development (concept, prototype, verification, validation, transfer)
3. **Deliverables** — Specific outputs expected at each milestone
4. **Review Schedule** — Points at which formal design reviews occur
5. **Responsible Individuals** — Named owners for each deliverable
6. **Resource Allocation** — Compute, data, and personnel requirements

**Document Control**: Design plans are version-controlled in Git with semantic versioning. All changes require a pull request with at least one reviewer.

### 5.2 Design Input Requirements

Design inputs shall be documented in `compliance/design_controls/design_inputs.md` and must include:

- **Functional requirements** — What the system shall do (e.g., predict deterioration within 12 hours)
- **Performance requirements** — Quantitative targets (e.g., AUROC ≥ 0.85, AUPRC ≥ 0.60)
- **Safety requirements** — Failure modes and acceptable error rates (e.g., false negative rate < 15%)
- **Interface requirements** — API contracts, data formats, integration points
- **Regulatory requirements** — FDA, CE, IEC 62304, ISO 14971 compliance
- **Clinical workflow requirements** — How predictions integrate into ICU workflows

**Acceptance Criteria**: Every design input shall have a measurable acceptance criterion.

### 5.3 Design Output Documentation

Design outputs shall be documented in `compliance/design_controls/design_outputs.md` and must include:

- **Source code** — All modules in `src/` with inline documentation
- **Model artifacts** — Trained models in `models/artifacts/` with model cards
- **API specifications** — OpenAPI schema from FastAPI auto-documentation
- **Configuration schema** — All settings in `config/settings.py` with validation rules
- **Test suite** — Comprehensive tests in `tests/` with ≥80% code coverage

**Traceability**: Each design output shall reference the design input(s) it satisfies.

### 5.4 Design Reviews

Formal design reviews shall be conducted at:

| Milestone | Review Focus | Required Attendees |
|-----------|-------------|-------------------|
| **Concept** | User needs completeness, risk acceptability | Clinical, Engineering, QA |
| **Prototype** | Architecture decisions, model selection | Engineering, ML, QA |
| **Pre-Verification** | Test plan adequacy, coverage gaps | QA, Engineering |
| **Pre-Validation** | Verification results, clinical protocol | Clinical, QA, Regulatory |
| **Pre-Release** | Full DHF review, labeling, IFU | All stakeholders |

**Documentation**: Design review minutes shall include:
- Date, attendees, and their roles
- Items reviewed
- Action items with owners and deadlines
- Decision: Approve / Approve with conditions / Reject

### 5.5 Design Verification

Verification confirms that design outputs meet design inputs. Methods include:

1. **Unit Testing** — `pytest tests/` with per-module coverage
2. **Integration Testing** — API endpoint testing with `TestClient`
3. **Performance Testing** — Model metrics evaluation (AUROC, AUPRC, calibration ECE)
4. **Load Testing** — API throughput and latency under simulated ICU load
5. **Security Testing** — Input validation, authentication bypass attempts

**Evidence**: Test results shall be captured in CI/CD artifacts (`.github/workflows/ci.yml`).

### 5.6 Design Validation

Validation confirms the device meets user needs in the intended use environment:

1. **Clinical Validation Study** — Prospective study per `docs/regulatory/clinical_validation/study_protocol.md`
2. **Usability Testing** — Clinician interaction with dashboard and mobile app
3. **Clinical Workflow Integration** — End-to-end testing in simulated ICU environment
4. **Expert Review** — Board-certified intensivists review prediction outputs

**Acceptance**: Validation is complete when all primary endpoints are met and clinical reviewers provide documented approval.

### 5.7 Design Transfer

Transfer activities ensure reliable production deployment:

1. **Containerization** — Docker images per `docker/docker-compose.yml`
2. **Infrastructure as Code** — Deployment configurations version-controlled
3. **Model Registry** — MLflow tracking for all production model artifacts
4. **Monitoring** — Prometheus metrics + drift detection operational
5. **Rollback Procedures** — Documented model version rollback capability

### 5.8 Design Changes

All design changes after initial verification shall follow `SOP-004_change_management.md`:

1. Change request documented with rationale
2. Impact assessment (clinical safety, performance, regulatory)
3. Verification/validation activities repeated as appropriate
4. Design review for significant changes
5. DHF updated

## 6. Design History File (DHF)

The DHF for G_One_Sync AI is maintained in the following locations:

| DHF Component | Location |
|--------------|----------|
| Design Plan | `compliance/design_controls/` |
| User Needs | `compliance/design_controls/user_needs.md` |
| Design Inputs | `compliance/design_controls/design_inputs.md` |
| Design Outputs | `compliance/design_controls/design_outputs.md` |
| Risk Analysis | `compliance/risk_management/risk_analysis.md` |
| Verification Reports | `tests/` + CI/CD artifacts |
| Validation Reports | `docs/regulatory/clinical_validation/` |
| Design Reviews | `compliance/design_reviews/` (meeting minutes) |
| Change History | Git commit history + `compliance/qms/SOP-004_change_management.md` |

## 7. Records

- All design control records shall be retained for the lifetime of the device plus 2 years
- Electronic records comply with FDA 21 CFR Part 11 (electronic signatures)
- Version-controlled documents in Git serve as the authoritative record

## 8. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial release |
