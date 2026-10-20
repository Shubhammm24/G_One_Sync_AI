# SOP-003: Software Development Lifecycle

> **G_One_Sync AI — Quality Management System**
> Standard Operating Procedure for Software Development Lifecycle
> Per IEC 62304:2006+A1:2015 — Medical Device Software — Software Life Cycle Processes

---

## 1. Purpose

Define the software development lifecycle (SDLC) processes for G_One_Sync AI, ensuring all software activities comply with IEC 62304 requirements for Class B medical device software.

## 2. Software Safety Classification

### 2.1 Classification Determination

| Criterion | Assessment |
|-----------|-----------|
| **Can the software cause harm?** | No — software provides decision support only; clinician makes all treatment decisions |
| **Can harm result from incorrect output?** | Yes — delayed recognition of deterioration if clinician relies solely on prediction |
| **Are there external risk controls?** | Yes — physician clinical judgment, nursing assessments, existing monitoring equipment |
| **Classification** | **Class B** — Software system that can contribute to a hazardous situation but external risk controls exist |

### 2.2 Classification Rationale

G_One_Sync AI is classified as **IEC 62304 Class B** because:
- It provides supplementary clinical decision support, not autonomous treatment decisions
- A false negative requires the simultaneous failure of: (1) the AI system, (2) nursing bedside assessment, (3) physician clinical rounds, and (4) existing physiological monitors
- The intended use statement explicitly requires physician review before clinical action

## 3. Software Development Process

### 3.1 Development Planning (IEC 62304 §5.1)

Each development iteration shall include:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Planning   │───▶│Requirements  │───▶│Architecture  │───▶│  Detailed    │
│              │    │  Analysis    │    │   Design     │    │   Design     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│   Release    │◀───│ Integration  │◀───│    Unit      │◀─────────┘
│              │    │   Testing    │    │  Implementation│
└──────────────┘    └──────────────┘    └──────────────┘
```

### 3.2 Software Requirements (IEC 62304 §5.2)

Software requirements are derived from:
1. **User Needs** → `compliance/design_controls/user_needs.md`
2. **Design Inputs** → `compliance/design_controls/design_inputs.md`
3. **Risk Controls** → `compliance/risk_management/risk_analysis.md`

Each requirement shall:
- Have a unique identifier (e.g., `REQ-PRED-001`)
- Be testable with objective acceptance criteria
- Be traceable to user needs and risk controls
- Be traceable to implementation (source code) and tests

### 3.3 Software Architecture (IEC 62304 §5.3)

The software architecture is documented in `README.md` and consists of:

| Software Item | Classification | Description |
|---------------|---------------|-------------|
| `src/ingestion/` | Class B | Data ingestion and validation |
| `src/preprocessing/` | Class B | Feature engineering pipeline |
| `src/modeling/` | Class B | ML model training and ensemble |
| `src/serving/` | Class B | Prediction API serving |
| `src/explainability/` | Class A | SHAP explanation generation |
| `src/monitoring/` | Class A | Performance monitoring and drift |
| `src/dashboard/` | Class A | Clinical visualization |
| `config/` | Class B | System configuration |

### 3.4 Software Unit Implementation (IEC 62304 §5.5)

Coding standards:
- **Language**: Python ≥ 3.10 with mandatory type hints
- **Linter**: Ruff (configured in `pyproject.toml`)
- **Style**: Per `AGENTS.md` coding conventions
- **Documentation**: Docstrings on all public classes and functions
- **Version Control**: Git with conventional commit messages

### 3.5 Software Unit Verification (IEC 62304 §5.5.5)

| Verification Activity | Tool | Minimum Criteria |
|----------------------|------|-----------------|
| Static analysis | Ruff | Zero errors (E, F, I, W rules) |
| Unit testing | pytest | ≥80% line coverage |
| Type checking | Ruff / mypy | No type errors in public APIs |
| Code review | Git PR | At least 1 reviewer approval |

### 3.6 Software Integration Testing (IEC 62304 §5.6)

Integration tests verify:
1. API endpoints return correct status codes and schemas
2. Model predictions produce valid probabilities [0, 1]
3. Feature pipeline output dimensions match model input expectations
4. SHAP explanations are generated for all features
5. Dashboard WebSocket connections deliver real-time updates

### 3.7 Software System Testing (IEC 62304 §5.7)

System-level tests include:
1. End-to-end data flow: raw vitals → preprocessing → prediction → dashboard
2. Performance under load: ≥100 concurrent predictions with p99 < 500ms
3. Failover scenarios: Kafka unavailable → local fallback queue
4. Model versioning: hot-swap model artifacts without API restart

## 4. Software Maintenance (IEC 62304 §6)

### 4.1 Problem Resolution

Software problems are tracked via GitHub Issues with the following labels:
- `bug/critical` — Patient safety impact, immediate fix required
- `bug/major` — Functional impact, fix within 1 sprint
- `bug/minor` — Cosmetic or minor usability, scheduled fix
- `enhancement` — Feature request

### 4.2 Change Control

All changes follow `SOP-004_change_management.md`:
1. Change request with impact assessment
2. Risk analysis update if safety-relevant
3. Implementation with tests
4. Verification appropriate to the change scope
5. Release with updated documentation

### 4.3 Continuous Integration

The CI pipeline (`.github/workflows/ci.yml`) enforces:
- Automated linting on every push
- Full test suite execution
- Coverage reporting
- Import validation for all critical modules

## 5. Software Configuration Management (IEC 62304 §8)

| Item | Method |
|------|--------|
| Source code | Git (GitHub) |
| Model artifacts | MLflow Model Registry |
| Configuration | `config/settings.py` + environment variables |
| Dependencies | `requirements.txt` with pinned minimum versions |
| Docker images | Versioned tags in `docker/docker-compose.yml` |

### 5.1 Software Bill of Materials (SBOM)

An SBOM is generated automatically via `compliance/scripts/generate_sbom.py` and includes:
- All Python packages with versions
- System dependencies
- Known vulnerabilities (via `pip-audit`)

## 6. Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-10-20 | G_One_Sync Team | Initial release |
