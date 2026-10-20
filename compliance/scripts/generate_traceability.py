"""
G_One_Sync AI — Traceability Matrix Generator
===============================================
Auto-generates requirement ↔ implementation ↔ test traceability matrix
for IEC 62304 compliance and FDA 510(k) submissions.

Scans:
  - compliance/design_controls/design_inputs.md  →  Requirements (DI-xxx)
  - src/                                          →  Implementation files
  - tests/                                        →  Test coverage
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── Project root detection ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DESIGN_INPUTS = PROJECT_ROOT / "compliance" / "design_controls" / "design_inputs.md"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_DIR = PROJECT_ROOT / "compliance" / "traceability"


def extract_design_inputs(filepath: Path) -> list[dict]:
    """Parse design input IDs and descriptions from design_inputs.md."""
    requirements: list[dict] = []
    if not filepath.exists():
        print(f"[WARN] Design inputs file not found: {filepath}")
        return requirements

    content = filepath.read_text(encoding="utf-8")
    # Match pattern: ### DI-NNN: Title
    pattern = re.compile(
        r"###\s+(DI-\d{3}):\s+(.+?)$",
        re.MULTILINE,
    )
    for match in pattern.finditer(content):
        req_id = match.group(1)
        title = match.group(2).strip()
        requirements.append({"id": req_id, "title": title})

    return requirements


def find_implementation_references(req_id: str, src_dir: Path) -> list[str]:
    """Search source files for references to a requirement ID."""
    references: list[str] = []
    if not src_dir.exists():
        return references

    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if req_id in content:
                rel_path = py_file.relative_to(PROJECT_ROOT)
                references.append(str(rel_path))
        except (OSError, UnicodeDecodeError):
            continue

    return references


def find_test_coverage(req_id: str, tests_dir: Path) -> list[str]:
    """Search test files for references to a requirement ID."""
    test_files: list[str] = []
    if not tests_dir.exists():
        return test_files

    for py_file in tests_dir.rglob("test_*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if req_id in content:
                rel_path = py_file.relative_to(PROJECT_ROOT)
                test_files.append(str(rel_path))
        except (OSError, UnicodeDecodeError):
            continue

    return test_files


def map_requirement_to_modules(req_id: str) -> list[str]:
    """Static mapping of requirements to known implementation modules."""
    mapping: dict[str, list[str]] = {
        "DI-001": ["src/serving/model_server.py"],
        "DI-002": ["src/modeling/ensemble.py"],
        "DI-003": ["src/serving/model_server.py"],
        "DI-004": ["src/explainability/shap_explainer.py"],
        "DI-005": ["src/modeling/train_pipeline.py"],
        "DI-006": ["src/modeling/calibration.py"],
        "DI-007": ["config/settings.py"],
        "DI-008": ["src/monitoring/alerting.py"],
        "DI-009": ["src/serving/push_notifications.py"],
        "DI-010": ["config/settings.py"],
        "DI-011": ["src/dashboard/app.py"],
        "DI-012": ["mobile/"],
        "DI-013": ["src/ingestion/schemas.py"],
        "DI-014": ["src/ingestion/emr/fhir_mapper.py"],
        "DI-015": ["src/ingestion/emr/hl7_mapper.py"],
        "DI-016": ["docker/docker-compose.yml"],
        "DI-017": ["src/ingestion/kafka_producer.py"],
        "DI-018": ["src/serving/auth.py"],
        "DI-019": ["config/logging_config.py"],
    }
    return mapping.get(req_id, [])


def map_requirement_to_tests(req_id: str) -> list[str]:
    """Static mapping of requirements to known test modules."""
    mapping: dict[str, list[str]] = {
        "DI-001": ["tests/test_serving.py"],
        "DI-002": ["tests/test_modeling.py"],
        "DI-003": ["tests/test_serving.py"],
        "DI-004": ["tests/test_serving.py"],
        "DI-005": ["tests/test_modeling.py"],
        "DI-006": ["tests/test_modeling.py"],
        "DI-007": ["tests/test_serving.py"],
        "DI-008": ["tests/test_serving.py"],
        "DI-009": ["tests/test_mobile_api.py"],
        "DI-010": ["tests/test_serving.py"],
        "DI-011": [],
        "DI-012": ["tests/test_mobile_api.py"],
        "DI-013": ["tests/test_ingestion.py"],
        "DI-014": ["tests/test_emr_integration.py"],
        "DI-015": ["tests/test_emr_integration.py"],
        "DI-016": [],
        "DI-017": ["tests/test_ingestion.py"],
        "DI-018": ["tests/test_mobile_api.py"],
        "DI-019": [],
    }
    return mapping.get(req_id, [])


def generate_traceability_matrix() -> str:
    """Generate the full traceability matrix as a Markdown document."""
    requirements = extract_design_inputs(DESIGN_INPUTS)

    lines: list[str] = [
        "# Requirements Traceability Matrix",
        "",
        "> **Auto-generated by `compliance/scripts/generate_traceability.py`**",
        f"> Generated: {__import__('datetime').datetime.now().isoformat()[:19]}",
        "",
        "## Traceability: Requirement → Implementation → Test",
        "",
        "| Req ID | Requirement | Implementation | Tests | Status |",
        "|:------:|-------------|---------------|-------|:------:|",
    ]

    total = 0
    covered = 0

    for req in requirements:
        total += 1
        req_id = req["id"]
        title = req["title"]

        # Combine static mapping with dynamic search
        impl_files = map_requirement_to_modules(req_id)
        dynamic_refs = find_implementation_references(req_id, SRC_DIR)
        all_impl = sorted(set(impl_files + dynamic_refs))

        test_files = map_requirement_to_tests(req_id)
        dynamic_tests = find_test_coverage(req_id, TESTS_DIR)
        all_tests = sorted(set(test_files + dynamic_tests))

        impl_str = ", ".join(f"`{f}`" for f in all_impl) if all_impl else "—"
        test_str = ", ".join(f"`{f}`" for f in all_tests) if all_tests else "—"

        has_impl = len(all_impl) > 0
        has_test = len(all_tests) > 0

        if has_impl and has_test:
            status = "✅ Covered"
            covered += 1
        elif has_impl:
            status = "⚠️ No Test"
        else:
            status = "❌ Missing"

        lines.append(f"| {req_id} | {title} | {impl_str} | {test_str} | {status} |")

    # Summary
    coverage_pct = (covered / total * 100) if total > 0 else 0
    lines.extend([
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|:-----:|",
        f"| Total Requirements | {total} |",
        f"| Fully Covered (impl + test) | {covered} |",
        f"| Coverage Rate | {coverage_pct:.1f}% |",
    ])

    return "\n".join(lines)


def main() -> None:
    """Generate and save the traceability matrix."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "traceability_matrix.md"

    matrix_content = generate_traceability_matrix()
    output_file.write_text(matrix_content, encoding="utf-8")

    print(f"[OK] Traceability matrix generated: {output_file}")

    # Also generate JSON for CI consumption
    requirements = extract_design_inputs(DESIGN_INPUTS)
    json_data = []
    for req in requirements:
        req_id = req["id"]
        json_data.append({
            "id": req_id,
            "title": req["title"],
            "implementation": map_requirement_to_modules(req_id),
            "tests": map_requirement_to_tests(req_id),
        })

    json_file = OUTPUT_DIR / "traceability_matrix.json"
    json_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"[OK] Traceability JSON generated: {json_file}")


if __name__ == "__main__":
    main()
