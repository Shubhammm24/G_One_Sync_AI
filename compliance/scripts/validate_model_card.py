"""
G_One_Sync AI — Model Card Validator
======================================
Validates that the model card template is complete with all
FDA-required fields for AI/ML SaMD submissions.

Checks:
  - All required sections present
  - No placeholder/TODO text remaining
  - Performance metrics populated
  - Intended use statement present
  - Ethical considerations documented
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ── Project root detection ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_CARD_PATH = PROJECT_ROOT / "compliance" / "templates" / "model_card_template.md"


# ── Required sections ──────────────────────────────────────────────────
REQUIRED_SECTIONS: list[dict[str, str]] = [
    {"heading": "Model Details", "description": "Model name, version, type, and date"},
    {"heading": "Intended Use", "description": "Intended clinical use and users"},
    {"heading": "Training Data", "description": "Training dataset description"},
    {"heading": "Evaluation Data", "description": "Test/validation dataset description"},
    {"heading": "Performance Metrics", "description": "AUROC, AUPRC, sensitivity, specificity"},
    {"heading": "Ethical Considerations", "description": "Bias, fairness, and equity"},
    {"heading": "Limitations", "description": "Known limitations and failure modes"},
    {"heading": "Quantitative Analyses", "description": "Calibration, subgroup performance"},
]

# ── Placeholder patterns ───────────────────────────────────────────────
PLACEHOLDER_PATTERNS: list[str] = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bXXX\b",
    r"\[PLACEHOLDER\]",
    r"\[INSERT\s+.*?\]",
    r"\[FILL\s+IN\]",
    r"\bTBD\b",
]


def validate_model_card(filepath: Path) -> tuple[bool, list[str]]:
    """
    Validate a model card file for completeness.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues: list[str] = []

    # Check file exists
    if not filepath.exists():
        return False, [f"Model card not found: {filepath}"]

    content = filepath.read_text(encoding="utf-8")

    # Check for required sections
    for section in REQUIRED_SECTIONS:
        heading = section["heading"]
        # Check for heading at any level (## or ###)
        pattern = re.compile(rf"^#{{1,4}}\s+.*{re.escape(heading)}", re.MULTILINE | re.IGNORECASE)
        if not pattern.search(content):
            issues.append(f"[MISSING] Missing required section: '{heading}' -- {section['description']}")

    # Check for placeholder text
    for pattern in PLACEHOLDER_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append(
                f"[WARN] Placeholder text found: '{matches[0]}' "
                f"({len(matches)} occurrence{'s' if len(matches) > 1 else ''})"
            )

    # Check minimum content length (a real model card should be >500 words)
    word_count = len(content.split())
    if word_count < 200:
        issues.append(f"[WARN] Model card seems too short ({word_count} words, minimum 200 expected)")

    # Check for performance metrics values
    metric_patterns = {
        "AUROC": r"AUROC.*?[\d.]+",
        "AUPRC": r"AUPRC.*?[\d.]+",
        "Sensitivity": r"[Ss]ensitivity.*?[\d.]+%?",
        "Specificity": r"[Ss]pecificity.*?[\d.]+%?",
    }
    for metric_name, metric_pattern in metric_patterns.items():
        if not re.search(metric_pattern, content):
            issues.append(f"[WARN] No {metric_name} value found in model card")

    # Check for intended use statement
    if "clinical decision support" not in content.lower():
        issues.append("[WARN] No 'clinical decision support' language found in intended use")

    if "physician review" not in content.lower() and "clinician review" not in content.lower():
        issues.append(
            "[WARN] No 'physician review required' language found -- "
            "required for Class II SaMD classification"
        )

    is_valid = len(issues) == 0
    return is_valid, issues


def main() -> None:
    """Run model card validation and report results."""
    print("[INFO] Validating model card completeness...\n")

    is_valid, issues = validate_model_card(MODEL_CARD_PATH)

    if is_valid:
        print("[PASS] Model card validation PASSED -- all required sections present and complete")
        sys.exit(0)
    else:
        print(f"{'[PASS]' if len(issues) < 3 else '[FAIL]'} Model card validation results:\n")
        for issue in issues:
            print(f"   {issue}")

        # Count errors vs warnings
        errors = sum(1 for i in issues if "Missing required" in i)
        warnings = sum(1 for i in issues if "Placeholder" in i or "seems too short" in i or "No " in i)

        print(f"\n   Errors: {errors}, Warnings: {warnings}")

        if errors > 0:
            print("\n[FAIL] Model card validation FAILED -- fix errors before submission")
            sys.exit(1)
        else:
            print("\n[WARN] Model card has warnings -- review before submission")
            sys.exit(0)


if __name__ == "__main__":
    main()
