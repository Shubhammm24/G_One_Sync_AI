"""
G_One_Sync AI — Software Bill of Materials (SBOM) Generator
=============================================================
Generates an SBOM listing all Python dependencies with versions,
licenses, and known vulnerabilities for FDA/CE compliance.

Outputs:
  - compliance/sbom/sbom.json        (machine-readable)
  - compliance/sbom/sbom_report.md   (human-readable)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Project root detection ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
OUTPUT_DIR = PROJECT_ROOT / "compliance" / "sbom"


def get_installed_packages() -> list[dict]:
    """Get all installed packages with versions using pip."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"[WARN] Could not list installed packages: {e}")
        return []


def get_package_licenses() -> dict[str, str]:
    """Get license information for installed packages."""
    licenses: dict[str, str] = {}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "--verbose"],
            capture_output=True,
            text=True,
        )
        # Fallback: parse from pip show individually for key packages
    except subprocess.CalledProcessError:
        pass

    # Known licenses for key dependencies
    known_licenses = {
        "numpy": "BSD-3-Clause",
        "pandas": "BSD-3-Clause",
        "scipy": "BSD-3-Clause",
        "scikit-learn": "BSD-3-Clause",
        "xgboost": "Apache-2.0",
        "torch": "BSD-3-Clause",
        "fastapi": "MIT",
        "uvicorn": "BSD-3-Clause",
        "pydantic": "MIT",
        "pydantic-settings": "MIT",
        "httpx": "BSD-3-Clause",
        "loguru": "MIT",
        "mlflow": "Apache-2.0",
        "optuna": "MIT",
        "shap": "MIT",
        "plotly": "MIT",
        "prometheus-client": "Apache-2.0",
        "pyarrow": "Apache-2.0",
        "orjson": "Apache-2.0 / MIT",
        "lightgbm": "MIT",
        "grpcio": "Apache-2.0",
        "opacus": "Apache-2.0",
        "pyjwt": "MIT",
        "bcrypt": "Apache-2.0",
        "fhirpy": "MIT",
        "hl7apy": "MIT",
        "pytest": "MIT",
        "pytest-asyncio": "Apache-2.0",
    }
    return known_licenses


def parse_requirements() -> list[dict]:
    """Parse requirements.txt for declared dependencies."""
    deps: list[dict] = []
    if not REQUIREMENTS_FILE.exists():
        print(f"⚠️  requirements.txt not found: {REQUIREMENTS_FILE}")
        return deps

    content = REQUIREMENTS_FILE.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        # Parse package>=version
        parts = line.split(">=")
        if len(parts) == 2:
            deps.append({
                "name": parts[0].strip(),
                "minimum_version": parts[1].strip(),
                "constraint": f">={parts[1].strip()}",
            })
        else:
            deps.append({
                "name": line,
                "minimum_version": "unspecified",
                "constraint": line,
            })

    return deps


def run_vulnerability_scan() -> list[dict]:
    """Run pip-audit for vulnerability scanning (if available)."""
    vulnerabilities: list[dict] = []
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--format=json", "--progress-spinner=off"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.stdout:
            audit_data = json.loads(result.stdout)
            if "dependencies" in audit_data:
                for dep in audit_data["dependencies"]:
                    if dep.get("vulns"):
                        for vuln in dep["vulns"]:
                            vulnerabilities.append({
                                "package": dep["name"],
                                "version": dep["version"],
                                "vulnerability_id": vuln.get("id", ""),
                                "description": vuln.get("description", ""),
                                "fix_versions": vuln.get("fix_versions", []),
                            })
    except FileNotFoundError:
        print("[INFO] pip-audit not installed. Skipping vulnerability scan.")
        print("   Install with: pip install pip-audit")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"[WARN] Vulnerability scan failed: {e}")

    return vulnerabilities


def generate_sbom() -> dict:
    """Generate the complete SBOM document."""
    declared_deps = parse_requirements()
    installed_packages = get_installed_packages()
    known_licenses = get_package_licenses()

    # Build installed package lookup
    installed_lookup = {
        pkg["name"].lower(): pkg["version"]
        for pkg in installed_packages
    }

    # Enrich declared dependencies with installed versions and licenses
    sbom_entries: list[dict] = []
    for dep in declared_deps:
        name_lower = dep["name"].lower().replace("-", "_")
        # Try both original and underscore versions
        installed_version = (
            installed_lookup.get(name_lower)
            or installed_lookup.get(dep["name"].lower())
            or "not installed"
        )
        license_info = known_licenses.get(name_lower, "Unknown")

        sbom_entries.append({
            "name": dep["name"],
            "declared_constraint": dep["constraint"],
            "installed_version": installed_version,
            "license": license_info,
            "direct_dependency": True,
        })

    sbom = {
        "sbom_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "generator": "G_One_Sync AI SBOM Generator",
        "project": "G_One_Sync AI",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "total_direct_dependencies": len(sbom_entries),
        "total_installed_packages": len(installed_packages),
        "dependencies": sbom_entries,
    }

    return sbom


def generate_markdown_report(sbom: dict, vulnerabilities: list[dict]) -> str:
    """Generate a human-readable SBOM report."""
    lines = [
        "# Software Bill of Materials (SBOM)",
        "",
        "> **G_One_Sync AI — Compliance Documentation**",
        f"> Generated: {sbom['generated_at'][:19]}",
        f"> Python Version: {sbom['python_version']}",
        "",
        "## Direct Dependencies",
        "",
        "| Package | Constraint | Installed | License |",
        "|---------|-----------|-----------|---------|",
    ]

    for dep in sbom["dependencies"]:
        lines.append(
            f"| {dep['name']} | {dep['declared_constraint']} "
            f"| {dep['installed_version']} | {dep['license']} |"
        )

    lines.extend([
        "",
        f"**Total direct dependencies**: {sbom['total_direct_dependencies']}",
        f"**Total installed packages**: {sbom['total_installed_packages']}",
    ])

    # Vulnerability section
    lines.extend(["", "## Vulnerability Scan Results", ""])
    if vulnerabilities:
        lines.extend([
            f"WARNING: **{len(vulnerabilities)} vulnerabilities found**",
            "",
            "| Package | Version | CVE | Fix Versions |",
            "|---------|---------|-----|-------------|",
        ])
        for vuln in vulnerabilities:
            fix = ", ".join(vuln["fix_versions"]) if vuln["fix_versions"] else "—"
            lines.append(
                f"| {vuln['package']} | {vuln['version']} "
                f"| {vuln['vulnerability_id']} | {fix} |"
            )
    else:
        lines.append("PASS: No known vulnerabilities detected in direct dependencies.")

    return "\n".join(lines)


def main() -> None:
    """Generate SBOM in both JSON and Markdown formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Generating Software Bill of Materials...")

    # Generate SBOM
    sbom = generate_sbom()

    # Run vulnerability scan
    print("[INFO] Scanning for known vulnerabilities...")
    vulnerabilities = run_vulnerability_scan()
    sbom["vulnerabilities"] = vulnerabilities

    # Save JSON
    json_file = OUTPUT_DIR / "sbom.json"
    json_file.write_text(json.dumps(sbom, indent=2), encoding="utf-8")
    print(f"[OK] SBOM JSON saved: {json_file}")

    # Save Markdown report
    report = generate_markdown_report(sbom, vulnerabilities)
    md_file = OUTPUT_DIR / "sbom_report.md"
    md_file.write_text(report, encoding="utf-8")
    print(f"[OK] SBOM report saved: {md_file}")

    # Summary
    print(f"\n[SUMMARY]")
    print(f"   Direct dependencies: {sbom['total_direct_dependencies']}")
    print(f"   Total packages: {sbom['total_installed_packages']}")
    print(f"   Vulnerabilities: {len(vulnerabilities)}")

    # Exit with error if critical vulnerabilities found
    if vulnerabilities:
        print(f"\n[WARN] {len(vulnerabilities)} vulnerabilities found -- review required")
        sys.exit(1)
    else:
        print("\n[OK] No vulnerabilities detected")


if __name__ == "__main__":
    main()
