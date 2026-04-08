"""
JeevanSync AI — Schema Validator
=================================
Validates incoming ICU data against clinical range constraints.
Flags and quarantines out-of-range values. Enforces column presence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import feature_settings


# ── Clinical Range Definitions ───────────────────────────────────────────

CLINICAL_RANGES: dict[str, tuple[float, float]] = {
    # Vital signs
    "heart_rate": (20, 300),
    "respiratory_rate": (4, 60),
    "spo2_pct": (50, 100),
    "temperature_c": (30.0, 45.0),
    "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200),
    "oxygen_flow": (0, 100),
    "mobility_score": (0, 4),
    "nurse_alert": (0, 1),
    # Labs
    "wbc_count": (0, 100),
    "lactate": (0, 30),
    "creatinine": (0, 20),
    "crp_level": (0, 500),
    "hemoglobin": (2, 25),
    "sepsis_risk_score": (0, 1),
    # Demographics
    "age": (0, 120),
    "comorbidity_index": (0, 20),
    "hour_from_admission": (0, 72),
}

REQUIRED_VITALS_COLUMNS = [
    "patient_id", "hour_from_admission",
    "heart_rate", "respiratory_rate", "spo2_pct",
    "temperature_c", "systolic_bp", "diastolic_bp",
    "oxygen_device", "oxygen_flow", "mobility_score", "nurse_alert",
]

REQUIRED_LABS_COLUMNS = [
    "patient_id", "hour_from_admission",
    "wbc_count", "lactate", "creatinine",
    "crp_level", "hemoglobin", "sepsis_risk_score",
]

REQUIRED_PANEL_COLUMNS = REQUIRED_VITALS_COLUMNS + [
    "wbc_count", "lactate", "creatinine", "crp_level",
    "hemoglobin", "sepsis_risk_score",
    "age", "gender", "comorbidity_index", "admission_type",
]

VALID_CATEGORIES = {
    "gender": {"M", "F"},
    "admission_type": {"ED", "Elective", "Transfer"},
    "oxygen_device": {"none", "nasal", "mask", "hfnc", "niv"},
}


@dataclass
class ValidationReport:
    """Stores schema validation results."""

    total_rows: int = 0
    valid_rows: int = 0
    flagged_rows: int = 0
    missing_columns: list[str] = field(default_factory=list)
    range_violations: dict[str, int] = field(default_factory=dict)
    category_violations: dict[str, int] = field(default_factory=dict)
    type_errors: dict[str, int] = field(default_factory=dict)
    is_valid: bool = True

    def summary(self) -> str:
        lines = [
            f"Validation Report:",
            f"  Total rows:      {self.total_rows}",
            f"  Valid rows:      {self.valid_rows}",
            f"  Flagged rows:    {self.flagged_rows}",
            f"  Missing columns: {self.missing_columns or 'None'}",
        ]
        if self.range_violations:
            lines.append(f"  Range violations: {self.range_violations}")
        if self.category_violations:
            lines.append(f"  Category violations: {self.category_violations}")
        return "\n".join(lines)


class SchemaValidator:
    """
    Validates ICU DataFrames against expected schema and clinical ranges.
    """

    def __init__(
        self,
        clinical_ranges: Optional[dict[str, tuple[float, float]]] = None,
        valid_categories: Optional[dict[str, set[str]]] = None,
    ):
        self.clinical_ranges = clinical_ranges or CLINICAL_RANGES
        self.valid_categories = valid_categories or VALID_CATEGORIES

    def validate(
        self,
        df: pd.DataFrame,
        required_columns: Optional[list[str]] = None,
        mode: str = "panel",
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Validate the DataFrame and return cleaned data + report.

        Args:
            df: Input DataFrame
            required_columns: Columns that must be present
            mode: 'vitals', 'labs', or 'panel'

        Returns:
            (cleaned_df, report)
        """
        if required_columns is None:
            required_columns = {
                "vitals": REQUIRED_VITALS_COLUMNS,
                "labs": REQUIRED_LABS_COLUMNS,
                "panel": REQUIRED_PANEL_COLUMNS,
            }.get(mode, REQUIRED_PANEL_COLUMNS)

        report = ValidationReport(total_rows=len(df))

        # 1. Check required columns
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            report.missing_columns = missing
            report.is_valid = False
            logger.error("Missing required columns: {}", missing)

        # 2. Create a validity mask (all True initially)
        valid_mask = pd.Series(True, index=df.index)

        # 3. Check numeric ranges
        for col, (lo, hi) in self.clinical_ranges.items():
            if col not in df.columns:
                continue

            # Ensure numeric
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            type_errors = numeric_col.isna() & df[col].notna()
            if type_errors.any():
                count = int(type_errors.sum())
                report.type_errors[col] = count
                valid_mask &= ~type_errors
                logger.warning("{} type errors in column '{}'", count, col)

            # Range check
            out_of_range = (numeric_col < lo) | (numeric_col > hi)
            violations = int(out_of_range.sum())
            if violations > 0:
                report.range_violations[col] = violations
                valid_mask &= ~out_of_range
                logger.warning(
                    "{} range violations in '{}' (expected [{}, {}])",
                    violations, col, lo, hi,
                )

        # 4. Check categorical values
        for col, valid_values in self.valid_categories.items():
            if col not in df.columns:
                continue

            invalid = ~df[col].astype(str).isin(valid_values)
            violations = int(invalid.sum())
            if violations > 0:
                report.category_violations[col] = violations
                valid_mask &= ~invalid
                logger.warning(
                    "{} invalid categories in '{}' (expected {})",
                    violations, col, valid_values,
                )

        # 5. Compute final stats
        report.valid_rows = int(valid_mask.sum())
        report.flagged_rows = report.total_rows - report.valid_rows
        report.is_valid = report.is_valid and report.flagged_rows == 0

        cleaned_df = df[valid_mask].copy().reset_index(drop=True)
        logger.info(report.summary())

        return cleaned_df, report

    def enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast columns to their expected data types.
        Numeric columns → float64, integer columns → Int64 (nullable).
        """
        df = df.copy()

        # Integer columns
        int_columns = ["patient_id", "hour_from_admission", "mobility_score", "nurse_alert",
                        "age", "comorbidity_index"]
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Float columns
        float_columns = [
            "heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
            "systolic_bp", "diastolic_bp", "oxygen_flow",
            "wbc_count", "lactate", "creatinine", "crp_level",
            "hemoglobin", "sepsis_risk_score",
        ]
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # Categorical columns
        for col in ["gender", "admission_type", "oxygen_device"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        logger.debug("Data types enforced")
        return df

    def validate_patient_continuity(self, df: pd.DataFrame) -> dict[int, list[int]]:
        """
        Check that each patient has continuous hour_from_admission values.
        Returns dict of patient_id -> list of missing hours.
        """
        gaps = {}
        for pid, group in df.groupby("patient_id"):
            hours = sorted(group["hour_from_admission"].values)
            expected = list(range(hours[0], hours[-1] + 1))
            missing = sorted(set(expected) - set(hours))
            if missing:
                gaps[pid] = missing

        if gaps:
            logger.warning(
                "{} patients have gaps in hourly data", len(gaps)
            )
        else:
            logger.info("All patients have continuous hourly data")

        return gaps
