"""
JeevanSync AI — Missing Value Imputation
==========================================
Multiple imputation strategies for ICU time-series data.
Designed for robustness even though the current synthetic dataset
has no missing values — critical for production with real EMR feeds.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import feature_settings


class ImputationEngine:
    """
    Multi-strategy imputation engine for ICU data.

    Strategies:
    - forward_fill: Last Observation Carried Forward (LOCF) within patient stay
    - median: Per-feature median from training set
    - interpolate: Linear interpolation for short gaps
    - sentinel: Mark missing with a sentinel value + indicator flag
    """

    def __init__(self):
        self._training_medians: Optional[dict[str, float]] = None
        self._training_means: Optional[dict[str, float]] = None

    def fit(self, train_df: pd.DataFrame) -> "ImputationEngine":
        """
        Compute training-set statistics for imputation.
        Call this once on the training data before applying to any split.
        """
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        self._training_medians = train_df[numeric_cols].median().to_dict()
        self._training_means = train_df[numeric_cols].mean().to_dict()
        logger.info(
            "ImputationEngine fitted on {} numeric columns from {} rows",
            len(numeric_cols),
            len(train_df),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        strategy: str = "cascade",
        max_gap_for_interpolation: int = 3,
    ) -> pd.DataFrame:
        """
        Apply imputation to the DataFrame.

        Args:
            df: Input DataFrame (will not be modified in-place)
            strategy: 'forward_fill', 'median', 'interpolate', 'sentinel', or 'cascade'
            max_gap_for_interpolation: Maximum consecutive NaN gap for interpolation

        Returns:
            DataFrame with missing values imputed
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        missing_before = result[numeric_cols].isna().sum().sum()

        if missing_before == 0:
            logger.info("No missing values detected — imputation skipped")
            return result

        logger.info(
            "Imputing {} missing values using strategy='{}'",
            missing_before, strategy,
        )

        if strategy == "cascade":
            result = self._cascade_impute(result, numeric_cols, max_gap_for_interpolation)
        elif strategy == "forward_fill":
            result = self._forward_fill(result, numeric_cols)
        elif strategy == "median":
            result = self._median_impute(result, numeric_cols)
        elif strategy == "interpolate":
            result = self._interpolate(result, numeric_cols, max_gap_for_interpolation)
        elif strategy == "sentinel":
            result = self._sentinel_impute(result, numeric_cols)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        remaining_numeric = [c for c in numeric_cols if c in result.columns]
        missing_after = result[remaining_numeric].isna().sum().sum() if remaining_numeric else 0
        logger.info(
            "Imputation complete: {} → {} missing values",
            missing_before, missing_after,
        )

        return result

    def _cascade_impute(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        max_gap: int,
    ) -> pd.DataFrame:
        """
        Cascade imputation strategy (recommended for ICU data):
        1. Within-patient linear interpolation (for short gaps ≤ max_gap)
        2. Forward-fill within patient (LOCF)
        3. Backward-fill within patient (first-hour edge case)
        4. Global median fill (remaining NaN)
        """
        # Step 1: Interpolation within patient groups
        df = df.groupby("patient_id", group_keys=False).apply(
            lambda g: g.sort_values("hour_from_admission").apply(
                lambda col: col.interpolate(method="linear", limit=max_gap)
                if col.name in numeric_cols else col
            )
        )

        # Step 2: Forward-fill within patient
        df = df.groupby("patient_id", group_keys=False).apply(
            lambda g: g.sort_values("hour_from_admission").ffill()
        )

        # Step 3: Backward-fill within patient (for start-of-stay NaNs)
        df = df.groupby("patient_id", group_keys=False).apply(
            lambda g: g.sort_values("hour_from_admission").bfill()
        )

        # Step 4: Global median for any remaining NaN
        if self._training_medians:
            for col in numeric_cols:
                if col in self._training_medians and df[col].isna().any():
                    df[col] = df[col].fillna(self._training_medians[col])

        return df

    def _forward_fill(
        self, df: pd.DataFrame, numeric_cols: list[str]
    ) -> pd.DataFrame:
        """LOCF within each patient's stay."""
        if "patient_id" not in df.columns:
            return df.ffill()

        result = df.copy()
        if "hour_from_admission" in result.columns:
            result = result.sort_values(["patient_id", "hour_from_admission"])

        cols_to_fill = [c for c in numeric_cols if c in result.columns and c != "patient_id"]
        for col in cols_to_fill:
            result[col] = result.groupby("patient_id")[col].ffill()

        return result

    def _median_impute(
        self, df: pd.DataFrame, numeric_cols: list[str]
    ) -> pd.DataFrame:
        """Replace NaN with training-set median per feature."""
        if self._training_medians is None:
            raise RuntimeError("ImputationEngine not fitted. Call .fit() first.")

        for col in numeric_cols:
            if col in self._training_medians and df[col].isna().any():
                df[col] = df[col].fillna(self._training_medians[col])
        return df

    def _interpolate(
        self, df: pd.DataFrame, numeric_cols: list[str], max_gap: int
    ) -> pd.DataFrame:
        """Linear interpolation within patient, limited to short gaps."""
        return df.groupby("patient_id", group_keys=False).apply(
            lambda g: g.sort_values("hour_from_admission").apply(
                lambda col: col.interpolate(method="linear", limit=max_gap)
                if col.name in numeric_cols else col
            )
        )

    def _sentinel_impute(
        self, df: pd.DataFrame, numeric_cols: list[str], sentinel: float = -999.0
    ) -> pd.DataFrame:
        """
        Replace NaN with sentinel value and create binary indicator columns.
        Useful for tree-based models that can learn from missingness patterns.
        """
        for col in numeric_cols:
            if df[col].isna().any():
                indicator_col = f"{col}_missing"
                df[indicator_col] = df[col].isna().astype(int)
                df[col] = df[col].fillna(sentinel)
        return df

    def add_missingness_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary columns indicating whether each feature was originally missing.
        Useful even after imputation for models that benefit from missingness patterns.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isna().any():
                df[f"{col}_was_missing"] = df[col].isna().astype(int)
        return df

    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary of missing values per column."""
        missing = df.isna().sum()
        pct = (missing / len(df) * 100).round(2)
        summary = pd.DataFrame({
            "missing_count": missing,
            "missing_pct": pct,
        })
        return summary[summary["missing_count"] > 0].sort_values(
            "missing_count", ascending=False
        )
