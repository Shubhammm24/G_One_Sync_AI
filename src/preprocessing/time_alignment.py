"""
JeevanSync AI — Time Alignment & Resampling
=============================================
Aligns vitals and lab data to a uniform hourly grid per patient.
Handles merging, resampling, and padding for ICU time-series.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import data_settings


class TimeAligner:
    """
    Ensures all patient time-series are on a continuous hourly grid.
    Merges vitals and labs, handles irregular timestamps, and pads sequences.
    """

    def merge_vitals_and_labs(
        self,
        vitals_df: pd.DataFrame,
        labs_df: pd.DataFrame,
        patients_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge vitals, labs, and (optionally) patient demographics into
        a single hourly panel.

        Args:
            vitals_df: Hourly vitals (patient_id, hour_from_admission, ...)
            labs_df: Hourly labs (patient_id, hour_from_admission, ...)
            patients_df: Static patient data (patient_id, age, gender, ...)

        Returns:
            Merged DataFrame on (patient_id, hour_from_admission)
        """
        logger.info(
            "Merging vitals ({} rows) + labs ({} rows)",
            len(vitals_df), len(labs_df),
        )

        # Merge vitals and labs on shared keys
        merged = pd.merge(
            vitals_df,
            labs_df,
            on=["patient_id", "hour_from_admission"],
            how="outer",
            suffixes=("", "_lab"),
        )

        # Merge with static patient demographics
        if patients_df is not None:
            # Avoid duplicating columns that might exist in the panel
            demo_cols = [c for c in patients_df.columns if c not in merged.columns or c == "patient_id"]
            merged = pd.merge(
                merged,
                patients_df[demo_cols],
                on="patient_id",
                how="left",
            )
            logger.info("Merged with patient demographics ({} patients)", len(patients_df))

        merged = merged.sort_values(
            ["patient_id", "hour_from_admission"]
        ).reset_index(drop=True)

        logger.info(
            "Merged panel: {} rows × {} columns for {} patients",
            len(merged), len(merged.columns), merged["patient_id"].nunique(),
        )
        return merged

    def ensure_continuous_grid(
        self,
        df: pd.DataFrame,
        max_hours: int = 72,
    ) -> pd.DataFrame:
        """
        Ensure every patient has continuous hourly rows from hour 0 to their
        last recorded hour.

        Missing intermediate hours are filled with NaN (to be imputed later).
        """
        logger.info("Ensuring continuous hourly grid...")

        records = []
        for pid, group in df.groupby("patient_id"):
            min_h = int(group["hour_from_admission"].min())
            max_h = int(group["hour_from_admission"].max())
            max_h = min(max_h, max_hours - 1)

            full_grid = pd.DataFrame({
                "patient_id": pid,
                "hour_from_admission": range(min_h, max_h + 1),
            })

            # Merge with existing data to identify and fill gaps
            merged = pd.merge(
                full_grid,
                group,
                on=["patient_id", "hour_from_admission"],
                how="left",
            )
            records.append(merged)

        result = pd.concat(records, ignore_index=True)
        result = result.sort_values(
            ["patient_id", "hour_from_admission"]
        ).reset_index(drop=True)

        gaps_filled = len(result) - len(df)
        if gaps_filled > 0:
            logger.info("Filled {} gap rows across all patients", gaps_filled)
        else:
            logger.info("No gaps detected — all patients have continuous data")

        return result

    def propagate_static_features(
        self,
        df: pd.DataFrame,
        static_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Forward-fill static features (age, gender, etc.) within each patient.
        These are constant per patient but may be NaN after grid expansion.
        """
        if static_cols is None:
            static_cols = ["age", "gender", "comorbidity_index", "admission_type",
                          "baseline_risk_score", "los_hours",
                          "deterioration_event", "deterioration_within_12h_from_admission",
                          "deterioration_hour"]

        available_static = [c for c in static_cols if c in df.columns]
        if not available_static:
            return df

        df = df.copy()
        for col in available_static:
            df[col] = df.groupby("patient_id")[col].transform(
                lambda x: x.ffill().bfill()
            )

        logger.debug("Propagated static features: {}", available_static)
        return df

    def resample_to_frequency(
        self,
        df: pd.DataFrame,
        target_freq_hours: int = 1,
    ) -> pd.DataFrame:
        """
        Resample time-series to a target frequency (e.g., 1-hour intervals).
        Useful if source data arrives at sub-hourly or irregular intervals.

        For the current dataset (already hourly), this is mostly a no-op
        but handles edge cases.
        """
        if target_freq_hours == 1:
            # Already at hourly — just ensure integer hours
            df = df.copy()
            df["hour_from_admission"] = df["hour_from_admission"].astype(int)
            return df

        logger.info("Resampling to {}-hour intervals", target_freq_hours)

        records = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["patient_id", "hour_from_admission"]]

        for pid, group in df.groupby("patient_id"):
            group = group.sort_values("hour_from_admission")
            max_h = int(group["hour_from_admission"].max())

            for start_h in range(0, max_h + 1, target_freq_hours):
                end_h = start_h + target_freq_hours
                window = group[
                    (group["hour_from_admission"] >= start_h) &
                    (group["hour_from_admission"] < end_h)
                ]

                if window.empty:
                    continue

                # Aggregate: mean for numerics, last for categoricals
                row = {"patient_id": pid, "hour_from_admission": start_h}
                for col in numeric_cols:
                    row[col] = window[col].mean()

                for col in ["gender", "admission_type", "oxygen_device"]:
                    if col in window.columns:
                        row[col] = window[col].iloc[-1]

                records.append(row)

        result = pd.DataFrame(records)
        logger.info("Resampled: {} → {} rows", len(df), len(result))
        return result

    def clip_stay_length(
        self, df: pd.DataFrame, max_hours: int = 72
    ) -> pd.DataFrame:
        """Clip patient stays to a maximum number of hours."""
        before = len(df)
        result = df[df["hour_from_admission"] < max_hours].copy()
        clipped = before - len(result)
        if clipped > 0:
            logger.info("Clipped {} rows exceeding {} hours", clipped, max_hours)
        return result


def load_and_align_from_csvs() -> pd.DataFrame:
    """
    Convenience function: Load all CSV files and produce a fully aligned panel.
    """
    aligner = TimeAligner()

    # Load individual CSVs
    vitals = pd.read_csv(data_settings.vitals_csv)
    labs = pd.read_csv(data_settings.labs_csv)
    patients = pd.read_csv(data_settings.patients_csv)

    # Merge
    panel = aligner.merge_vitals_and_labs(vitals, labs, patients)

    # Ensure continuous hourly grid
    panel = aligner.ensure_continuous_grid(panel)

    # Propagate static features
    panel = aligner.propagate_static_features(panel)

    # Clip to 72 hours max
    panel = aligner.clip_stay_length(panel)

    return panel
