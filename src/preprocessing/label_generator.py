"""
JeevanSync AI — Label Generator
=================================
Generates target labels for clinical deterioration prediction.
Supports multiple prediction horizons (6h, 10h, 12h).
Provides class distribution analysis and rebalancing strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class LabelGenerator:
    """
    Creates binary deterioration labels for different prediction horizons.
    Uses the raw deterioration_hour from patient data.
    """

    def generate_labels(
        self,
        df: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Generate deterioration labels for multiple prediction horizons.

        For a row at (patient_id, hour=t) with deterioration_hour=h:
        - label = 1 if t < h <= t + horizon
        - label = 0 otherwise (including no event)

        Args:
            df: Panel DataFrame with 'patient_id', 'hour_from_admission',
                'deterioration_hour' columns
            horizons: List of prediction horizons in hours (default: [6, 10, 12])

        Returns:
            DataFrame with additional label columns
        """
        if horizons is None:
            horizons = [6, 10, 12]

        result = df.copy()

        for h in horizons:
            col_name = f"deterioration_next_{h}h"

            if "deterioration_hour" in result.columns:
                result[col_name] = (
                    (result["deterioration_hour"] > result["hour_from_admission"]) &
                    (result["deterioration_hour"] <= result["hour_from_admission"] + h)
                ).astype(int)

                # Patients with no deterioration event (deterioration_hour == -1) always get 0
                result.loc[result["deterioration_hour"] == -1, col_name] = 0

                pos_count = result[col_name].sum()
                total = len(result)
                logger.info(
                    "Label '{}': {} positive ({:.2%}) / {} negative ({:.2%})",
                    col_name, pos_count, pos_count / total,
                    total - pos_count, 1 - pos_count / total,
                )
            elif col_name in result.columns:
                logger.info(
                    "Label '{}' already exists. Positive rate: {:.2%}",
                    col_name, result[col_name].mean(),
                )
            else:
                logger.warning(
                    "Cannot generate '{}': missing 'deterioration_hour' column", col_name
                )

        return result

    def generate_sepsis_labels(
        self,
        df: pd.DataFrame,
        horizon_hours: int = 6,
        col_name: str = "sepsis_next_6h",
    ) -> pd.DataFrame:
        """
        Generate binary sepsis labels for prediction within horizon_hours.

        Clinical criteria (Sepsis-3 surrogate):
        - Sepsis risk score >= 0.6
        - OR qSOFA surrogate: RR >= 22 & SBP <= 100 with elevated lactate/WBC/temp
        - OR deterioration event with elevated sepsis risk score >= 0.4
        """
        result = df.copy()
        if col_name in result.columns:
            logger.info("Label '{}' already exists. Positive rate: {:.2%}", col_name, result[col_name].mean())
            return result

        # Compute instantaneous sepsis criteria event mask
        event_mask = pd.Series(False, index=result.index)
        if "sepsis_risk_score" in result.columns:
            event_mask = event_mask | (result["sepsis_risk_score"] >= 0.6)

        if all(c in result.columns for c in ["respiratory_rate", "systolic_bp"]):
            qsofa = (result["respiratory_rate"] >= 22) & (result["systolic_bp"] <= 100)
            org_dysfunction = pd.Series(False, index=result.index)
            if "lactate" in result.columns:
                org_dysfunction = org_dysfunction | (result["lactate"] >= 2.0)
            if "wbc_count" in result.columns:
                org_dysfunction = org_dysfunction | (result["wbc_count"] >= 12.0)
            if "temperature_c" in result.columns:
                org_dysfunction = org_dysfunction | (result["temperature_c"] >= 38.3) | (result["temperature_c"] <= 36.0)
            event_mask = event_mask | (qsofa & org_dysfunction)

        # Map to prediction horizon per patient if timeseries panel
        if "patient_id" in result.columns and "hour_from_admission" in result.columns:
            result[col_name] = 0
            for pid, group in result.groupby("patient_id"):
                event_hours = group.loc[event_mask.loc[group.index], "hour_from_admission"].values
                if len(event_hours) == 0:
                    continue
                for idx, row in group.iterrows():
                    t = row["hour_from_admission"]
                    # Positive if an event occurs in (t, t + horizon_hours]
                    if any((event_hours > t) & (event_hours <= t + horizon_hours)):
                        result.loc[idx, col_name] = 1
        else:
            result[col_name] = event_mask.astype(int)

        pos_count = int(result[col_name].sum())
        total = len(result)
        logger.info(
            "Label '{}': {} positive ({:.2%}) / {} negative ({:.2%})",
            col_name, pos_count, pos_count / max(total, 1),
            total - pos_count, (total - pos_count) / max(total, 1),
        )
        return result

    def generate_cardiac_arrest_labels(
        self,
        df: pd.DataFrame,
        horizon_hours: int = 12,
        col_name: str = "cardiac_arrest_next_12h",
    ) -> pd.DataFrame:
        """
        Generate binary cardiac arrest labels for prediction within horizon_hours.

        Clinical criteria:
        - Severe hemodynamic collapse (HR < 35 or HR > 180 or SBP < 65)
        - OR Deterioration with extreme arrhythmia (HR < 40 or HR > 150 or SBP < 70)
        """
        result = df.copy()
        if col_name in result.columns:
            logger.info("Label '{}' already exists. Positive rate: {:.2%}", col_name, result[col_name].mean())
            return result

        event_mask = pd.Series(False, index=result.index)
        if "heart_rate" in result.columns and "systolic_bp" in result.columns:
            extreme_vitals = (
                (result["heart_rate"] < 35) |
                (result["heart_rate"] > 180) |
                (result["systolic_bp"] < 65)
            )
            event_mask = event_mask | extreme_vitals

            if "deterioration_event" in result.columns or "deterioration_hour" in result.columns:
                is_det = (
                    (result.get("deterioration_event", pd.Series(0, index=result.index)) == 1) |
                    (result.get("deterioration_hour", pd.Series(-1, index=result.index)) > 0)
                )
                arrhythmia = (
                    (result["heart_rate"] < 40) |
                    (result["heart_rate"] > 150) |
                    (result["systolic_bp"] < 70)
                )
                event_mask = event_mask | (is_det & arrhythmia)

        if "patient_id" in result.columns and "hour_from_admission" in result.columns:
            result[col_name] = 0
            for pid, group in result.groupby("patient_id"):
                event_hours = group.loc[event_mask.loc[group.index], "hour_from_admission"].values
                if len(event_hours) == 0:
                    continue
                for idx, row in group.iterrows():
                    t = row["hour_from_admission"]
                    if any((event_hours > t) & (event_hours <= t + horizon_hours)):
                        result.loc[idx, col_name] = 1
        else:
            result[col_name] = event_mask.astype(int)

        pos_count = int(result[col_name].sum())
        total = len(result)
        logger.info(
            "Label '{}': {} positive ({:.2%}) / {} negative ({:.2%})",
            col_name, pos_count, pos_count / max(total, 1),
            total - pos_count, (total - pos_count) / max(total, 1),
        )
        return result

    def generate_respiratory_failure_labels(
        self,
        df: pd.DataFrame,
        horizon_hours: int = 12,
        col_name: str = "respiratory_failure_next_12h",
    ) -> pd.DataFrame:
        """
        Generate binary respiratory failure labels for prediction within horizon_hours.

        Clinical criteria:
        - Severe hypoxemia (SpO2 < 88%)
        - SpO2 < 90% with tachypnea (RR >= 28 breaths/min)
        - High oxygen flow (>= 6 L/min) with tachypnea (RR >= 26)
        """
        result = df.copy()
        if col_name in result.columns:
            logger.info("Label '{}' already exists. Positive rate: {:.2%}", col_name, result[col_name].mean())
            return result

        event_mask = pd.Series(False, index=result.index)
        if "spo2_pct" in result.columns:
            event_mask = event_mask | (result["spo2_pct"] < 88.0)

        if "spo2_pct" in result.columns and "respiratory_rate" in result.columns:
            event_mask = event_mask | ((result["spo2_pct"] < 90.0) & (result["respiratory_rate"] >= 28.0))

        if "oxygen_flow" in result.columns and "respiratory_rate" in result.columns:
            event_mask = event_mask | ((result["oxygen_flow"] >= 6.0) & (result["respiratory_rate"] >= 26.0))

        if "patient_id" in result.columns and "hour_from_admission" in result.columns:
            result[col_name] = 0
            for pid, group in result.groupby("patient_id"):
                event_hours = group.loc[event_mask.loc[group.index], "hour_from_admission"].values
                if len(event_hours) == 0:
                    continue
                for idx, row in group.iterrows():
                    t = row["hour_from_admission"]
                    if any((event_hours > t) & (event_hours <= t + horizon_hours)):
                        result.loc[idx, col_name] = 1
        else:
            result[col_name] = event_mask.astype(int)

        pos_count = int(result[col_name].sum())
        total = len(result)
        logger.info(
            "Label '{}': {} positive ({:.2%}) / {} negative ({:.2%})",
            col_name, pos_count, pos_count / max(total, 1),
            total - pos_count, (total - pos_count) / max(total, 1),
        )
        return result

    def generate_all_outcome_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for all 4 clinical outcomes:
        - deterioration_next_12h (primary deterioration)
        - sepsis_next_6h (sepsis onset)
        - cardiac_arrest_next_12h (cardiac arrest)
        - respiratory_failure_next_12h (acute respiratory failure)
        """
        df = self.generate_labels(df, horizons=[12])
        df = self.generate_sepsis_labels(df, horizon_hours=6)
        df = self.generate_cardiac_arrest_labels(df, horizon_hours=12)
        df = self.generate_respiratory_failure_labels(df, horizon_hours=12)
        return df

    def analyze_class_distribution(
        self,
        df: pd.DataFrame,
        label_column: str = "deterioration_next_12h",
    ) -> dict:
        """
        Analyze class distribution and recommend rebalancing strategy.

        Returns:
            Dict with distribution stats and recommendations
        """
        if label_column not in df.columns:
            logger.warning("Label column '{}' not found", label_column)
            return {}

        labels = df[label_column]
        total = len(labels)
        positive = int(labels.sum())
        negative = total - positive
        imbalance_ratio = negative / max(positive, 1)

        analysis = {
            "total_samples": total,
            "positive_samples": positive,
            "negative_samples": negative,
            "positive_rate": positive / total,
            "negative_rate": negative / total,
            "imbalance_ratio": imbalance_ratio,
        }

        # Recommend strategy
        if imbalance_ratio > 10:
            analysis["recommendation"] = "severe_imbalance"
            analysis["suggested_strategies"] = [
                "SMOTE oversampling",
                "Class-weighted loss function",
                "Focal loss",
                "Undersampling majority class",
            ]
            analysis["suggested_scale_pos_weight"] = imbalance_ratio
        elif imbalance_ratio > 3:
            analysis["recommendation"] = "moderate_imbalance"
            analysis["suggested_strategies"] = [
                "Class-weighted loss function",
                "scale_pos_weight in XGBoost",
            ]
            analysis["suggested_scale_pos_weight"] = imbalance_ratio
        else:
            analysis["recommendation"] = "acceptable_balance"
            analysis["suggested_strategies"] = ["No rebalancing needed"]

        # Per-patient distribution
        if "patient_id" in df.columns:
            patient_labels = df.groupby("patient_id")[label_column].max()
            analysis["patients_with_event"] = int(patient_labels.sum())
            analysis["patients_without_event"] = int((patient_labels == 0).sum())
            analysis["patient_event_rate"] = float(patient_labels.mean())

        logger.info(
            "Class distribution for '{}': {:.1%} positive, ratio={:.1f}:1, strategy={}",
            label_column,
            analysis["positive_rate"],
            imbalance_ratio,
            analysis["recommendation"],
        )

        return analysis

    def generate_sample_weights(
        self,
        labels: pd.Series,
        strategy: str = "balanced",
    ) -> np.ndarray:
        """
        Generate per-sample weights for imbalanced classification.

        Args:
            labels: Binary label series
            strategy: 'balanced' (inversely proportional to class frequency)
                      or 'sqrt_balanced' (square root of balanced weights)

        Returns:
            Array of sample weights
        """
        n_total = len(labels)
        n_pos = labels.sum()
        n_neg = n_total - n_pos
        n_classes = 2

        if strategy == "balanced":
            w_pos = n_total / (n_classes * max(n_pos, 1))
            w_neg = n_total / (n_classes * max(n_neg, 1))
        elif strategy == "sqrt_balanced":
            w_pos = np.sqrt(n_total / (n_classes * max(n_pos, 1)))
            w_neg = np.sqrt(n_total / (n_classes * max(n_neg, 1)))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        weights = np.where(labels == 1, w_pos, w_neg)
        logger.info(
            "Sample weights: positive={:.3f}, negative={:.3f}", w_pos, w_neg
        )
        return weights
