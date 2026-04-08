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
