"""
JeevanSync AI — Feature Extractor
===================================
Extracts three categories of engineered features from sliding windows:
1. Statistical features (mean, std, min, max, median, skew, kurtosis, IQR)
2. Trend features (slope, delta, rate of change, acceleration)
3. Variability features (HRV, BPV, SpO2 variability)
4. Derived clinical scores (MAP, Shock Index, qSOFA proxy)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from loguru import logger
from tqdm import tqdm

from config.settings import feature_settings


class FeatureExtractor:
    """
    Extracts engineered features from the hourly panel data using sliding windows.
    All features are computed per-patient per-window.
    """

    def __init__(
        self,
        window_size: int = 12,
        vital_columns: Optional[list[str]] = None,
        lab_columns: Optional[list[str]] = None,
    ):
        self.window_size = window_size
        self.vital_columns = vital_columns or feature_settings.vital_columns
        self.lab_columns = lab_columns or feature_settings.lab_columns
        self.all_numeric = self.vital_columns + self.lab_columns

    # ── Statistical Features ─────────────────────────────────────────────

    def extract_statistical_features(
        self, window: pd.DataFrame
    ) -> dict[str, float]:
        """
        Extract statistical summary features from a window.

        For each numeric column: mean, std, min, max, median, skew, kurtosis, IQR
        """
        features = {}
        for col in self.all_numeric:
            if col not in window.columns:
                continue

            vals = window[col].dropna()
            if len(vals) == 0:
                for stat in ["mean", "std", "min", "max", "median", "skew", "kurt", "iqr"]:
                    features[f"{col}_{stat}"] = 0.0
                continue

            features[f"{col}_mean"] = float(vals.mean())
            features[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
            features[f"{col}_min"] = float(vals.min())
            features[f"{col}_max"] = float(vals.max())
            features[f"{col}_median"] = float(vals.median())
            features[f"{col}_skew"] = float(vals.skew()) if len(vals) > 2 else 0.0
            features[f"{col}_kurt"] = float(vals.kurtosis()) if len(vals) > 3 else 0.0

            q25 = vals.quantile(0.25)
            q75 = vals.quantile(0.75)
            features[f"{col}_iqr"] = float(q75 - q25)

        # Nurse alert count in window
        if "nurse_alert" in window.columns:
            features["nurse_alert_count"] = int(window["nurse_alert"].sum())
            features["nurse_alert_ratio"] = float(window["nurse_alert"].mean())

        return features

    # ── Trend Features ───────────────────────────────────────────────────

    def extract_trend_features(
        self, window: pd.DataFrame
    ) -> dict[str, float]:
        """
        Extract trend features from a window.

        For each numeric column: linear slope, delta, rate of change, acceleration
        """
        features = {}
        for col in self.all_numeric:
            if col not in window.columns:
                continue

            vals = window[col].dropna().values
            if len(vals) < 2:
                for feat in ["slope", "delta", "roc", "accel"]:
                    features[f"{col}_{feat}"] = 0.0
                continue

            # Linear slope via OLS
            x = np.arange(len(vals), dtype=float)
            try:
                slope, _, _, _, _ = sp_stats.linregress(x, vals)
                features[f"{col}_slope"] = float(slope)
            except Exception:
                features[f"{col}_slope"] = 0.0

            # Delta (last - first)
            delta = float(vals[-1] - vals[0])
            features[f"{col}_delta"] = delta

            # Rate of change (delta / window_length)
            features[f"{col}_roc"] = delta / max(len(vals) - 1, 1)

            # Acceleration (second derivative approximation)
            if len(vals) >= 3:
                diffs = np.diff(vals)
                accel = float(np.mean(np.diff(diffs)))
                features[f"{col}_accel"] = accel
            else:
                features[f"{col}_accel"] = 0.0

        return features

    # ── Variability Features ─────────────────────────────────────────────

    def extract_variability_features(
        self, window: pd.DataFrame
    ) -> dict[str, float]:
        """
        Extract variability-specific features.

        - Heart Rate Variability (SDNN, RMSSD approximations)
        - Blood Pressure Variability (SBP std, coefficient of variation)
        - SpO2 variability (range, std)
        - Temperature variability
        """
        features = {}

        # Heart Rate Variability (from hourly measurements)
        if "heart_rate" in window.columns:
            hr = window["heart_rate"].dropna().values
            if len(hr) > 1:
                features["hrv_sdnn"] = float(np.std(hr, ddof=1))
                nn_diffs = np.abs(np.diff(hr))
                features["hrv_rmssd"] = float(np.sqrt(np.mean(nn_diffs ** 2)))
                features["hrv_range"] = float(hr.max() - hr.min())
                features["hrv_cv"] = float(np.std(hr) / max(np.mean(hr), 1e-8))
            else:
                features.update({
                    "hrv_sdnn": 0.0, "hrv_rmssd": 0.0,
                    "hrv_range": 0.0, "hrv_cv": 0.0,
                })

        # Blood Pressure Variability
        if "systolic_bp" in window.columns:
            sbp = window["systolic_bp"].dropna().values
            if len(sbp) > 1:
                features["bpv_sbp_std"] = float(np.std(sbp, ddof=1))
                features["bpv_sbp_cv"] = float(np.std(sbp) / max(np.mean(sbp), 1e-8))
                features["bpv_sbp_range"] = float(sbp.max() - sbp.min())
            else:
                features.update({
                    "bpv_sbp_std": 0.0, "bpv_sbp_cv": 0.0, "bpv_sbp_range": 0.0,
                })

        if "diastolic_bp" in window.columns:
            dbp = window["diastolic_bp"].dropna().values
            if len(dbp) > 1:
                features["bpv_dbp_std"] = float(np.std(dbp, ddof=1))
                features["bpv_dbp_range"] = float(dbp.max() - dbp.min())
            else:
                features.update({"bpv_dbp_std": 0.0, "bpv_dbp_range": 0.0})

        # SpO2 Variability
        if "spo2_pct" in window.columns:
            spo2 = window["spo2_pct"].dropna().values
            if len(spo2) > 1:
                features["spo2_var_std"] = float(np.std(spo2, ddof=1))
                features["spo2_var_range"] = float(spo2.max() - spo2.min())
                features["spo2_below_90_ratio"] = float(np.mean(spo2 < 90))
            else:
                features.update({
                    "spo2_var_std": 0.0, "spo2_var_range": 0.0,
                    "spo2_below_90_ratio": 0.0,
                })

        # Temperature Variability
        if "temperature_c" in window.columns:
            temp = window["temperature_c"].dropna().values
            if len(temp) > 1:
                features["temp_var_std"] = float(np.std(temp, ddof=1))
                features["temp_var_range"] = float(temp.max() - temp.min())
                features["temp_above_38"] = float(np.mean(temp > 38.0))
            else:
                features.update({
                    "temp_var_std": 0.0, "temp_var_range": 0.0, "temp_above_38": 0.0,
                })

        return features

    # ── Derived Clinical Scores ──────────────────────────────────────────

    def extract_clinical_scores(
        self, window: pd.DataFrame
    ) -> dict[str, float]:
        """
        Compute derived clinical scores from the latest values in the window.

        - Mean Arterial Pressure (MAP)
        - Shock Index (HR / SBP)
        - Modified qSOFA proxy
        - Respiratory distress indicators
        """
        features = {}
        latest = window.iloc[-1] if len(window) > 0 else pd.Series(dtype=float)

        # Mean Arterial Pressure
        if "systolic_bp" in latest.index and "diastolic_bp" in latest.index:
            sbp = latest["systolic_bp"]
            dbp = latest["diastolic_bp"]
            features["map_current"] = float(dbp + (sbp - dbp) / 3)

            # MAP trend over window
            if len(window) > 1:
                maps = window["diastolic_bp"] + (window["systolic_bp"] - window["diastolic_bp"]) / 3
                features["map_mean"] = float(maps.mean())
                features["map_min"] = float(maps.min())
        else:
            features.update({"map_current": 0.0, "map_mean": 0.0, "map_min": 0.0})

        # Shock Index (HR / SBP) — elevated > 0.7 indicates shock
        if "heart_rate" in latest.index and "systolic_bp" in latest.index:
            sbp = max(latest["systolic_bp"], 1e-8)
            features["shock_index_current"] = float(latest["heart_rate"] / sbp)

            if len(window) > 1:
                si_series = window["heart_rate"] / window["systolic_bp"].clip(lower=1)
                features["shock_index_mean"] = float(si_series.mean())
                features["shock_index_max"] = float(si_series.max())
        else:
            features.update({
                "shock_index_current": 0.0, "shock_index_mean": 0.0,
                "shock_index_max": 0.0,
            })

        # Modified qSOFA proxy
        # qSOFA ≥ 2 = sepsis risk: RR ≥ 22, SBP ≤ 100, altered mentation
        qsofa = 0
        if "respiratory_rate" in latest.index and latest["respiratory_rate"] >= 22:
            qsofa += 1
        if "systolic_bp" in latest.index and latest["systolic_bp"] <= 100:
            qsofa += 1
        if "mobility_score" in latest.index and latest["mobility_score"] <= 1:
            qsofa += 1  # proxy for altered mental status
        features["qsofa_score"] = qsofa
        features["qsofa_ge2"] = int(qsofa >= 2)

        # Hours of elevated qSOFA in window
        if len(window) > 0:
            has_rr = "respiratory_rate" in window.columns
            has_sbp = "systolic_bp" in window.columns
            has_ms = "mobility_score" in window.columns

            qsofa_series = pd.Series(0, index=window.index)
            if has_rr:
                qsofa_series += (window["respiratory_rate"] >= 22).astype(int)
            if has_sbp:
                qsofa_series += (window["systolic_bp"] <= 100).astype(int)
            if has_ms:
                qsofa_series += (window["mobility_score"] <= 1).astype(int)

            features["qsofa_ge2_hours"] = int((qsofa_series >= 2).sum())

        # Lactate danger indicator
        if "lactate" in latest.index:
            features["lactate_elevated"] = int(latest["lactate"] > 2.0)
            if len(window) > 0 and "lactate" in window.columns:
                features["lactate_max_window"] = float(window["lactate"].max())

        return features

    # ── Full Feature Extraction for a Window ─────────────────────────────

    def extract_all_features(
        self, window: pd.DataFrame
    ) -> dict[str, float]:
        """
        Extract ALL feature categories from a single window.
        Returns a flat dict of feature_name → value.
        """
        features = {}
        features.update(self.extract_statistical_features(window))
        features.update(self.extract_trend_features(window))
        features.update(self.extract_variability_features(window))
        features.update(self.extract_clinical_scores(window))

        # Add last-known values as point-in-time features
        if len(window) > 0:
            latest = window.iloc[-1]
            for col in self.all_numeric:
                if col in latest.index:
                    features[f"{col}_latest"] = float(latest[col])

        return features

    # ── Batch Feature Extraction ─────────────────────────────────────────

    def extract_features_for_dataset(
        self,
        df: pd.DataFrame,
        window_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Extract engineered features for all patients and all time steps.

        For each (patient_id, hour_from_admission), constructs a lookback window
        and extracts all feature categories.

        Args:
            df: Full panel DataFrame
            window_size: Override window size
            show_progress: Show progress bar

        Returns:
            DataFrame with engineered features + patient_id + hour_from_admission
        """
        ws = window_size or self.window_size
        logger.info(
            "Extracting features: window_size={}, {} patients",
            ws, df["patient_id"].nunique(),
        )

        all_features = []
        patient_groups = df.groupby("patient_id")
        iterator = tqdm(patient_groups, desc="Extracting features") if show_progress else patient_groups

        for pid, group in iterator:
            group = group.sort_values("hour_from_admission").reset_index(drop=True)

            for idx in range(len(group)):
                hour = group.iloc[idx]["hour_from_admission"]

                # Build window: [idx - ws + 1, idx]
                start_idx = max(0, idx - ws + 1)
                window = group.iloc[start_idx : idx + 1]

                # Extract features
                feat_dict = self.extract_all_features(window)
                feat_dict["patient_id"] = pid
                feat_dict["hour_from_admission"] = hour

                all_features.append(feat_dict)

        result = pd.DataFrame(all_features)
        logger.info(
            "Feature extraction complete: {} rows × {} features",
            len(result), len(result.columns) - 2,  # exclude pid and hour
        )

        return result

    def get_feature_names(self) -> list[str]:
        """Return a list of all generated feature names (without identifiers)."""
        # Generate from a dummy window to get all feature names
        dummy_data = {col: [70.0, 72.0, 71.0] for col in self.all_numeric}
        dummy_data["nurse_alert"] = [0, 1, 0]
        dummy_data["mobility_score"] = [3, 2, 2]
        dummy_df = pd.DataFrame(dummy_data)

        features = self.extract_all_features(dummy_df)
        return sorted(features.keys())
