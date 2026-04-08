"""
JeevanSync AI — Sliding Window Mechanism
==========================================
Constructs lookback windows (6–12 hours) from ICU time-series data.
Produces arrays suitable for both tabular models (flattened) and
sequence models (3D tensors).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.settings import feature_settings


class SlidingWindowBuilder:
    """
    For each observation at hour t, gathers features from [t - window_size, t].
    Handles edge cases at the start of a stay by padding with earliest values.
    """

    def __init__(
        self,
        window_size: int = 12,
        feature_columns: Optional[list[str]] = None,
        step_size: int = 1,
    ):
        """
        Args:
            window_size: Number of past hours to look back (inclusive of current)
            feature_columns: Columns to include in the window
            step_size: Step between consecutive windows (1 = every hour)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.feature_columns = feature_columns or feature_settings.numeric_columns

    def build_windows_for_patient(
        self,
        patient_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sliding windows for a single patient.

        Args:
            patient_df: DataFrame for one patient, sorted by hour_from_admission

        Returns:
            Tuple of:
            - windows: np.ndarray of shape (n_windows, window_size, n_features)
            - hours: np.ndarray of shape (n_windows,) — the target hour for each window
            - patient_ids: np.ndarray of shape (n_windows,) — repeated patient_id
        """
        patient_df = patient_df.sort_values("hour_from_admission").reset_index(drop=True)
        pid = patient_df["patient_id"].iloc[0]

        # Extract feature matrix
        available_cols = [c for c in self.feature_columns if c in patient_df.columns]
        feature_matrix = patient_df[available_cols].values  # (T, F)
        hours = patient_df["hour_from_admission"].values    # (T,)

        n_timesteps = len(patient_df)
        windows = []
        target_hours = []

        for t in range(0, n_timesteps, self.step_size):
            # Window spans [t - window_size + 1, t] (inclusive)
            start_idx = max(0, t - self.window_size + 1)
            window = feature_matrix[start_idx : t + 1]  # shape: (actual_len, F)

            # Pad with earliest values if window is shorter than window_size
            if len(window) < self.window_size:
                pad_length = self.window_size - len(window)
                pad = np.tile(window[0], (pad_length, 1))  # repeat first row
                window = np.vstack([pad, window])

            windows.append(window)
            target_hours.append(hours[t])

        windows_array = np.array(windows)                    # (N, W, F)
        hours_array = np.array(target_hours)                 # (N,)
        pids_array = np.full(len(windows), pid)              # (N,)

        return windows_array, hours_array, pids_array

    def build_windows(
        self,
        df: pd.DataFrame,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Build sliding windows for all patients.

        Args:
            df: Full panel DataFrame
            show_progress: Show tqdm progress bar

        Returns:
            Dict with keys:
            - 'windows': (N_total, window_size, n_features)
            - 'hours': (N_total,)
            - 'patient_ids': (N_total,)
            - 'feature_names': list of feature column names
        """
        available_cols = [c for c in self.feature_columns if c in df.columns]
        logger.info(
            "Building sliding windows: window_size={}, step={}, features={}",
            self.window_size, self.step_size, len(available_cols),
        )

        all_windows = []
        all_hours = []
        all_pids = []

        patient_groups = df.groupby("patient_id")
        iterator = tqdm(patient_groups, desc="Building windows") if show_progress else patient_groups

        for pid, group in iterator:
            windows, hours, pids = self.build_windows_for_patient(group)
            all_windows.append(windows)
            all_hours.append(hours)
            all_pids.append(pids)

        result = {
            "windows": np.concatenate(all_windows, axis=0),
            "hours": np.concatenate(all_hours, axis=0),
            "patient_ids": np.concatenate(all_pids, axis=0),
            "feature_names": available_cols,
        }

        logger.info(
            "Windows built: {} total windows of shape ({}, {})",
            result["windows"].shape[0],
            self.window_size,
            len(available_cols),
        )

        return result

    def flatten_windows(self, windows: np.ndarray) -> np.ndarray:
        """
        Flatten 3D windows to 2D for tabular models (XGBoost, LightGBM).
        Shape: (N, window_size, n_features) → (N, window_size * n_features)
        """
        n_samples = windows.shape[0]
        flattened = windows.reshape(n_samples, -1)
        logger.debug("Flattened windows: {} → {}", windows.shape, flattened.shape)
        return flattened

    def get_flattened_feature_names(self, feature_names: list[str]) -> list[str]:
        """
        Generate column names for flattened windows.
        Format: {feature_name}_t-{offset}
        """
        names = []
        for t in range(self.window_size):
            offset = self.window_size - 1 - t
            suffix = f"t-{offset}" if offset > 0 else "t0"
            for feat in feature_names:
                names.append(f"{feat}_{suffix}")
        return names

    def build_tabular_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = "deterioration_next_12h",
        include_static: bool = True,
        show_progress: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build a complete tabular dataset with flattened windows + static features.
        Ready for XGBoost/LightGBM training.

        Returns:
            (features_df, target_series)
        """
        # Build windows
        window_data = self.build_windows(df, show_progress=show_progress)

        # Flatten for tabular model
        flattened = self.flatten_windows(window_data["windows"])
        col_names = self.get_flattened_feature_names(window_data["feature_names"])

        features_df = pd.DataFrame(flattened, columns=col_names)
        features_df["patient_id"] = window_data["patient_ids"]
        features_df["hour_from_admission"] = window_data["hours"]

        # Add static features
        if include_static:
            static_cols = [c for c in feature_settings.static_columns if c in df.columns]
            if static_cols:
                static_df = df.groupby("patient_id")[static_cols].first().reset_index()
                features_df = features_df.merge(static_df, on="patient_id", how="left")

        # Extract target
        target_lookup = df.set_index(["patient_id", "hour_from_admission"])[target_column]
        target_keys = list(
            zip(features_df["patient_id"], features_df["hour_from_admission"])
        )
        target = pd.Series(
            [target_lookup.get(k, np.nan) for k in target_keys],
            name=target_column,
        )

        # Drop rows where target is NaN
        valid_mask = target.notna()
        features_df = features_df[valid_mask].reset_index(drop=True)
        target = target[valid_mask].astype(int).reset_index(drop=True)

        logger.info(
            "Tabular dataset: {} samples × {} features, target balance: {:.1%} positive",
            len(features_df),
            len(features_df.columns),
            target.mean(),
        )

        return features_df, target
