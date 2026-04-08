"""
JeevanSync AI — Preprocessing Pipeline
========================================
End-to-end orchestrator that chains:
1. Data loading (CSV or data lake)
2. Schema validation
3. Missing value imputation
4. Time alignment
5. Feature extraction (sliding windows + engineered features)
6. Label generation
7. Train/val/test split (patient-level)
8. Save processed datasets

Usage:
    python -m src.preprocessing.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

import torch

from config.settings import data_settings, feature_settings, model_settings
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.gpu_feature_extractor import GPUFeatureExtractor, get_device
from src.preprocessing.imputation import ImputationEngine
from src.preprocessing.label_generator import LabelGenerator
from src.preprocessing.schema_validator import SchemaValidator
from src.preprocessing.sliding_window import SlidingWindowBuilder
from src.preprocessing.time_alignment import TimeAligner


class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing and feature engineering pipeline.
    Designed for both batch processing (offline training) and
    incremental processing (online serving).
    """

    def __init__(
        self,
        window_size: int = 12,
        prediction_horizons: list[int] | None = None,
        output_dir: Optional[Path] = None,
        use_gpu: bool = True,
    ):
        self.window_size = window_size
        self.prediction_horizons = prediction_horizons or [6, 10, 12]
        self.output_dir = output_dir or data_settings.processed_dir

        # Detect GPU
        self.device = get_device() if use_gpu else torch.device("cpu")
        self.use_gpu = self.device.type == "cuda"

        # Pipeline components
        self.validator = SchemaValidator()
        self.imputer = ImputationEngine()
        self.aligner = TimeAligner()
        self.window_builder = SlidingWindowBuilder(window_size=window_size)
        self.label_generator = LabelGenerator()

        # Feature extractors — GPU-accelerated or CPU fallback
        if self.use_gpu:
            self.feature_extractor = GPUFeatureExtractor(
                window_size=window_size, device=self.device
            )
            logger.info("🔥 Pipeline using GPU-accelerated feature extraction")
        else:
            self.feature_extractor = FeatureExtractor(window_size=window_size)
            logger.info("⚙️  Pipeline using CPU feature extraction")

        # Track pipeline state
        self._pipeline_stats = {"device": str(self.device)}

    # ── Step 1: Data Loading ─────────────────────────────────────────────

    def load_data(
        self,
        source: str = "csv",
        use_prejoined: bool = True,
    ) -> pd.DataFrame:
        """
        Load raw data from CSV files or the data lake.

        Args:
            source: 'csv' or 'datalake'
            use_prejoined: If True, load the pre-joined hourly panel directly

        Returns:
            Raw panel DataFrame
        """
        logger.info("═══ Step 1: Loading Data (source={}) ═══", source)

        if source == "csv":
            if use_prejoined and data_settings.hourly_panel_csv.exists():
                df = pd.read_csv(data_settings.hourly_panel_csv)
                logger.info(
                    "Loaded pre-joined panel: {} rows × {} cols, {} patients",
                    len(df), len(df.columns), df["patient_id"].nunique(),
                )
            else:
                # Load and merge individual CSVs
                from src.preprocessing.time_alignment import load_and_align_from_csvs
                df = load_and_align_from_csvs()
        else:
            raise ValueError(f"Unknown data source: {source}")

        self._pipeline_stats["raw_rows"] = len(df)
        self._pipeline_stats["raw_patients"] = int(df["patient_id"].nunique())
        return df

    # ── Step 2: Schema Validation ────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, data types, and clinical ranges."""
        logger.info("═══ Step 2: Schema Validation ═══")

        # Enforce data types
        df = self.validator.enforce_dtypes(df)

        # Validate ranges and categories
        df, report = self.validator.validate(df, mode="panel")

        # Check patient continuity
        gaps = self.validator.validate_patient_continuity(df)

        self._pipeline_stats["validation_report"] = {
            "total_rows": report.total_rows,
            "valid_rows": report.valid_rows,
            "flagged_rows": report.flagged_rows,
            "is_valid": report.is_valid,
            "patients_with_gaps": len(gaps),
        }

        return df

    # ── Step 3: Imputation ───────────────────────────────────────────────

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using cascade imputation."""
        logger.info("═══ Step 3: Missing Value Imputation ═══")

        missing_before = df.isna().sum().sum()
        self._pipeline_stats["missing_before"] = int(missing_before)

        if missing_before > 0:
            self.imputer.fit(df)
            df = self.imputer.transform(df, strategy="cascade")

        missing_after = df.isna().sum().sum()
        self._pipeline_stats["missing_after"] = int(missing_after)

        return df

    # ── Step 4: Time Alignment ───────────────────────────────────────────

    def align_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure continuous hourly grid and propagate static features."""
        logger.info("═══ Step 4: Time Alignment ═══")

        df = self.aligner.ensure_continuous_grid(df)
        df = self.aligner.propagate_static_features(df)
        df = self.aligner.clip_stay_length(df)

        self._pipeline_stats["aligned_rows"] = len(df)
        return df

    # ── Step 5: Label Generation ─────────────────────────────────────────

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target labels for multiple prediction horizons."""
        logger.info("═══ Step 5: Label Generation ═══")

        df = self.label_generator.generate_labels(df, horizons=self.prediction_horizons)

        # Analyze class distribution for the primary target
        primary_target = f"deterioration_next_{self.prediction_horizons[-1]}h"
        if primary_target in df.columns:
            analysis = self.label_generator.analyze_class_distribution(
                df, label_column=primary_target
            )
            self._pipeline_stats["class_distribution"] = analysis

        return df

    # ── Step 6: Feature Extraction ───────────────────────────────────────

    def extract_features(
        self,
        df: pd.DataFrame,
        mode: str = "engineered",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Extract features from the aligned panel.

        Args:
            df: Aligned panel DataFrame
            mode: 'engineered' (aggregated features) or 'windows' (raw windowed data)

        Returns:
            (features_df, metadata_dict)
        """
        logger.info("═══ Step 6: Feature Extraction (mode={}) ═══", mode)

        if mode == "engineered":
            # Extract aggregated statistical/trend/variability features
            features_df = self.feature_extractor.extract_features_for_dataset(
                df, window_size=self.window_size, show_progress=True
            )

            # Attach static features
            static_cols = [c for c in feature_settings.static_columns if c in df.columns]
            if static_cols:
                static_map = df.groupby("patient_id")[static_cols].first().reset_index()
                features_df = features_df.merge(static_map, on="patient_id", how="left")

            # Encode categoricals
            features_df = self._encode_categoricals(features_df)

            metadata = {
                "mode": "engineered",
                "n_features": len(features_df.columns) - 2,  # exclude pid and hour
                "window_size": self.window_size,
                "feature_columns": [
                    c for c in features_df.columns
                    if c not in ["patient_id", "hour_from_admission"]
                ],
            }

        elif mode == "windows":
            # Build raw sliding windows for sequence models
            window_data = self.window_builder.build_windows(df, show_progress=True)
            features_df = pd.DataFrame({
                "patient_id": window_data["patient_ids"],
                "hour_from_admission": window_data["hours"],
            })

            metadata = {
                "mode": "windows",
                "window_shape": list(window_data["windows"].shape),
                "feature_names": window_data["feature_names"],
                "windows_array": window_data["windows"],  # stored in metadata
            }

        else:
            raise ValueError(f"Unknown feature mode: {mode}")

        self._pipeline_stats["feature_extraction"] = {
            k: v for k, v in metadata.items()
            if not isinstance(v, np.ndarray)
        }

        return features_df, metadata

    # ── Step 7: Train/Val/Test Split ─────────────────────────────────────

    def split_dataset(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
        target_column: str = "deterioration_next_12h",
    ) -> dict[str, pd.DataFrame]:
        """
        Patient-level train/validation/test split.
        Ensures no patient appears in multiple splits (prevents data leakage).
        """
        logger.info("═══ Step 7: Patient-Level Data Split ═══")

        all_patients = df["patient_id"].unique()
        n_patients = len(all_patients)

        # Split patients (not rows)
        train_pids, temp_pids = train_test_split(
            all_patients,
            test_size=(feature_settings.val_ratio + feature_settings.test_ratio),
            random_state=feature_settings.random_seed,
        )
        val_pids, test_pids = train_test_split(
            temp_pids,
            test_size=feature_settings.test_ratio / (feature_settings.val_ratio + feature_settings.test_ratio),
            random_state=feature_settings.random_seed,
        )

        logger.info(
            "Patient split: train={}, val={}, test={}",
            len(train_pids), len(val_pids), len(test_pids),
        )

        # Attach target to features
        target_lookup = df.set_index(["patient_id", "hour_from_admission"])[target_column]
        target_keys = list(
            zip(features_df["patient_id"], features_df["hour_from_admission"])
        )
        features_df = features_df.copy()
        features_df[target_column] = [
            target_lookup.get(k, np.nan) for k in target_keys
        ]

        # Drop rows with missing target
        features_df = features_df.dropna(subset=[target_column])
        features_df[target_column] = features_df[target_column].astype(int)

        # Split into sets
        splits = {
            "train": features_df[features_df["patient_id"].isin(train_pids)].reset_index(drop=True),
            "val": features_df[features_df["patient_id"].isin(val_pids)].reset_index(drop=True),
            "test": features_df[features_df["patient_id"].isin(test_pids)].reset_index(drop=True),
        }

        for name, split_df in splits.items():
            pos = split_df[target_column].sum()
            total = len(split_df)
            logger.info(
                "  {} set: {} rows, {:.2%} positive ({} patients)",
                name, total, pos / max(total, 1), split_df["patient_id"].nunique(),
            )

        self._pipeline_stats["split"] = {
            name: {
                "rows": len(split_df),
                "patients": int(split_df["patient_id"].nunique()),
                "positive_rate": float(split_df[target_column].mean()),
            }
            for name, split_df in splits.items()
        }

        return splits

    # ── Step 8: Save Processed Data ──────────────────────────────────────

    def save_outputs(
        self,
        splits: dict[str, pd.DataFrame],
        metadata: dict,
        windows_array: Optional[np.ndarray] = None,
    ) -> None:
        """Save processed datasets and metadata to disk."""
        logger.info("═══ Step 8: Saving Processed Data ═══")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        for name, split_df in splits.items():
            filepath = self.output_dir / f"{name}_features.parquet"
            split_df.to_parquet(filepath, index=False)
            logger.info("Saved {} → {} ({:.1f} MB)", name, filepath, filepath.stat().st_size / 1e6)

        # Save 3D windows for sequence models (if available)
        if windows_array is not None:
            for name, split_df in splits.items():
                mask = np.isin(
                    metadata.get("patient_ids_array", np.array([])),
                    split_df["patient_id"].unique(),
                )
                if mask.any():
                    np.save(self.output_dir / f"{name}_windows.npy", windows_array[mask])

        # Save pipeline metadata
        stats_path = self.output_dir / "pipeline_stats.json"
        serializable_stats = {
            k: v for k, v in self._pipeline_stats.items()
            if not isinstance(v, (np.ndarray, pd.DataFrame))
        }
        with open(stats_path, "w") as f:
            json.dump(serializable_stats, f, indent=2, default=str)
        logger.info("Pipeline stats saved → {}", stats_path)

        # Save feature column list
        if "feature_columns" in metadata:
            cols_path = self.output_dir / "feature_columns.json"
            with open(cols_path, "w") as f:
                json.dump(metadata["feature_columns"], f, indent=2)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        cat_cols = []
        for col in ["gender", "admission_type", "oxygen_device"]:
            if col in df.columns:
                cat_cols.append(col)

        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, drop_first=False)
            # Convert boolean columns to int
            bool_cols = df.select_dtypes(include=["bool"]).columns
            df[bool_cols] = df[bool_cols].astype(int)
            logger.debug("Encoded categorical columns: {}", cat_cols)

        return df

    # ── Main Pipeline ────────────────────────────────────────────────────

    def run(
        self,
        source: str = "csv",
        use_prejoined: bool = True,
        feature_mode: str = "engineered",
        target_column: str = "deterioration_next_12h",
    ) -> dict[str, pd.DataFrame]:
        """
        Execute the full preprocessing pipeline.

        Args:
            source: Data source ('csv' or 'datalake')
            use_prejoined: Use pre-joined CSV if available
            feature_mode: 'engineered' or 'windows'
            target_column: Target label column

        Returns:
            Dict with 'train', 'val', 'test' DataFrames
        """
        import time as _time
        pipeline_start = _time.time()

        device_label = f"🔥 GPU ({torch.cuda.get_device_name(self.device)})" if self.use_gpu else "⚙️  CPU"
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  G_One_Sync AI — Preprocessing Pipeline              ║")
        logger.info("║  Window: {}h | Device: {}" .format(self.window_size, device_label))
        logger.info("║  Target: {}" .format(target_column))
        logger.info("╚══════════════════════════════════════════════════════╝")

        # Step 1: Load
        df = self.load_data(source=source, use_prejoined=use_prejoined)

        # Step 2: Validate
        df = self.validate(df)

        # Step 3: Impute
        df = self.impute(df)

        # Step 4: Time alignment
        df = self.align_time(df)

        # Step 5: Labels
        df = self.generate_labels(df)

        # Step 6: Feature extraction (GPU-accelerated when available)
        features_df, metadata = self.extract_features(df, mode=feature_mode)

        # Step 7: Split
        splits = self.split_dataset(df, features_df, target_column=target_column)

        # Step 8: Save
        windows_array = metadata.get("windows_array")
        self.save_outputs(splits, metadata, windows_array)

        elapsed = _time.time() - pipeline_start
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  ✅ Pipeline Complete in {:.1f}s                     ║".format(elapsed))
        logger.info("║  Device: {}" .format(device_label))
        logger.info("║  Output: {}".format(self.output_dir))
        if self.use_gpu:
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1e9
            logger.info("║  Peak GPU Memory: {:.2f} GB / 4.0 GB            ║".format(peak_mem))
        logger.info("╚══════════════════════════════════════════════════════╝")

        return splits


# ── CLI Entrypoint ───────────────────────────────────────────────────────

def main():
    """Run the full preprocessing pipeline from CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="JeevanSync Preprocessing Pipeline")
    parser.add_argument("--window-size", type=int, default=12, help="Lookback window (hours)")
    parser.add_argument("--mode", choices=["engineered", "windows"], default="engineered")
    parser.add_argument("--target", default="deterioration_next_12h")
    parser.add_argument("--source", choices=["csv", "datalake"], default="csv")
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(window_size=args.window_size)
    splits = pipeline.run(
        source=args.source,
        feature_mode=args.mode,
        target_column=args.target,
    )

    # Print summary
    for name, split_df in splits.items():
        print(f"  {name}: {len(split_df)} rows, {split_df['patient_id'].nunique()} patients")


if __name__ == "__main__":
    main()
