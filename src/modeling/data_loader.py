"""
G_One_Sync AI — PyTorch Data Loaders
======================================
GPU-optimized data loading for sequence models (BiLSTM, Transformer).
Handles both windowed 3D sequences and tabular features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from config.settings import model_settings, data_settings


class ICUTabularDataset(Dataset):
    """Dataset for tabular models (XGBoost). Returns (features, label) tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "deterioration_next_12h",
        feature_columns: Optional[list[str]] = None,
    ):
        exclude = {"patient_id", "hour_from_admission", target_column}
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c not in exclude]

        self.features = torch.tensor(df[feature_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(df[target_column].values, dtype=torch.float32)
        self.feature_names = feature_columns
        self.patient_ids = df["patient_id"].values if "patient_id" in df.columns else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ICUSequenceDataset(Dataset):
    """Dataset for sequence models (BiLSTM, Transformer). Each sample is (W, F) + label."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, patient_ids: Optional[np.ndarray] = None):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def create_tabular_dataloaders(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    target_column: str = "deterioration_next_12h", batch_size: int = 64,
) -> dict[str, DataLoader]:
    """Create DataLoaders for tabular datasets."""
    train_ds = ICUTabularDataset(train_df, target_column)
    val_ds = ICUTabularDataset(val_df, target_column, feature_columns=train_ds.feature_names)
    test_ds = ICUTabularDataset(test_df, target_column, feature_columns=train_ds.feature_names)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
    }
    logger.info("Tabular loaders: train={}, val={}, test={} ({} features)",
                len(train_ds), len(val_ds), len(test_ds), len(train_ds.feature_names))
    return loaders


def create_sequence_dataloaders(
    train_windows: np.ndarray, train_labels: np.ndarray,
    val_windows: np.ndarray, val_labels: np.ndarray,
    test_windows: np.ndarray, test_labels: np.ndarray,
    batch_size: int = 64,
) -> dict[str, DataLoader]:
    """Create DataLoaders for sequence datasets."""
    train_ds = ICUSequenceDataset(train_windows, train_labels)
    val_ds = ICUSequenceDataset(val_windows, val_labels)
    test_ds = ICUSequenceDataset(test_windows, test_labels)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
    }
    logger.info("Sequence loaders: train={}, val={}, test={} (window={}x{})",
                len(train_ds), len(val_ds), len(test_ds),
                train_windows.shape[1], train_windows.shape[2])
    return loaders


def load_processed_splits(
    processed_dir: Optional[Path] = None,
    target_column: str = "deterioration_next_12h",
) -> dict[str, pd.DataFrame]:
    """Load pre-processed train/val/test parquet splits from disk."""
    processed_dir = processed_dir or data_settings.processed_dir
    splits = {}
    for name in ["train", "val", "test"]:
        path = processed_dir / f"{name}_features.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)
            pos_rate = splits[name][target_column].mean() if target_column in splits[name].columns else 0
            logger.info("Loaded {}: {} rows, {:.1%} positive", name, len(splits[name]), pos_rate)
        else:
            logger.warning("Missing split file: {}", path)
    return splits
