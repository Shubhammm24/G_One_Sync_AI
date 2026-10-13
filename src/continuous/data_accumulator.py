"""
G_One_Sync AI — Live Data Accumulator
=========================================
Accumulates live prediction data into a retraining buffer.
When enough samples are collected, they are available for
the retraining orchestrator to merge with historical data.

Data is stored as Parquet files in the retraining buffer
directory, partitioned by date for efficient management.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import continuous_settings


class DataAccumulator:
    """
    Accumulates live ICU data for model retraining.

    Collects feature vectors and outcomes from live predictions,
    stores them in Parquet format, and tracks buffer statistics.

    The accumulator does NOT trigger retraining — that is the
    orchestrator's responsibility. It only manages the data buffer.
    """

    def __init__(
        self,
        buffer_dir: Path | None = None,
        min_samples: int | None = None,
    ):
        self.buffer_dir = buffer_dir or continuous_settings.retraining_buffer_dir
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        self.min_samples = min_samples or continuous_settings.min_retrain_samples

        # In-memory buffer (flushed to disk periodically)
        self._memory_buffer: list[dict] = []
        self._flush_threshold: int = 500  # Flush every 500 samples

        # Statistics
        self._total_accumulated: int = 0
        self._last_flush_time: float = time.time()

        logger.info(
            "DataAccumulator initialized: buffer_dir={} | min_samples={}",
            self.buffer_dir,
            self.min_samples,
        )

    def add_sample(
        self,
        features: dict[str, float],
        prediction: float,
        actual_outcome: int | None = None,
        patient_id: int | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Add a single sample to the retraining buffer.

        Args:
            features: Feature dict (column_name → value).
            prediction: Model's predicted probability.
            actual_outcome: True label (0 or 1), if available.
            patient_id: Patient identifier.
            timestamp: Sample timestamp (defaults to now).
        """
        sample = {
            **features,
            "_prediction": prediction,
            "_actual_outcome": actual_outcome,
            "_patient_id": patient_id,
            "_timestamp": timestamp or time.time(),
            "_ingested_at": time.time(),
        }
        self._memory_buffer.append(sample)
        self._total_accumulated += 1

        if len(self._memory_buffer) >= self._flush_threshold:
            self.flush()

    def add_batch(
        self,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray | None = None,
    ) -> None:
        """Add a batch of samples to the retraining buffer.

        Args:
            features_df: DataFrame of features.
            predictions: Array of predicted probabilities.
            actuals: Array of true labels (optional).
        """
        for i in range(len(features_df)):
            row = features_df.iloc[i].to_dict()
            actual = int(actuals[i]) if actuals is not None else None
            self.add_sample(
                features=row,
                prediction=float(predictions[i]),
                actual_outcome=actual,
            )

    def flush(self) -> Path | None:
        """Flush the in-memory buffer to a Parquet file on disk.

        Returns:
            Path to the created Parquet file, or None if buffer was empty.
        """
        if not self._memory_buffer:
            return None

        df = pd.DataFrame(self._memory_buffer)

        # Partition by date
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        partition_dir = self.buffer_dir / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename with timestamp
        ts = int(time.time() * 1000)
        file_path = partition_dir / f"buffer_{ts}.parquet"
        df.to_parquet(file_path, index=False)

        count = len(self._memory_buffer)
        self._memory_buffer.clear()
        self._last_flush_time = time.time()

        logger.info(
            "Flushed {} samples to {} | total accumulated: {}",
            count,
            file_path,
            self._total_accumulated,
        )

        return file_path

    def get_buffer_data(self) -> pd.DataFrame:
        """Load all accumulated data from the buffer directory.

        Returns:
            Combined DataFrame of all buffered samples.
        """
        # Flush any in-memory data first
        self.flush()

        parquet_files = list(self.buffer_dir.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files]
        combined = pd.concat(dfs, ignore_index=True)

        logger.info(
            "Loaded buffer data: {} samples from {} files",
            len(combined),
            len(parquet_files),
        )

        return combined

    def clear_buffer(self) -> int:
        """Clear all data from the retraining buffer.

        Returns:
            Number of files removed.
        """
        import shutil

        self._memory_buffer.clear()
        parquet_files = list(self.buffer_dir.rglob("*.parquet"))
        count = len(parquet_files)

        for f in parquet_files:
            f.unlink()

        # Remove empty partition directories
        for d in self.buffer_dir.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

        logger.info("Cleared retraining buffer: {} files removed", count)
        return count

    @property
    def buffer_size(self) -> int:
        """Total number of samples in the buffer (disk + memory)."""
        disk_count = 0
        for f in self.buffer_dir.rglob("*.parquet"):
            try:
                df = pd.read_parquet(f, columns=["_timestamp"])
                disk_count += len(df)
            except Exception:
                pass
        return disk_count + len(self._memory_buffer)

    @property
    def has_enough_samples(self) -> bool:
        """Check if the buffer has enough samples for retraining."""
        return self.buffer_size >= self.min_samples

    def get_stats(self) -> dict:
        """Get buffer statistics for monitoring."""
        return {
            "total_accumulated": self._total_accumulated,
            "memory_buffer_size": len(self._memory_buffer),
            "disk_buffer_size": self.buffer_size - len(self._memory_buffer),
            "min_samples_required": self.min_samples,
            "has_enough_samples": self.has_enough_samples,
            "buffer_dir": str(self.buffer_dir),
            "last_flush_time": self._last_flush_time,
        }
