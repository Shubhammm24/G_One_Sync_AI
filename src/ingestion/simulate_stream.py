"""
JeevanSync AI — ICU Stream Simulator
======================================
Reads the existing CSV dataset and replays it through the Ingest API,
simulating real-time or accelerated ICU data streams.

Usage:
    python -m src.ingestion.simulate_stream --rate 100 --patients 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.settings import api_settings, data_settings


def load_panel_data(
    csv_path: Path,
    n_patients: int | None = None,
    patient_ids: list[int] | None = None,
) -> pd.DataFrame:
    """
    Load the hourly panel CSV and optionally filter to a subset of patients.

    Args:
        csv_path: Path to hospital_deterioration_hourly_panel.csv
        n_patients: Number of patients to include (random sample)
        patient_ids: Specific patient IDs to include

    Returns:
        DataFrame sorted by (patient_id, hour_from_admission)
    """
    logger.info("Loading panel data from {}", csv_path)
    df = pd.read_csv(csv_path)

    if patient_ids:
        df = df[df["patient_id"].isin(patient_ids)]
    elif n_patients:
        sampled_ids = df["patient_id"].unique()[:n_patients]
        df = df[df["patient_id"].isin(sampled_ids)]

    df = df.sort_values(["patient_id", "hour_from_admission"]).reset_index(drop=True)
    logger.info("Loaded {} records for {} patients", len(df), df["patient_id"].nunique())
    return df


def stream_to_api(
    df: pd.DataFrame,
    base_url: str = f"http://localhost:{api_settings.port}",
    batch_size: int = 100,
    delay_ms: float = 0,
) -> dict:
    """
    Stream records to the Ingest API in batches.

    Args:
        df: DataFrame of ICU records
        base_url: Ingest API base URL
        batch_size: Number of records per batch request
        delay_ms: Delay between batches (ms) to simulate real-time

    Returns:
        dict with ingestion statistics
    """
    endpoint = f"{base_url}/ingest/batch"
    total_records = len(df)
    total_sent = 0
    total_accepted = 0
    total_rejected = 0
    errors = 0

    logger.info(
        "Streaming {} records to {} (batch_size={}, delay={}ms)",
        total_records,
        endpoint,
        batch_size,
        delay_ms,
    )

    with httpx.Client(timeout=30.0) as client:
        for start_idx in tqdm(
            range(0, total_records, batch_size),
            desc="Streaming batches",
            unit="batch",
        ):
            batch_df = df.iloc[start_idx : start_idx + batch_size]
            records = batch_df.to_dict(orient="records")

            try:
                response = client.post(
                    endpoint,
                    json={"records": records},
                )
                response.raise_for_status()

                result = response.json()
                total_sent += len(records)
                total_accepted += result.get("records_valid", 0)
                total_rejected += result.get("records_rejected", 0)

            except httpx.HTTPStatusError as e:
                logger.error("HTTP error: {} — {}", e.response.status_code, e.response.text)
                errors += 1
            except httpx.ConnectError:
                logger.error(
                    "Connection refused — is the Ingest API running at {}?", base_url
                )
                errors += 1
                if errors > 5:
                    logger.error("Too many connection errors. Aborting.")
                    break
            except Exception as e:
                logger.error("Unexpected error: {}", e)
                errors += 1

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    stats = {
        "total_records": total_records,
        "total_sent": total_sent,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "errors": errors,
    }
    logger.info("Stream complete: {}", stats)
    return stats


def stream_direct_to_datalake(
    df: pd.DataFrame,
    n_patients: int | None = None,
) -> dict:
    """
    Bypass the API and write directly to the data lake.
    Useful for initial data loading without running the API server.
    """
    from src.ingestion.data_lake_router import DataLakeRouter

    router = DataLakeRouter()
    total = 0

    if n_patients:
        patient_ids = df["patient_id"].unique()[:n_patients]
        df = df[df["patient_id"].isin(patient_ids)]

    logger.info("Direct loading {} records to data lake", len(df))

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading to data lake"):
        router.store(
            data=row.to_dict(),
            patient_id=str(int(row["patient_id"])),
            data_type="icu-batch",
        )
        total += 1

    logger.info("Direct load complete: {} records", total)
    return {"total_loaded": total}


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JeevanSync ICU Data Stream Simulator")
    parser.add_argument(
        "--mode",
        choices=["api", "direct"],
        default="direct",
        help="Streaming mode: 'api' sends to Ingest API, 'direct' writes to data lake",
    )
    parser.add_argument("--patients", type=int, default=None, help="Number of patients to stream")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for API mode")
    parser.add_argument("--delay", type=float, default=0, help="Delay between batches (ms)")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(data_settings.hourly_panel_csv),
        help="Path to the panel CSV file",
    )
    args = parser.parse_args()

    df = load_panel_data(Path(args.csv), n_patients=args.patients)

    if args.mode == "api":
        stats = stream_to_api(df, batch_size=args.batch_size, delay_ms=args.delay)
    else:
        stats = stream_direct_to_datalake(df, n_patients=args.patients)

    logger.info("Final stats: {}", stats)


if __name__ == "__main__":
    main()
