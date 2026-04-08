"""
G_One_Sync AI — Training Pipeline
====================================
End-to-end training orchestrator that:
1. Loads preprocessed data
2. Trains XGBoost (GPU), BiLSTM (GPU), Transformer (GPU)
3. Generates SHAP explanations
4. Builds hybrid ensemble
5. Logs everything to MLflow

Usage:
    python -m src.modeling.train_pipeline
    python -m src.modeling.train_pipeline --model xgboost
    python -m src.modeling.train_pipeline --model all --hpo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config.settings import data_settings, model_settings, feature_settings
from src.modeling.data_loader import load_processed_splits
from src.modeling.xgboost_trainer import XGBoostTrainer
from src.modeling.lstm_trainer import BiLSTMTrainer
from src.modeling.transformer_trainer import TransformerTrainer
from src.modeling.ensemble import HybridEnsemble
from src.preprocessing.sliding_window import SlidingWindowBuilder


def prepare_sequence_data(
    splits: dict[str, pd.DataFrame],
    window_size: int = 12,
    target_column: str = "deterioration_next_12h",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Convert tabular splits into windowed sequences for LSTM/Transformer.
    Uses the raw numeric features to build (N, W, F) tensors.
    """
    logger.info("Building sequence data (window={}h)...", window_size)

    # Use the core numeric features for sequence models
    seq_features = feature_settings.vital_columns + feature_settings.lab_columns

    builder = SlidingWindowBuilder(
        window_size=window_size,
        feature_columns=seq_features,
    )

    sequence_data = {}
    for split_name, df in splits.items():
        # Load the hourly panel for this split's patients
        panel_path = data_settings.hourly_panel_csv
        if panel_path.exists():
            full_panel = pd.read_csv(panel_path)
            patient_ids = df["patient_id"].unique()
            panel = full_panel[full_panel["patient_id"].isin(patient_ids)].copy()

            if target_column in panel.columns:
                # Build windows
                window_data = builder.build_windows(panel, show_progress=False)
                windows = window_data["windows"]  # (N, W, F)
                pids = window_data["patient_ids"]
                hours = window_data["hours"]

                # Get labels for each window
                target_lookup = panel.set_index(["patient_id", "hour_from_admission"])[target_column]
                labels = np.array([
                    target_lookup.get((pid, hour), np.nan)
                    for pid, hour in zip(pids, hours)
                ])

                # Drop NaN labels
                valid = ~np.isnan(labels)
                sequence_data[split_name] = (
                    windows[valid].astype(np.float32),
                    labels[valid].astype(np.float32),
                )
                logger.info(
                    "  {}: {} windows ({}x{}), {:.1%} positive",
                    split_name, valid.sum(), window_size, len(seq_features),
                    labels[valid].mean(),
                )
            else:
                logger.warning("Target column '{}' not in panel", target_column)
        else:
            logger.warning("Hourly panel not found at {}", panel_path)

    return sequence_data


def train_xgboost(
    splits: dict[str, pd.DataFrame],
    target_column: str = "deterioration_next_12h",
) -> tuple[XGBoostTrainer, dict]:
    """Train XGBoost with GPU acceleration."""
    logger.info("╔═══════════════════════════════════════╗")
    logger.info("║  Training XGBoost (GPU)               ║")
    logger.info("╚═══════════════════════════════════════╝")

    trainer = XGBoostTrainer(
        n_estimators=model_settings.xgb_n_estimators,
        max_depth=model_settings.xgb_max_depth,
        learning_rate=model_settings.xgb_learning_rate,
    )

    val_metrics = trainer.train(
        train_data=splits["train"],
        val_data=splits["val"],
        run_name="xgboost-gpu",
        target_column=target_column,
    )

    # Test evaluation
    test_metrics = trainer.evaluate(splits["test"], split_name="test")

    # Feature importance
    importance = trainer.get_feature_importance(top_n=20)

    # Save
    trainer.finalize_run()

    return trainer, {
        "val": val_metrics.to_dict(),
        "test": test_metrics.to_dict(),
        "importance": importance.head(20).to_dict("records"),
    }


def train_bilstm(
    seq_data: dict[str, tuple[np.ndarray, np.ndarray]],
    input_size: int = 11,
) -> tuple[BiLSTMTrainer, dict]:
    """Train BiLSTM + Attention with GPU acceleration."""
    logger.info("╔═══════════════════════════════════════╗")
    logger.info("║  Training BiLSTM + Attention (GPU)    ║")
    logger.info("╚═══════════════════════════════════════╝")

    trainer = BiLSTMTrainer(
        input_size=input_size,
        hidden_size=model_settings.lstm_hidden_size,
        num_layers=model_settings.lstm_num_layers,
        batch_size=model_settings.batch_size,
        epochs=model_settings.epochs,
    )

    val_metrics = trainer.train(
        train_data=seq_data["train"],
        val_data=seq_data["val"],
        run_name="bilstm-attention-gpu",
    )

    test_metrics = trainer.evaluate(seq_data["test"], split_name="test")
    trainer.finalize_run()

    return trainer, {
        "val": val_metrics.to_dict(),
        "test": test_metrics.to_dict(),
    }


def train_transformer(
    seq_data: dict[str, tuple[np.ndarray, np.ndarray]],
    input_size: int = 11,
) -> tuple[TransformerTrainer, dict]:
    """Train Temporal Transformer with GPU acceleration."""
    logger.info("╔═══════════════════════════════════════╗")
    logger.info("║  Training Transformer (GPU)           ║")
    logger.info("╚═══════════════════════════════════════╝")

    trainer = TransformerTrainer(
        input_size=input_size,
        d_model=model_settings.transformer_d_model,
        nhead=model_settings.transformer_nhead,
        num_layers=model_settings.transformer_num_layers,
        batch_size=model_settings.batch_size,
        epochs=model_settings.epochs,
    )

    val_metrics = trainer.train(
        train_data=seq_data["train"],
        val_data=seq_data["val"],
        run_name="transformer-gpu",
    )

    test_metrics = trainer.evaluate(seq_data["test"], split_name="test")
    trainer.finalize_run()

    return trainer, {
        "val": val_metrics.to_dict(),
        "test": test_metrics.to_dict(),
    }


def build_ensemble(
    trainers: dict[str, object],
    splits: dict[str, pd.DataFrame],
    seq_data: dict[str, tuple[np.ndarray, np.ndarray]],
    target_column: str = "deterioration_next_12h",
) -> dict:
    """Build and evaluate the hybrid ensemble."""
    logger.info("╔═══════════════════════════════════════╗")
    logger.info("║  Building Hybrid Ensemble             ║")
    logger.info("╚═══════════════════════════════════════╝")

    val_predictions = {}
    test_predictions = {}

    if "xgboost" in trainers:
        val_predictions["xgboost"] = trainers["xgboost"]._predict_proba_impl(splits["val"])
        test_predictions["xgboost"] = trainers["xgboost"]._predict_proba_impl(splits["test"])

    if "bilstm" in trainers:
        val_predictions["bilstm"] = trainers["bilstm"]._predict_proba_impl(seq_data["val"])
        test_predictions["bilstm"] = trainers["bilstm"]._predict_proba_impl(seq_data["test"])

    if "transformer" in trainers:
        val_predictions["transformer"] = trainers["transformer"]._predict_proba_impl(seq_data["val"])
        test_predictions["transformer"] = trainers["transformer"]._predict_proba_impl(seq_data["test"])

    if len(val_predictions) < 2:
        logger.warning("Need at least 2 models for ensemble, skipping")
        return {}

    # Get labels
    val_labels = splits["val"][target_column].values
    test_labels = splits["test"][target_column].values

    # Fit ensemble
    ensemble = HybridEnsemble(strategy="learned")
    weights = ensemble.fit(val_predictions, val_labels)

    # Evaluate on test
    test_metrics = ensemble.evaluate(test_predictions, test_labels)

    return {
        "weights": weights,
        "test": test_metrics.to_dict(),
    }


def run_shap_explanations(
    trainer: XGBoostTrainer,
    splits: dict[str, pd.DataFrame],
    target_column: str = "deterioration_next_12h",
) -> dict:
    """Generate SHAP explanations for the XGBoost model."""
    logger.info("╔═══════════════════════════════════════╗")
    logger.info("║  SHAP Explainability                  ║")
    logger.info("╚═══════════════════════════════════════╝")

    from src.explainability.shap_explainer import SHAPExplainer

    exclude = {"patient_id", "hour_from_admission", target_column}
    feature_cols = [c for c in splits["test"].columns if c not in exclude]
    X_test = splits["test"][feature_cols].values.astype(np.float32)

    explainer = SHAPExplainer(
        model=trainer.model,
        model_type="xgboost",
        feature_names=feature_cols,
    )

    importance = explainer.compute_global_importance(X_test, max_samples=500)
    summary = explainer.generate_clinical_summary(importance)

    # Explain a high-risk patient
    y_prob = trainer._predict_proba_impl(splits["test"])
    high_risk_idx = np.argmax(y_prob)
    patient_explanation = explainer.explain_patient(
        X_test[high_risk_idx],
        patient_id=int(splits["test"].iloc[high_risk_idx].get("patient_id", 0)),
    )

    return {
        "top_features": importance.head(10).to_dict("records"),
        "clinical_summary": summary,
        "sample_explanation": {
            "patient_id": patient_explanation["patient_id"],
            "top_risk_feature": patient_explanation["top_risk_feature"],
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="G_One_Sync Training Pipeline")
    parser.add_argument("--model", choices=["xgboost", "bilstm", "transformer", "all"], default="all")
    parser.add_argument("--target", default="deterioration_next_12h")
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    start_time = time.time()

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("🔥 GPU: {} ({:.1f} GB VRAM)", gpu_name, vram)
    else:
        logger.warning("⚠️ No GPU detected — training will be slower")

    # Step 1: Load preprocessed data
    logger.info("═══ Loading Preprocessed Data ═══")
    splits = load_processed_splits(target_column=args.target)
    if not splits:
        logger.error("No processed data found. Run preprocessing first:")
        logger.error("  python -m src.preprocessing.pipeline --window-size 12")
        return

    # Step 2: Prepare sequence data for LSTM/Transformer
    seq_data = {}
    if args.model in ("bilstm", "transformer", "all"):
        seq_data = prepare_sequence_data(splits, window_size=args.window_size, target_column=args.target)

    # Step 3: Train models
    trainers = {}
    all_results = {}

    if args.model in ("xgboost", "all"):
        xgb_trainer, xgb_results = train_xgboost(splits, args.target)
        trainers["xgboost"] = xgb_trainer
        all_results["xgboost"] = xgb_results

    if args.model in ("bilstm", "all") and seq_data:
        input_size = seq_data["train"][0].shape[2]
        lstm_trainer, lstm_results = train_bilstm(seq_data, input_size)
        trainers["bilstm"] = lstm_trainer
        all_results["bilstm"] = lstm_results

    if args.model in ("transformer", "all") and seq_data:
        input_size = seq_data["train"][0].shape[2]
        tf_trainer, tf_results = train_transformer(seq_data, input_size)
        trainers["transformer"] = tf_trainer
        all_results["transformer"] = tf_results

    # Step 4: Ensemble
    if not args.no_ensemble and len(trainers) >= 2:
        ens_results = build_ensemble(trainers, splits, seq_data, args.target)
        all_results["ensemble"] = ens_results

    # Step 5: SHAP Explanations
    if not args.no_shap and "xgboost" in trainers:
        shap_results = run_shap_explanations(trainers["xgboost"], splits, args.target)
        all_results["shap"] = shap_results

    # Step 6: Summary
    elapsed = time.time() - start_time
    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info("║  🏁 Training Pipeline Complete                    ║")
    logger.info("║  Total time: {:.1f}s                              ║".format(elapsed))
    logger.info("╠═══════════════════════════════════════════════════╣")

    for model_name, results in all_results.items():
        if "test" in results:
            test = results["test"]
            logger.info(
                "║  {} — AUROC: {:.4f} | AUPRC: {:.4f} | F1: {:.4f}",
                model_name.ljust(12), test.get("auroc", 0), test.get("auprc", 0), test.get("f1", 0),
            )

    logger.info("╚═══════════════════════════════════════════════════╝")

    # Save summary
    summary_path = model_settings.artifacts_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved → {}", summary_path)


if __name__ == "__main__":
    main()
