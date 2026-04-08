"""
G_One_Sync AI — SHAP Explainability Engine
=============================================
Provides local and global feature importance explanations
for all model types using SHAP (SHapley Additive exPlanations).

Supports:
- TreeExplainer (XGBoost) — exact, GPU-computed
- DeepExplainer (BiLSTM, Transformer) — gradient-based
- Global summary plots + local patient-level explanations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
from loguru import logger

from config.settings import model_settings


class SHAPExplainer:
    """
    SHAP-based model explainability for clinical deterioration predictions.
    Generates both global feature importance and per-patient explanations.
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "xgboost",
        feature_names: Optional[list[str]] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Args:
            model: Trained model (XGBClassifier, PyTorch Module, etc.)
            model_type: 'xgboost', 'bilstm', or 'transformer'
            feature_names: Feature column names for labeling
            output_dir: Directory to save explanation artifacts
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.output_dir = output_dir or model_settings.artifacts_dir / "explanations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.background_data: Optional[np.ndarray] = None

        logger.info(
            "SHAP Explainer initialized: model_type='{}', features={}",
            model_type, len(self.feature_names),
        )

    def _create_explainer(
        self,
        background_data: np.ndarray,
        max_background: int = 200,
    ) -> shap.Explainer:
        """Create the appropriate SHAP explainer for the model type."""

        # Subsample background for efficiency
        if len(background_data) > max_background:
            idx = np.random.choice(len(background_data), max_background, replace=False)
            background_data = background_data[idx]

        self.background_data = background_data

        if self.model_type == "xgboost":
            # TreeExplainer — exact SHAP values, very fast
            explainer = shap.TreeExplainer(self.model)
            logger.info("Created TreeExplainer (exact Shapley values)")

        elif self.model_type in ("bilstm", "transformer"):
            # GradientExplainer for PyTorch models
            import torch
            bg_tensor = torch.tensor(background_data, dtype=torch.float32)
            if torch.cuda.is_available():
                bg_tensor = bg_tensor.to("cuda")
                self.model = self.model.to("cuda")

            explainer = shap.GradientExplainer(self.model, bg_tensor)
            logger.info(
                "Created GradientExplainer ({} background samples)",
                len(background_data),
            )

        else:
            # KernelExplainer — model agnostic
            explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model,
                background_data,
            )
            logger.info("Created KernelExplainer (model-agnostic)")

        return explainer

    # ── Global Explanations ──────────────────────────────────────────────

    def compute_global_importance(
        self,
        X: np.ndarray,
        background: Optional[np.ndarray] = None,
        max_samples: int = 1000,
    ) -> pd.DataFrame:
        """
        Compute global SHAP feature importance.

        Args:
            X: Feature matrix to explain (N, F) or (N, W, F) for sequences
            background: Background data for explainer initialization
            max_samples: Max samples to explain (for speed)

        Returns:
            DataFrame with feature importance rankings
        """
        if background is None:
            background = X

        if self.explainer is None:
            self.explainer = self._create_explainer(background)

        # Subsample for speed
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        logger.info("Computing SHAP values for {} samples...", len(X_sample))
        self.shap_values = self.explainer.shap_values(X_sample)

        # Handle multi-output (binary classification returns list of 2)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # P(deterioration=1)

        # Compute mean absolute SHAP values per feature
        if self.shap_values.ndim == 2:
            mean_abs = np.abs(self.shap_values).mean(axis=0)
        elif self.shap_values.ndim == 3:
            # Sequence models: (N, W, F) → average over time then samples
            mean_abs = np.abs(self.shap_values).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(self.shap_values).mean(axis=0)

        # Build importance DataFrame
        names = self.feature_names if len(self.feature_names) == len(mean_abs) else [
            f"feature_{i}" for i in range(len(mean_abs))
        ]
        importance_df = pd.DataFrame({
            "feature": names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["cumulative_importance"] = (
            importance_df["mean_abs_shap"].cumsum() / importance_df["mean_abs_shap"].sum()
        )

        # Log top features
        logger.info("═══ Global Feature Importance (SHAP) ═══")
        for _, row in importance_df.head(15).iterrows():
            logger.info(
                "  #{:2d} {:.4f} — {} ({:.1%} cumulative)",
                int(row["rank"]), row["mean_abs_shap"],
                row["feature"], row["cumulative_importance"],
            )

        # Save
        importance_df.to_csv(self.output_dir / "global_importance.csv", index=False)

        return importance_df

    # ── Local (Patient-Level) Explanations ───────────────────────────────

    def explain_patient(
        self,
        patient_features: np.ndarray,
        patient_id: Optional[int] = None,
        background: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """
        Generate a local explanation for a single patient.

        Args:
            patient_features: (F,) or (W, F) feature array for one patient
            patient_id: Optional patient ID for labeling
            background: Background data for explainer

        Returns:
            Dict with SHAP values, risk factors, and protective factors
        """
        if self.explainer is None:
            if background is None:
                raise ValueError("Must provide background data or call compute_global_importance first")
            self.explainer = self._create_explainer(background)

        # Ensure 2D input
        if patient_features.ndim == 1:
            patient_features = patient_features.reshape(1, -1)

        # Compute SHAP values
        sv = self.explainer.shap_values(patient_features)
        if isinstance(sv, list):
            sv = sv[1]

        shap_vals = sv.flatten()

        # Build explanation
        names = self.feature_names if len(self.feature_names) == len(shap_vals) else [
            f"feature_{i}" for i in range(len(shap_vals))
        ]

        explanation_df = pd.DataFrame({
            "feature": names,
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals),
            "feature_value": patient_features.flatten()[:len(names)],
        }).sort_values("abs_shap", ascending=False)

        # Separate risk and protective factors
        risk_factors = explanation_df[explanation_df["shap_value"] > 0].head(10)
        protective_factors = explanation_df[explanation_df["shap_value"] < 0].head(10)

        result = {
            "patient_id": patient_id,
            "shap_values": shap_vals,
            "risk_factors": risk_factors.to_dict("records"),
            "protective_factors": protective_factors.to_dict("records"),
            "top_risk_feature": risk_factors.iloc[0]["feature"] if len(risk_factors) > 0 else None,
            "predicted_probability": float(self.explainer.expected_value + shap_vals.sum())
                if hasattr(self.explainer, "expected_value") else None,
        }

        # Log
        pid_label = f"Patient {patient_id}" if patient_id else "Patient"
        logger.info("═══ {} Explanation ═══", pid_label)
        logger.info("  Risk Factors (↑ deterioration risk):")
        for _, row in risk_factors.head(5).iterrows():
            logger.info("    +{:.4f} {} (value={:.2f})",
                         row["shap_value"], row["feature"], row["feature_value"])
        logger.info("  Protective Factors (↓ deterioration risk):")
        for _, row in protective_factors.head(5).iterrows():
            logger.info("    {:.4f} {} (value={:.2f})",
                         row["shap_value"], row["feature"], row["feature_value"])

        return result

    # ── Batch Explanations ───────────────────────────────────────────────

    def explain_batch(
        self,
        X: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
        top_n_features: int = 5,
    ) -> pd.DataFrame:
        """
        Generate explanations for a batch of patients.

        Returns:
            DataFrame with top-N risk factors per patient
        """
        if self.shap_values is None:
            self.compute_global_importance(X)

        records = []
        for i in range(len(X)):
            sv = self.shap_values[i]
            if sv.ndim > 1:
                sv = sv.mean(axis=0)  # Average over timesteps for sequences

            pid = patient_ids[i] if patient_ids is not None else i

            # Top risk factors
            top_idx = np.argsort(sv)[-top_n_features:][::-1]
            for rank, idx in enumerate(top_idx, 1):
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                records.append({
                    "patient_id": pid,
                    "rank": rank,
                    "feature": name,
                    "shap_value": float(sv[idx]),
                    "direction": "risk" if sv[idx] > 0 else "protective",
                })

        result_df = pd.DataFrame(records)
        result_df.to_csv(self.output_dir / "batch_explanations.csv", index=False)
        logger.info("Batch explanations: {} patients, saved to {}", len(X), self.output_dir)

        return result_df

    # ── Clinical Summary ─────────────────────────────────────────────────

    def generate_clinical_summary(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """
        Generate a clinical summary of the most important features.

        Maps feature names to clinical interpretations.
        """
        clinical_mapping = {
            "lactate": "Tissue hypoperfusion / sepsis marker",
            "heart_rate": "Cardiovascular instability",
            "respiratory_rate": "Respiratory distress",
            "systolic_bp": "Hemodynamic instability",
            "spo2_pct": "Oxygenation status",
            "temperature_c": "Infection / inflammatory response",
            "creatinine": "Renal function",
            "wbc_count": "Immune response / infection",
            "crp_level": "Systemic inflammation",
            "hemoglobin": "Oxygen-carrying capacity",
            "shock_index": "Compensated shock indicator",
            "map_": "Organ perfusion pressure",
            "qsofa": "Sepsis screening criteria",
            "hrv_": "Autonomic nervous system function",
            "bpv_": "Baroreflex sensitivity",
        }

        summary = {
            "model_type": self.model_type,
            "n_features_explained": len(importance_df),
            "top_features": [],
        }

        for _, row in importance_df.head(top_n).iterrows():
            feature = row["feature"]
            clinical_meaning = "Clinical feature"
            for key, meaning in clinical_mapping.items():
                if key in feature.lower():
                    clinical_meaning = meaning
                    break

            summary["top_features"].append({
                "feature": feature,
                "importance": float(row["mean_abs_shap"]),
                "rank": int(row["rank"]),
                "clinical_meaning": clinical_meaning,
            })

        # Save summary
        summary_path = self.output_dir / "clinical_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Clinical summary saved → {}", summary_path)
        return summary
