"""
G_One_Sync AI — Prometheus Metrics Collector
================================================
Collects and exposes prediction, system, and clinical KPI metrics
via Prometheus for real-time monitoring and Grafana dashboards.
"""

from __future__ import annotations

import time
import threading
from typing import Optional

from loguru import logger
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    start_http_server,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


class MetricsCollector:
    """
    Centralized Prometheus metrics for the G_One_Sync prediction system.

    Metrics exposed:
        - Prediction latency, count, errors
        - Model confidence distribution
        - Risk level distribution
        - GPU utilization and VRAM
        - Clinical KPIs (alert rate, high-risk count)
    """

    _instance: Optional["MetricsCollector"] = None

    def __new__(cls, *args, **kwargs):
        """Singleton — only one collector per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, port: int = 9090):
        if self._initialized:
            return
        self._initialized = True

        # ── Prediction Metrics ───────────────────────────────────
        self.prediction_count = Counter(
            "g1sync_predictions_total",
            "Total number of predictions made",
            labelnames=["model_type", "risk_level"],
        )

        self.prediction_latency = Histogram(
            "g1sync_prediction_latency_seconds",
            "Prediction latency in seconds",
            labelnames=["model_type"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )

        self.prediction_errors = Counter(
            "g1sync_prediction_errors_total",
            "Total prediction errors",
            labelnames=["model_type", "error_type"],
        )

        self.prediction_confidence = Histogram(
            "g1sync_prediction_confidence",
            "Distribution of predicted deterioration probabilities",
            labelnames=["model_type"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        # ── Risk Level Distribution ──────────────────────────────
        self.risk_level_gauge = Gauge(
            "g1sync_active_patients_by_risk",
            "Number of active patients at each risk level",
            labelnames=["risk_level"],
        )

        # ── Model Performance ────────────────────────────────────
        self.model_auroc = Gauge(
            "g1sync_model_auroc",
            "Current model AUROC on rolling evaluation window",
            labelnames=["model_type"],
        )

        self.model_f1 = Gauge(
            "g1sync_model_f1",
            "Current model F1 score",
            labelnames=["model_type"],
        )

        # ── System Metrics ───────────────────────────────────────
        self.gpu_utilization = Gauge(
            "g1sync_gpu_utilization_percent",
            "GPU utilization percentage",
        )

        self.gpu_memory_used = Gauge(
            "g1sync_gpu_memory_used_bytes",
            "GPU memory used in bytes",
        )

        self.gpu_memory_total = Gauge(
            "g1sync_gpu_memory_total_bytes",
            "Total GPU memory in bytes",
        )

        self.cpu_usage = Gauge(
            "g1sync_cpu_usage_percent",
            "CPU usage percentage",
        )

        self.memory_usage = Gauge(
            "g1sync_memory_usage_percent",
            "System memory usage percentage",
        )

        # ── Clinical KPIs ────────────────────────────────────────
        self.alerts_fired = Counter(
            "g1sync_clinical_alerts_total",
            "Total clinical alerts fired",
            labelnames=["severity", "alert_type"],
        )

        self.high_risk_patients = Gauge(
            "g1sync_high_risk_patients_current",
            "Current number of HIGH/CRITICAL risk patients",
        )

        self.positive_prediction_rate = Gauge(
            "g1sync_positive_prediction_rate",
            "Rolling positive prediction rate (fraction above threshold)",
            labelnames=["model_type"],
        )

        # ── Drift Metrics ────────────────────────────────────────
        self.feature_drift_psi = Gauge(
            "g1sync_feature_drift_psi",
            "Population Stability Index for feature drift",
            labelnames=["feature_name"],
        )

        self.prediction_drift = Gauge(
            "g1sync_prediction_drift",
            "Mean predicted probability drift from baseline",
            labelnames=["model_type"],
        )

        # ── Model Info ───────────────────────────────────────────
        self.model_info = Info(
            "g1sync_model",
            "Currently deployed model metadata",
        )

        self._port = port
        self._server_started = False

        logger.info("MetricsCollector initialized (port={})", port)

    # ── Recording Methods ────────────────────────────────────────

    def record_prediction(
        self,
        model_type: str,
        risk_level: str,
        probability: float,
        latency_seconds: float,
    ) -> None:
        """Record a single prediction event."""
        self.prediction_count.labels(model_type=model_type, risk_level=risk_level).inc()
        self.prediction_latency.labels(model_type=model_type).observe(latency_seconds)
        self.prediction_confidence.labels(model_type=model_type).observe(probability)

    def record_error(self, model_type: str, error_type: str) -> None:
        """Record a prediction error."""
        self.prediction_errors.labels(model_type=model_type, error_type=error_type).inc()

    def record_alert(self, severity: str, alert_type: str) -> None:
        """Record a clinical alert."""
        self.alerts_fired.labels(severity=severity, alert_type=alert_type).inc()

    def update_risk_distribution(self, distribution: dict[str, int]) -> None:
        """Update active patient risk level counts."""
        for level, count in distribution.items():
            self.risk_level_gauge.labels(risk_level=level).set(count)

    def update_model_performance(
        self, model_type: str, auroc: float, f1: float
    ) -> None:
        """Update rolling model performance metrics."""
        self.model_auroc.labels(model_type=model_type).set(auroc)
        self.model_f1.labels(model_type=model_type).set(f1)

    def update_drift_metrics(
        self, feature_name: str, psi: float
    ) -> None:
        """Update drift PSI for a feature."""
        self.feature_drift_psi.labels(feature_name=feature_name).set(psi)

    def update_system_metrics(self) -> None:
        """Poll and update system resource metrics."""
        try:
            import psutil
            self.cpu_usage.set(psutil.cpu_percent())
            self.memory_usage.set(psutil.virtual_memory().percent)
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_mem
                self.gpu_memory_used.set(allocated)
                self.gpu_memory_total.set(total)
                self.gpu_utilization.set(allocated / total * 100)
        except Exception:
            pass

    # ── Server Management ────────────────────────────────────────

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        if not self._server_started:
            start_http_server(self._port)
            self._server_started = True
            logger.info("Prometheus metrics server started on port {}", self._port)

    def get_metrics_text(self) -> bytes:
        """Get current metrics as Prometheus text format."""
        return generate_latest()

    def start_system_polling(self, interval_seconds: float = 15.0) -> None:
        """Start background thread to poll system metrics."""

        def _poll():
            while True:
                self.update_system_metrics()
                time.sleep(interval_seconds)

        thread = threading.Thread(target=_poll, daemon=True)
        thread.start()
        logger.info("System metrics polling started ({}s interval)", interval_seconds)
