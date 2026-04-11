"""
G_One_Sync AI — Clinical Alerting Engine
============================================
Rule-based alerting for clinical deterioration, system health,
and data quality issues. Supports severity levels and cooldowns.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from loguru import logger


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """A single alert instance."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str  # "clinical", "system", "drift"
    timestamp: float = field(default_factory=time.time)
    patient_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "metadata": self.metadata,
        }


@dataclass
class AlertRule:
    """Defines a condition that triggers an alert."""
    rule_id: str
    name: str
    severity: AlertSeverity
    source: str
    condition: Callable[..., bool]
    message_template: str
    cooldown_seconds: float = 300.0  # 5 min default
    _last_fired: float = 0.0


class AlertingEngine:
    """
    Rule-based alerting engine for G_One_Sync AI.

    Supports:
    - Clinical alerts (high-risk patients, rapid deterioration)
    - System alerts (GPU memory, model errors, latency)
    - Drift alerts (feature/prediction distribution shift)
    - Cooldown periods to prevent alert fatigue
    """

    def __init__(self):
        self.rules: dict[str, AlertRule] = {}
        self.alert_history: list[Alert] = []
        self.alert_callbacks: list[Callable[[Alert], None]] = []
        self._alert_counter = 0

        # Register default rules
        self._register_default_rules()

        logger.info("Alerting engine initialized with {} rules", len(self.rules))

    def _register_default_rules(self) -> None:
        """Register built-in clinical and system alert rules."""

        # ── Clinical Alerts ──────────────────────────────────────
        self.add_rule(AlertRule(
            rule_id="critical_risk",
            name="Critical Deterioration Risk",
            severity=AlertSeverity.CRITICAL,
            source="clinical",
            condition=lambda prob, **_: prob >= 0.75,
            message_template=(
                "⚠️ CRITICAL: Patient {patient_id} has {probability:.1%} "
                "deterioration risk. Immediate review recommended."
            ),
            cooldown_seconds=60.0,  # Allow rapid re-alerting for critical
        ))

        self.add_rule(AlertRule(
            rule_id="high_risk",
            name="High Deterioration Risk",
            severity=AlertSeverity.WARNING,
            source="clinical",
            condition=lambda prob, **_: 0.50 <= prob < 0.75,
            message_template=(
                "🔴 HIGH RISK: Patient {patient_id} at {probability:.1%} risk. "
                "Close monitoring required."
            ),
            cooldown_seconds=300.0,
        ))

        self.add_rule(AlertRule(
            rule_id="rapid_increase",
            name="Rapid Risk Increase",
            severity=AlertSeverity.CRITICAL,
            source="clinical",
            condition=lambda prob, prev_prob=None, **_: (
                prev_prob is not None and prob - prev_prob > 0.20
            ),
            message_template=(
                "📈 RAPID INCREASE: Patient {patient_id} risk jumped from "
                "{prev_probability:.1%} to {probability:.1%} in last hour."
            ),
            cooldown_seconds=120.0,
        ))

        # ── System Alerts ────────────────────────────────────────
        self.add_rule(AlertRule(
            rule_id="gpu_memory_high",
            name="GPU Memory High",
            severity=AlertSeverity.WARNING,
            source="system",
            condition=lambda gpu_pct, **_: gpu_pct > 90,
            message_template=(
                "🔧 GPU memory at {gpu_pct:.0f}%. Consider reducing batch size."
            ),
            cooldown_seconds=600.0,
        ))

        self.add_rule(AlertRule(
            rule_id="high_latency",
            name="High Prediction Latency",
            severity=AlertSeverity.WARNING,
            source="system",
            condition=lambda latency_ms, **_: latency_ms > 500,
            message_template=(
                "⏱️ Prediction latency is {latency_ms:.0f}ms (threshold: 500ms)."
            ),
            cooldown_seconds=300.0,
        ))

        self.add_rule(AlertRule(
            rule_id="model_error_spike",
            name="Model Error Rate Spike",
            severity=AlertSeverity.CRITICAL,
            source="system",
            condition=lambda error_rate, **_: error_rate > 0.05,
            message_template=(
                "🚨 Model error rate at {error_rate:.1%}. Check model health."
            ),
            cooldown_seconds=120.0,
        ))

        # ── Drift Alerts ─────────────────────────────────────────
        self.add_rule(AlertRule(
            rule_id="feature_drift",
            name="Feature Distribution Drift",
            severity=AlertSeverity.WARNING,
            source="drift",
            condition=lambda psi, **_: psi > 0.2,
            message_template=(
                "📊 Feature drift detected: {feature_name} PSI={psi:.3f}. "
                "Model retraining may be needed."
            ),
            cooldown_seconds=3600.0,  # Once per hour
        ))

        self.add_rule(AlertRule(
            rule_id="prediction_drift",
            name="Prediction Distribution Drift",
            severity=AlertSeverity.CRITICAL,
            source="drift",
            condition=lambda relative_drift, **_: relative_drift > 0.25,
            message_template=(
                "📉 Prediction drift: {relative_drift:.1%} shift from baseline. "
                "Model performance may be degraded."
            ),
            cooldown_seconds=1800.0,
        ))

    # ── Rule Management ──────────────────────────────────────────

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alerting rule."""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        """Remove an alerting rule."""
        self.rules.pop(rule_id, None)

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Register a callback to invoke when an alert fires.
        Use this to integrate with Slack, email, PagerDuty, etc.
        """
        self.alert_callbacks.append(callback)

    # ── Alert Evaluation ─────────────────────────────────────────

    def evaluate(self, rule_id: str, **context) -> Optional[Alert]:
        """
        Evaluate a single rule against provided context.

        Args:
            rule_id: The rule to evaluate
            **context: Variables used by the rule condition and message template.

        Returns:
            Alert if rule fires, None otherwise.
        """
        rule = self.rules.get(rule_id)
        if rule is None:
            return None

        # Check cooldown
        now = time.time()
        if now - rule._last_fired < rule.cooldown_seconds:
            return None

        # Evaluate condition
        try:
            if rule.condition(**context):
                self._alert_counter += 1
                alert = Alert(
                    alert_id=f"ALERT-{self._alert_counter:06d}",
                    severity=rule.severity,
                    title=rule.name,
                    message=rule.message_template.format(**context),
                    source=rule.source,
                    patient_id=context.get("patient_id"),
                    metadata=context,
                )

                rule._last_fired = now
                self.alert_history.append(alert)

                # Log the alert
                log_method = {
                    AlertSeverity.INFO: logger.info,
                    AlertSeverity.WARNING: logger.warning,
                    AlertSeverity.CRITICAL: logger.error,
                }[rule.severity]
                log_method("[{}] {}", alert.alert_id, alert.message)

                # Fire callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error("Alert callback error: {}", e)

                return alert

        except Exception as e:
            logger.debug("Rule '{}' evaluation error: {}", rule_id, e)

        return None

    def evaluate_prediction(
        self,
        patient_id: int,
        probability: float,
        prev_probability: Optional[float] = None,
    ) -> list[Alert]:
        """
        Evaluate all clinical rules for a patient prediction.

        Returns:
            List of fired alerts
        """
        context = {
            "patient_id": patient_id,
            "probability": probability,
            "prob": probability,
            "prev_prob": prev_probability,
            "prev_probability": prev_probability or 0,
        }

        alerts = []
        for rule_id, rule in self.rules.items():
            if rule.source == "clinical":
                alert = self.evaluate(rule_id, **context)
                if alert:
                    alerts.append(alert)

        return alerts

    def evaluate_system(
        self,
        gpu_pct: float = 0,
        latency_ms: float = 0,
        error_rate: float = 0,
    ) -> list[Alert]:
        """Evaluate all system health rules."""
        alerts = []
        sys_context = {
            "gpu_pct": gpu_pct,
            "latency_ms": latency_ms,
            "error_rate": error_rate,
        }

        for rule_id, rule in self.rules.items():
            if rule.source == "system":
                alert = self.evaluate(rule_id, **sys_context)
                if alert:
                    alerts.append(alert)

        return alerts

    def get_recent_alerts(
        self,
        limit: int = 50,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
    ) -> list[dict]:
        """Get recent alerts, optionally filtered."""
        alerts = self.alert_history[-limit:]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]

        return [a.to_dict() for a in reversed(alerts)]

    def get_stats(self) -> dict:
        """Get alert statistics."""
        counts = defaultdict(int)
        for a in self.alert_history:
            counts[a.severity.value] += 1

        return {
            "total_alerts": len(self.alert_history),
            "by_severity": dict(counts),
            "active_rules": len(self.rules),
        }
