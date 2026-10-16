"""
G_One_Sync AI — Firebase Cloud Messaging Push Notifications
==============================================================
Manages device token registration and push notification dispatch
for the mobile companion app. When ``fcm_enabled`` is False
(default), all push operations are no-ops — no Firebase
dependency required at dev time.

Device tokens are persisted to a local JSON file so registrations
survive server restarts. In production, replace with a database-backed
store.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from config.settings import mobile_settings

# ── Pydantic Schemas ────────────────────────────────────────────────────


class DeviceRegistration(BaseModel):
    """Device token registration request."""

    device_token: str = Field(..., min_length=10, description="FCM device token")
    device_type: str = Field(..., pattern="^(ios|android)$", description="Device platform")
    physician_id: str = Field(..., description="Physician username or ID")


class DeviceInfo(BaseModel):
    """Stored device information."""

    device_token: str
    device_type: str
    physician_id: str
    registered_at: str
    last_push_at: Optional[str] = None
    push_count: int = 0


class PushPayload(BaseModel):
    """Push notification payload."""

    title: str
    body: str
    patient_id: Optional[int] = None
    risk_level: Optional[str] = None
    deterioration_probability: Optional[float] = None
    priority: str = "high"


class PushResult(BaseModel):
    """Result of a push notification send attempt."""

    success: bool
    device_token: str
    message_id: Optional[str] = None
    error: Optional[str] = None


# ── Device Registry ─────────────────────────────────────────────────────


class DeviceRegistry:
    """Manages FCM device token registrations.

    Persists tokens to a JSON file for survival across restarts.
    Thread-safe for single-process FastAPI (uvicorn workers=1).
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or mobile_settings.device_registry_path
        self._devices: dict[str, DeviceInfo] = {}
        self._load()

    def _load(self) -> None:
        """Load device registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    raw = json.load(f)
                self._devices = {
                    token: DeviceInfo(**info) for token, info in raw.items()
                }
                logger.info(
                    "Device registry loaded: {} devices from {}",
                    len(self._devices),
                    self.registry_path,
                )
            except Exception as e:
                logger.error("Failed to load device registry: {}", e)
                self._devices = {}
        else:
            logger.info("No device registry found at {} — starting fresh", self.registry_path)

    def _save(self) -> None:
        """Persist device registry to disk."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, "w") as f:
                json.dump(
                    {token: info.model_dump() for token, info in self._devices.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error("Failed to save device registry: {}", e)

    def register(self, registration: DeviceRegistration) -> DeviceInfo:
        """Register or update a device token.

        Args:
            registration: Device token registration data.

        Returns:
            The stored DeviceInfo.
        """
        info = DeviceInfo(
            device_token=registration.device_token,
            device_type=registration.device_type,
            physician_id=registration.physician_id,
            registered_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._devices[registration.device_token] = info
        self._save()
        logger.info(
            "Device registered: physician={} type={} token={}...{}",
            registration.physician_id,
            registration.device_type,
            registration.device_token[:8],
            registration.device_token[-4:],
        )
        return info

    def unregister(self, device_token: str) -> bool:
        """Remove a device token from the registry.

        Args:
            device_token: The FCM token to remove.

        Returns:
            True if the token was found and removed, False otherwise.
        """
        if device_token in self._devices:
            del self._devices[device_token]
            self._save()
            logger.info("Device unregistered: token={}...{}", device_token[:8], device_token[-4:])
            return True
        return False

    def get_devices_for_physician(self, physician_id: str) -> list[DeviceInfo]:
        """Get all registered devices for a physician.

        Args:
            physician_id: Physician username or ID.

        Returns:
            List of DeviceInfo for the physician's devices.
        """
        return [
            info for info in self._devices.values()
            if info.physician_id == physician_id
        ]

    def get_all_devices(self) -> list[DeviceInfo]:
        """Get all registered devices.

        Returns:
            List of all DeviceInfo records.
        """
        return list(self._devices.values())

    def device_count(self) -> int:
        """Get total number of registered devices."""
        return len(self._devices)

    def update_push_stats(self, device_token: str) -> None:
        """Update push statistics for a device after a successful push.

        Args:
            device_token: The device token that received the push.
        """
        if device_token in self._devices:
            self._devices[device_token].last_push_at = (
                datetime.now(tz=timezone.utc).isoformat()
            )
            self._devices[device_token].push_count += 1
            self._save()


# ── Push Notification Service ───────────────────────────────────────────


class PushNotificationService:
    """Sends push notifications via Firebase Cloud Messaging.

    When ``fcm_enabled`` is False, all sends are logged but no actual
    HTTP request is made to Firebase. This allows development and
    testing without FCM credentials.
    """

    def __init__(self, device_registry: Optional[DeviceRegistry] = None):
        self.registry = device_registry or DeviceRegistry()
        self._fcm_enabled = mobile_settings.fcm_enabled

    async def send_to_device(
        self,
        device_token: str,
        payload: PushPayload,
    ) -> PushResult:
        """Send a push notification to a single device.

        Args:
            device_token: FCM device token.
            payload: Notification content.

        Returns:
            PushResult indicating success or failure.
        """
        if not self._fcm_enabled:
            logger.info(
                "FCM disabled — simulating push to {}...{}: title='{}' body='{}'",
                device_token[:8],
                device_token[-4:],
                payload.title,
                payload.body,
            )
            self.registry.update_push_stats(device_token)
            return PushResult(
                success=True,
                device_token=device_token,
                message_id=f"sim-{int(time.time() * 1000)}",
            )

        # Production FCM send via HTTP v1 API
        try:
            import httpx

            message = {
                "message": {
                    "token": device_token,
                    "notification": {
                        "title": payload.title,
                        "body": payload.body,
                    },
                    "data": {
                        "patient_id": str(payload.patient_id or ""),
                        "risk_level": payload.risk_level or "",
                        "probability": str(payload.deterioration_probability or ""),
                    },
                    "android": {
                        "priority": payload.priority,
                        "notification": {"sound": "critical_alert"},
                    },
                    "apns": {
                        "payload": {
                            "aps": {
                                "sound": "critical_alert.caf",
                                "badge": 1,
                            }
                        }
                    },
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://fcm.googleapis.com/v1/projects/"
                    f"{mobile_settings.fcm_project_id}/messages:send",
                    json=message,
                    headers={
                        "Authorization": f"Bearer {mobile_settings.fcm_server_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                result = response.json()

            self.registry.update_push_stats(device_token)
            logger.info("FCM push sent to {}...{}", device_token[:8], device_token[-4:])

            return PushResult(
                success=True,
                device_token=device_token,
                message_id=result.get("name", ""),
            )

        except Exception as e:
            logger.error("FCM push failed for {}...{}: {}", device_token[:8], device_token[-4:], e)
            return PushResult(
                success=False,
                device_token=device_token,
                error=str(e),
            )

    async def send_risk_alert(
        self,
        physician_id: str,
        patient_id: int,
        risk_level: str,
        probability: float,
    ) -> list[PushResult]:
        """Send a risk alert to all of a physician's registered devices.

        Args:
            physician_id: Target physician.
            patient_id: Patient triggering the alert.
            risk_level: Risk level string (CRITICAL, HIGH, etc.).
            probability: Deterioration probability.

        Returns:
            List of PushResult for each device.
        """
        devices = self.registry.get_devices_for_physician(physician_id)
        if not devices:
            logger.debug("No devices registered for physician={}", physician_id)
            return []

        emoji = "🔴" if risk_level == "CRITICAL" else "🟠"
        payload = PushPayload(
            title=f"{emoji} {risk_level} — Patient #{patient_id}",
            body=(
                f"Deterioration probability: {probability:.1%}. "
                f"Immediate review recommended."
            ),
            patient_id=patient_id,
            risk_level=risk_level,
            deterioration_probability=probability,
            priority="high" if risk_level == "CRITICAL" else "normal",
        )

        results = []
        for device in devices:
            result = await self.send_to_device(device.device_token, payload)
            results.append(result)

        logger.info(
            "Risk alert sent to {} devices for physician={}: patient={} risk={}",
            len(results),
            physician_id,
            patient_id,
            risk_level,
        )
        return results

    async def send_test_notification(self, device_token: str) -> PushResult:
        """Send a test push notification to verify connectivity.

        Args:
            device_token: Target device token.

        Returns:
            PushResult.
        """
        payload = PushPayload(
            title="🧪 G_One_Sync AI — Test Notification",
            body="Push notifications are working correctly.",
            priority="normal",
        )
        return await self.send_to_device(device_token, payload)
