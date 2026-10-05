"""
G_One_Sync AI — FHIR R4 REST Client
=======================================
Async FHIR R4 client with OAuth2 SMART on FHIR authentication.
Uses httpx.AsyncClient for non-blocking HTTP requests.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from loguru import logger

from config.settings import emr_settings
from src.ingestion.emr.fhir_schemas import (
    FHIRBundle,
    FHIRObservation,
    FHIRPatient,
)


class FHIRAuthError(Exception):
    """Raised when FHIR OAuth2 authentication fails."""


class FHIRRequestError(Exception):
    """Raised when a FHIR REST request fails."""


class FHIRClient:
    """
    Async FHIR R4 REST client with OAuth2 SMART on FHIR authentication.

    Handles:
        - OAuth2 client credentials flow for SMART on FHIR auth
        - Token caching + automatic refresh
        - Retry with exponential backoff
        - Fetching Observations (vitals + labs) and Patient resources
    """

    # ── Retry Configuration ──────────────────────────────────────────
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # seconds

    def __init__(
        self,
        base_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        auth_token_url: str | None = None,
        timeout_s: int | None = None,
    ):
        self.base_url = (base_url or emr_settings.fhir_base_url).rstrip("/")
        self.client_id = client_id or emr_settings.fhir_client_id
        self.client_secret = client_secret or emr_settings.fhir_client_secret
        self.auth_token_url = auth_token_url or emr_settings.fhir_auth_token_url
        self.timeout_s = timeout_s or emr_settings.fhir_timeout_s

        # Token state
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

        # HTTP client (created lazily)
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "FHIRClient initialized | base_url={} | auth={}",
            self.base_url,
            "oauth2" if self.auth_token_url else "none",
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        logger.info("FHIRClient closed")

    # ── OAuth2 Authentication ────────────────────────────────────────

    async def authenticate(self) -> None:
        """
        Obtain an OAuth2 access token using client credentials flow.
        Tokens are cached and reused until expiry.
        """
        if not self.auth_token_url:
            logger.debug("No auth token URL configured — skipping authentication")
            return

        if self._access_token and time.time() < self._token_expires_at - 60:
            return  # Token still valid (with 60s buffer)

        client = await self._get_client()
        try:
            response = await client.post(
                self.auth_token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": "system/*.read",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            token_data = response.json()

            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in

            logger.info("FHIR OAuth2 token obtained (expires in {}s)", expires_in)

        except httpx.HTTPStatusError as e:
            logger.error("FHIR OAuth2 authentication failed: {} {}", e.response.status_code, e)
            raise FHIRAuthError(f"OAuth2 authentication failed: {e}") from e
        except Exception as e:
            logger.error("FHIR OAuth2 authentication error: {}", e)
            raise FHIRAuthError(f"OAuth2 authentication error: {e}") from e

    def _auth_headers(self) -> dict[str, str]:
        """Build authorization headers."""
        headers: dict[str, str] = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers

    # ── HTTP Request with Retry ──────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make an HTTP request to the FHIR server with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Resource path (e.g., /Patient/12345)
            params: Query parameters

        Returns:
            Parsed JSON response dict
        """
        await self.authenticate()
        client = await self._get_client()
        url = f"{self.base_url}{path}"

        last_error: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=self._auth_headers(),
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (401, 403):
                    # Token may have expired — force refresh
                    self._access_token = None
                    await self.authenticate()
                elif e.response.status_code >= 500:
                    # Server error — retry with backoff
                    backoff = self.RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "FHIR server error {} on attempt {}/{} — retrying in {:.1f}s",
                        e.response.status_code, attempt, self.MAX_RETRIES, backoff,
                    )
                    import asyncio
                    await asyncio.sleep(backoff)
                else:
                    # Client error (4xx) — don't retry
                    logger.error("FHIR request failed: {} {}", e.response.status_code, path)
                    raise FHIRRequestError(
                        f"FHIR request failed: {e.response.status_code} {path}"
                    ) from e

            except httpx.TimeoutException as e:
                last_error = e
                backoff = self.RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "FHIR request timeout on attempt {}/{} — retrying in {:.1f}s",
                    attempt, self.MAX_RETRIES, backoff,
                )
                import asyncio
                await asyncio.sleep(backoff)

            except Exception as e:
                logger.error("FHIR request unexpected error: {}", e)
                raise FHIRRequestError(f"Unexpected FHIR error: {e}") from e

        raise FHIRRequestError(
            f"FHIR request failed after {self.MAX_RETRIES} retries: {last_error}"
        )

    # ── Resource Fetchers ────────────────────────────────────────────

    async def get_patient(self, patient_id: str) -> FHIRPatient:
        """
        Fetch a Patient resource by ID.

        Args:
            patient_id: FHIR Patient resource ID

        Returns:
            Parsed FHIRPatient model
        """
        data = await self._request("GET", f"/Patient/{patient_id}")
        patient = FHIRPatient(**data)
        logger.debug("Fetched Patient/{} (gender={}, dob={})",
                      patient_id, patient.gender.value, patient.birthDate)
        return patient

    async def get_observation(self, observation_id: str) -> FHIRObservation:
        """
        Fetch a single Observation resource by ID.

        Args:
            observation_id: FHIR Observation resource ID

        Returns:
            Parsed FHIRObservation model
        """
        data = await self._request("GET", f"/Observation/{observation_id}")
        return FHIRObservation(**data)

    async def get_observations(
        self,
        patient_id: str,
        category: str | None = None,
        code: str | None = None,
        count: int = 50,
    ) -> list[FHIRObservation]:
        """
        Search for Observations for a patient.

        Args:
            patient_id: FHIR Patient resource ID
            category: Observation category filter (e.g., 'vital-signs', 'laboratory')
            code: LOINC code filter
            count: Maximum results to return

        Returns:
            List of FHIRObservation models
        """
        params: dict[str, Any] = {
            "patient": patient_id,
            "_count": count,
            "_sort": "-date",
        }
        if category:
            params["category"] = category
        if code:
            params["code"] = f"http://loinc.org|{code}"

        data = await self._request("GET", "/Observation", params=params)
        bundle = FHIRBundle(**data)
        observations = bundle.get_observations()

        logger.debug(
            "Fetched {} observations for Patient/{} (category={})",
            len(observations), patient_id, category or "all",
        )
        return observations

    async def get_vitals(
        self,
        patient_id: str,
        count: int = 50,
    ) -> list[FHIRObservation]:
        """Fetch vital signs observations for a patient."""
        return await self.get_observations(
            patient_id=patient_id,
            category="vital-signs",
            count=count,
        )

    async def get_labs(
        self,
        patient_id: str,
        count: int = 50,
    ) -> list[FHIRObservation]:
        """Fetch laboratory observations for a patient."""
        return await self.get_observations(
            patient_id=patient_id,
            category="laboratory",
            count=count,
        )

    async def check_connectivity(self) -> dict[str, Any]:
        """
        Check FHIR server connectivity via metadata endpoint.

        Returns:
            Dict with connectivity status and server info
        """
        try:
            data = await self._request("GET", "/metadata")
            return {
                "connected": True,
                "fhir_version": data.get("fhirVersion", "unknown"),
                "server_name": data.get("software", {}).get("name", "unknown"),
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
            }
