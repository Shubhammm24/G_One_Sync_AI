"""
G_One_Sync AI — JWT Authentication Middleware
================================================
Provides optional JWT-based authentication for the mobile companion
app. When ``auth_enabled`` is False (default), all requests pass
through without any token validation — preserving backward
compatibility with the existing unauthenticated serving API.

Security Notes
--------------
* JWT secret **must** be set via ``JEEVAN_MOBILE_JWT_SECRET`` env var.
* Tokens carry ``sub`` (physician username) and ``exp`` (expiry).
* Refresh tokens have a longer TTL (default 7 days).
* This module does NOT manage user persistence — physicians are
  validated against a simple credential store (demo mode) or an
  external identity provider in production.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel

from config.settings import mobile_settings

# ── Bearer token extractor (optional) ───────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


# ── Pydantic Schemas ────────────────────────────────────────────────────


class TokenPayload(BaseModel):
    """Decoded JWT payload."""

    sub: str  # Physician username / ID
    exp: float  # Expiry timestamp (Unix epoch)
    iat: float  # Issued-at timestamp
    token_type: str = "access"  # "access" or "refresh"


class LoginRequest(BaseModel):
    """Login request body."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token pair returned on successful auth."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until access token expires


class RefreshRequest(BaseModel):
    """Refresh token request body."""

    refresh_token: str


# ── Token Creation ──────────────────────────────────────────────────────


def create_access_token(username: str) -> str:
    """Create a signed JWT access token.

    Args:
        username: Physician identifier to embed as ``sub`` claim.

    Returns:
        Encoded JWT string.
    """
    import jwt

    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": username,
        "iat": now.timestamp(),
        "exp": (now + timedelta(hours=mobile_settings.jwt_expiry_hours)).timestamp(),
        "token_type": "access",
    }
    token = jwt.encode(
        payload,
        mobile_settings.jwt_secret,
        algorithm=mobile_settings.jwt_algorithm,
    )
    logger.debug("Access token created for user={}", username)
    return token


def create_refresh_token(username: str) -> str:
    """Create a longer-lived refresh token.

    Args:
        username: Physician identifier to embed as ``sub`` claim.

    Returns:
        Encoded JWT string.
    """
    import jwt

    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": username,
        "iat": now.timestamp(),
        "exp": (now + timedelta(hours=mobile_settings.jwt_refresh_expiry_hours)).timestamp(),
        "token_type": "refresh",
    }
    token = jwt.encode(
        payload,
        mobile_settings.jwt_secret,
        algorithm=mobile_settings.jwt_algorithm,
    )
    logger.debug("Refresh token created for user={}", username)
    return token


# ── Token Verification ──────────────────────────────────────────────────


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token.

    Args:
        token: Raw JWT string.

    Returns:
        Parsed TokenPayload.

    Raises:
        HTTPException: If the token is expired, malformed, or invalid.
    """
    import jwt

    try:
        payload = jwt.decode(
            token,
            mobile_settings.jwt_secret,
            algorithms=[mobile_settings.jwt_algorithm],
        )
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Credential Validation ──────────────────────────────────────────────


def validate_credentials(username: str, password: str) -> bool:
    """Validate physician credentials.

    In demo mode (empty ``demo_password_hash``), any password is accepted
    for the demo username. In production, integrate with hospital LDAP/SSO.

    Args:
        username: Physician username.
        password: Plaintext password.

    Returns:
        True if credentials are valid.
    """
    # Demo mode: accept any password when hash is empty
    if not mobile_settings.demo_password_hash:
        logger.warning("Auth running in demo mode — accepting any password for '{}'", username)
        return username == mobile_settings.demo_username

    # Production mode: verify bcrypt hash
    try:
        import bcrypt

        stored_hash = mobile_settings.demo_password_hash.encode("utf-8")
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash)
    except ImportError:
        logger.error("bcrypt not installed — cannot verify password hash")
        return False
    except Exception as e:
        logger.error("Password verification failed: {}", e)
        return False


# ── FastAPI Dependency ──────────────────────────────────────────────────


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[TokenPayload]:
    """FastAPI dependency that enforces JWT auth when enabled.

    When ``auth_enabled`` is False, returns None (all requests pass).
    When ``auth_enabled`` is True, validates the Bearer token and
    returns the decoded payload.

    Args:
        request: The incoming FastAPI request.
        credentials: Extracted Bearer token (if present).

    Returns:
        TokenPayload if auth is enabled and valid, else None.
    """
    if not mobile_settings.auth_enabled:
        return None

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)

    # Ensure it's an access token, not a refresh token
    if payload.token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — expected access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
