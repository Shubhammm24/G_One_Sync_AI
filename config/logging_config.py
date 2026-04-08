"""
JeevanSync AI — Logging Configuration
======================================
Structured logging via loguru with console + file sinks.
"""

import sys
from pathlib import Path
from loguru import logger

# ── Log directory ────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure loguru with two sinks:
    1. Console — colored, human-readable
    2. File   — JSON-structured, rotated daily, 30-day retention
    """
    # Remove default handler
    logger.remove()

    # Console sink
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File sink — JSON structured
    logger.add(
        LOG_DIR / "jeevansync_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DDTHH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation="00:00",       # rotate at midnight
        retention="30 days",    # keep 30 days
        compression="gz",       # compress old logs
        serialize=True,         # JSON structured
    )

    logger.info("Logging initialized — level={}", level)


# Auto-initialize on import
setup_logging()
