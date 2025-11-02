"""
Centralized logging configuration for the application.

This module configures loguru logger with appropriate settings for
development and production environments.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    level: str = "INFO", log_file: bool = False, log_dir: str = "logs"
) -> None:
    """
    Configure the application logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Whether to write logs to file
        log_dir: Directory for log files
    """
    # Remove default handler
    logger.remove()

    # Console handler with custom format
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # File handler (optional)
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        logger.add(
            log_path / "app_{time:YYYY-MM-DD}.log",
            rotation="00:00",  # Rotate at midnight
            retention="7 days",  # Keep logs for 7 days
            compression="zip",  # Compress old logs
            level=level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
        )

    logger.info(f"Logger initialized with level: {level}")


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)
