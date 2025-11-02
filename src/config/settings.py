"""
Application settings and configuration.

This module uses Pydantic Settings for environment variable management
and configuration validation.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables with the
    same name. Example: APP_NAME="My API" in .env file.
    """

    # Application Info
    APP_NAME: str = "Unified Embedding API - Dense & Sparse Embedding"
    VERSION: str = "3.5.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 7860
    WORKERS: int = 1
    RELOAD: bool = False  # Auto-reload on code changes (dev only)

    # Model Configuration
    MODEL_CONFIG_PATH: str = "src/config/models.yaml"
    MODEL_CACHE_DIR: str = "./model_cache"
    PRELOAD_MODELS: bool = True  # Load all models at startup

    # Request Limits
    MAX_TEXT_LENGTH: int = 8192  # Maximum characters per text
    MAX_BATCH_SIZE: int = 100  # Maximum texts per batch request
    REQUEST_TIMEOUT: int = 30  # Request timeout in seconds

    # Cache Configuration
    ENABLE_CACHE: bool = False  # Enable response caching (Phase 2)
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds
    CACHE_MAX_SIZE: int = 1000  # Maximum cache entries

    # Logging
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE: bool = False  # Write logs to file
    LOG_DIR: str = "logs"

    # CORS (if needed for web frontends)
    CORS_ENABLED: bool = False
    CORS_ORIGINS: list[str] = ["*"]

    # Model Settings
    TRUST_REMOTE_CODE: bool = True  # For models requiring remote code

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra fields in .env
    )

    @property
    def model_config_file(self) -> Path:
        """Get Path object for model configuration file."""
        return Path(self.MODEL_CONFIG_PATH)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"

    def validate_paths(self) -> None:
        """
        Validate that required paths exist.

        Raises:
            FileNotFoundError: If model config file is not found
        """
        if not self.model_config_file.exists():
            raise FileNotFoundError(
                f"Model configuration file not found: {self.MODEL_CONFIG_PATH}"
            )

        # Create cache directory if it doesn't exist
        Path(self.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

        # Create log directory if logging to file
        if self.LOG_FILE:
            Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once
    and reused across the application.

    Returns:
        Settings instance
    """
    settings = Settings()
    settings.validate_paths()
    return settings
