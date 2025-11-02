"""
Dependency injection for FastAPI routes.

This module provides dependency functions that can be injected into
route handlers, ensuring consistent access to shared resources.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status

from src.config.settings import Settings, get_settings
from src.core.manager import ModelManager
from src.core.cache import EmbeddingCache


# Global instances (initialized at startup)
_model_manager: Optional[ModelManager] = None
_embedding_cache: Optional[EmbeddingCache] = None


def set_model_manager(manager: ModelManager) -> None:
    """
    Set the global model manager instance.

    Called during application startup.

    Args:
        manager: ModelManager instance
    """
    global _model_manager
    _model_manager = manager


def set_embedding_cache(cache: EmbeddingCache) -> None:
    """
    Set the global embedding cache instance.

    Called during application startup if caching is enabled.

    Args:
        cache: EmbeddingCache instance
    """
    global _embedding_cache
    _embedding_cache = cache


def get_model_manager() -> ModelManager:
    """
    Get the model manager instance.

    This is a dependency function that can be used in route handlers.

    Returns:
        ModelManager instance

    Raises:
        HTTPException: If model manager is not initialized

    Example:
        @router.post("/embed")
        async def embed(
            request: EmbedRequest,
            manager: ModelManager = Depends(get_model_manager)
        ):
            model = manager.get_model(request.model_id)
            ...
    """
    if _model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is not ready. Model manager not initialized.",
        )
    return _model_manager


def get_embedding_cache() -> Optional[EmbeddingCache]:
    """
    Get the embedding cache instance (if enabled).

    Returns:
        EmbeddingCache instance or None if caching is disabled
    """
    return _embedding_cache


def get_cache_if_enabled(
    settings: Settings = Depends(get_settings),
) -> Optional[EmbeddingCache]:
    """
    Get cache only if caching is enabled in settings.

    Args:
        settings: Application settings

    Returns:
        EmbeddingCache instance if enabled, None otherwise
    """
    if settings.ENABLE_CACHE:
        return _embedding_cache
    return None
