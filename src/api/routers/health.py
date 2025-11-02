"""
Health check and system information endpoints.

This module provides routes for monitoring the API health
and getting system information.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends
from loguru import logger

from src.models.schemas import RootResponse, HealthStatus
from src.core.manager import ModelManager
from src.core.cache import EmbeddingCache
from src.api.dependencies import get_model_manager, get_cache_if_enabled
from src.config.settings import get_settings


router = APIRouter(tags=["system"])


@router.get(
    "/",
    response_model=RootResponse,
    summary="API root",
    description="Get basic API information",
)
async def root(settings=Depends(get_settings)):
    """
    API root endpoint.

    Returns basic information about the API including
    version and documentation URL.

    Args:
        settings: Application settings

    Returns:
        RootResponse with API information
    """
    return RootResponse(
        message=f"{settings.APP_NAME} - Dense & Sparse Embeddings",
        version=settings.VERSION,
        docs_url="/docs",
    )


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Health check",
    description="Check API health status",
)
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """
    Health check endpoint.

    Returns the current health status of the API including:
    - Overall status (ok/error)
    - Total configured models
    - Currently loaded models
    - Startup completion status

    Args:
        manager: Model manager dependency

    Returns:
        HealthStatus with health information
    """
    try:
        memory_info = manager.get_memory_usage()

        return HealthStatus(
            status="ok",
            total_models=memory_info["total_available"],
            loaded_models=memory_info["loaded_count"],
            startup_complete=memory_info["preload_complete"],
        )

    except Exception:
        logger.exception("Health check failed")
        return HealthStatus(
            status="error", total_models=0, loaded_models=0, startup_complete=False
        )


@router.get(
    "/info",
    summary="System information",
    description="Get detailed system and cache information",
)
async def system_info(
    manager: ModelManager = Depends(get_model_manager),
    cache: EmbeddingCache = Depends(get_cache_if_enabled),
    settings=Depends(get_settings),
) -> Dict[str, Any]:
    """
    Get detailed system information.

    Returns comprehensive information about:
    - API configuration
    - Model statistics
    - Cache statistics (if enabled)
    - System settings

    Args:
        manager: Model manager dependency
        cache: Cache dependency (if enabled)
        settings: Application settings

    Returns:
        Dictionary with system information
    """
    try:
        memory_info = manager.get_memory_usage()

        info = {
            "api": {
                "name": settings.APP_NAME,
                "version": settings.VERSION,
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG,
            },
            "models": {
                "total": memory_info["total_available"],
                "loaded": memory_info["loaded_count"],
                "preload_complete": memory_info["preload_complete"],
                "loaded_models": memory_info["loaded_models"],
            },
            "limits": {
                "max_text_length": settings.MAX_TEXT_LENGTH,
                "max_batch_size": settings.MAX_BATCH_SIZE,
                "request_timeout": settings.REQUEST_TIMEOUT,
            },
            "cache": {"enabled": settings.ENABLE_CACHE},
        }

        # Add cache stats if enabled
        if cache is not None:
            info["cache"]["stats"] = cache.stats

        return info

    except Exception as e:
        logger.exception("Failed to get system info")
        return {"error": "Failed to retrieve system information", "detail": str(e)}


@router.get(
    "/cache/stats",
    summary="Cache statistics",
    description="Get detailed cache statistics (if caching is enabled)",
)
async def cache_stats(
    cache: EmbeddingCache = Depends(get_cache_if_enabled),
) -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns detailed information about the embedding cache including:
    - Current size
    - Hit/miss counts
    - Hit rate
    - TTL configuration

    Args:
        cache: Cache dependency (if enabled)

    Returns:
        Dictionary with cache statistics or disabled message
    """
    if cache is None:
        return {"enabled": False, "message": "Caching is disabled"}

    try:
        return {"enabled": True, **cache.stats}
    except Exception as e:
        logger.exception("Failed to get cache stats")
        return {"enabled": True, "error": str(e)}


@router.post(
    "/cache/clear",
    summary="Clear cache",
    description="Clear all cached embeddings (if caching is enabled)",
)
async def clear_cache(
    cache: EmbeddingCache = Depends(get_cache_if_enabled),
) -> Dict[str, Any]:
    """
    Clear all cached embeddings.

    This endpoint removes all entries from the cache, forcing
    fresh computation for subsequent requests.

    Args:
        cache: Cache dependency (if enabled)

    Returns:
        Dictionary with operation status
    """
    if cache is None:
        return {"success": False, "message": "Caching is disabled"}

    try:
        cache.clear()
        return {"success": True, "message": "Cache cleared successfully"}
    except Exception as e:
        logger.exception("Failed to clear cache")
        return {"success": False, "error": str(e)}
