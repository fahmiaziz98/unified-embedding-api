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
from src.api.dependencies import get_model_manager
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
    settings=Depends(get_settings),
) -> Dict[str, Any]:
    """
    Get detailed system information.

    Returns comprehensive information about:
    - API configuration
    - Model statistics
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
        }

        return info

    except Exception as e:
        logger.exception("Failed to get system info")
        return {"error": "Failed to retrieve system information", "detail": str(e)}
