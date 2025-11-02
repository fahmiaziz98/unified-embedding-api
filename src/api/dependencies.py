"""
Dependency injection for FastAPI routes.

This module provides dependency functions that can be injected into
route handlers, ensuring consistent access to shared resources.
"""

from typing import Optional
from fastapi import HTTPException, status

from src.core.manager import ModelManager


# Global instances (initialized at startup)
_model_manager: Optional[ModelManager] = None


def set_model_manager(manager: ModelManager) -> None:
    """
    Set the global model manager instance.

    Called during application startup.

    Args:
        manager: ModelManager instance
    """
    global _model_manager
    _model_manager = manager


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
