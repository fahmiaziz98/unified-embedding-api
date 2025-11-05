"""
Model management endpoints.

This module provides routes for listing and inspecting
available embedding models.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Path, status
from loguru import logger

from src.models.schemas import ModelsListResponse, ModelInfo
from src.core.manager import ModelManager
from src.core.exceptions import ModelNotFoundError
from src.api.dependencies import get_model_manager


router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "",
    response_model=ModelsListResponse,
    summary="List all models",
    description="Get a list of all available embedding models",
)
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """
    List all available embedding models.

    Returns information about all configured models including:
    - Model ID
    - Model name (Hugging Face path)
    - Model type (dense or sparse)
    - Load status
    - Repository URL

    Args:
        manager: Model manager dependency

    Returns:
        ModelsListResponse with list of models and total count
    """
    try:
        models_info = manager.list_models()

        return ModelsListResponse(
            models=[ModelInfo(**info) for info in models_info], total=len(models_info)
        )

    except Exception:
        logger.exception("Failed to list models")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list models",
        )


@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    summary="Get model info",
    description="Get detailed information about a specific model",
)
async def get_model_info(
    model_id: str = Path(..., description="Model identifier"),
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Get detailed information about a specific model.

    Args:
        model_id: The model identifier
        manager: Model manager dependency

    Returns:
        ModelInfo with model details

    Raises:
        HTTPException: If model not found
    """
    try:
        info = manager.get_model_info(model_id)

        if not info:
            raise ModelNotFoundError(model_id)

        return ModelInfo(**info)

    except ModelNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception(f"Failed to get info for model {model_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@router.get(
    "/{model_id}/load",
    summary="Load a model",
    description="Manually trigger loading of a specific model",
)
async def load_model(
    model_id: str = Path(..., description="Model identifier"),
    manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Manually load a specific model into memory.

    This endpoint is useful for preloading models on-demand
    without waiting for the first request.

    Args:
        model_id: The model identifier
        manager: Model manager dependency

    Returns:
        Dictionary with load status

    Raises:
        HTTPException: If model not found or fails to load
    """
    try:
        # This will load the model if not already loaded
        model = manager.get_model(model_id)

        return {
            "status": "success",
            "message": f"Model '{model_id}' loaded successfully",
            "model_id": model_id,
            "loaded": model.is_loaded,
        }

    except ModelNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception(f"Failed to load model {model_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )
