"""
Common schemas shared across request and response models.

This module contains Pydantic models used by both requests and responses,
such as SparseEmbedding and ModelInfo.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class ModelInfo(BaseModel):
    """
    Information about an available model.

    Attributes:
        id: Unique identifier for the model
        name: Full model name (e.g., Hugging Face model path)
        type: Model type ('embeddings' or 'sparse-embeddings')
        loaded: Whether the model is currently loaded in memory
    """

    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Full model name")
    type: str = Field(..., description="Model type (embeddings or sparse-embeddings)")
    loaded: bool = Field(..., description="Whether model is loaded in memory")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "qwen3-0.6b",
                "name": "Qwen/Qwen3-Embedding-0.6B",
                "type": "embeddings",
                "loaded": True,
            }
        }


class HealthStatus(BaseModel):
    """
    Health check status information.

    Attributes:
        status: Overall status (ok or error)
        total_models: Total number of configured models
        loaded_models: Number of models currently loaded
        startup_complete: Whether startup sequence is complete
    """

    status: str = Field(..., description="Overall status")
    total_models: int = Field(..., description="Total configured models")
    loaded_models: int = Field(..., description="Currently loaded models")
    startup_complete: bool = Field(..., description="Startup completion status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "total_models": 2,
                "loaded_models": 2,
                "startup_complete": True,
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response format.

    Attributes:
        error: Error type/name
        message: Detailed error message
        detail: Additional error details (optional)
    """

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ModelNotFoundError",
                "message": "Model 'unknown-model' not found in configuration",
                "detail": "Available models: qwen3-0.6b, splade-pp-v2",
            }
        }


class EmbeddingOptions(BaseModel):
    """
    Optional parameters for embedding generation.

    These parameters are passed directly to the underlying sentence-transformers
    model. Not all parameters work with all models - check model documentation.

    Attributes:
        normalize_embeddings: L2 normalize output embeddings
        prompt: Optional instruction prompt for the model
        batch_size: Batch size for processing
        convert_to_numpy: Return numpy arrays instead of lists
        precision: Computation precision
    """

    normalize_embeddings: Optional[bool] = Field(
        None, description="L2 normalize the output embeddings"
    )
    batch_size: Optional[int] = Field(
        None, ge=1, le=256, description="Batch size for processing texts"
    )
    prompt: Optional[str] = Field(
        None, description="Optional instruction prompt for the model", max_length=512
    )
    convert_to_numpy: Optional[bool] = Field(
        None, description="Return numpy arrays instead of Python lists"
    )

    precision: Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]] = (
        Field(None, description="Computation precision for embeddings")
    )

    model_config = ConfigDict(
        extra="forbid",  # Prevent typos in field names
        json_schema_extra={
            "example": {
                "normalize_embeddings": True,
                "batch_size": 32,
            }
        },
    )

    def to_kwargs(self) -> dict:
        """
        Convert options to kwargs dict, excluding None values.

        Returns:
            Dictionary of non-None parameters
        """
        return {k: v for k, v in self.model_dump().items() if v is not None}
