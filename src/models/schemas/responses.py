"""
Response schemas for API endpoints.

This module defines all Pydantic models for API responses,
ensuring consistent output format across all endpoints.
"""

from typing import List
from pydantic import BaseModel, Field
from .common import SparseEmbedding, ModelInfo


class BaseEmbedResponse(BaseModel):
    """
    Base class for embedding responses.

    Attributes:
        model_id: Identifier of the model used
        processing_time: Time taken to process the request (seconds)
    """

    model_id: str = Field(..., description="Model identifier used")
    processing_time: float = Field(..., description="Processing time in seconds", ge=0)


class DenseEmbedResponse(BaseEmbedResponse):
    """
    Response model for single/batch dense embeddings.

    Used for /embed & /query endpoint with dense models.

    Attributes:
        embeddings: List of generated dense embedding vectors
        dimension: Dimensionality of the embeddings
        count: Number of embeddings returned
        model_id: Identifier of the model used
        processing_time: Time taken to process the request
    """

    embeddings: List[List[float]] = Field(
        ..., description="List of dense embedding vectors"
    )
    dimension: int = Field(..., description="Embedding dimensionality", ge=1)
    count: int = Field(..., description="Number of embeddings", ge=1)

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [
                    [0.123, -0.456, 0.789],
                    [0.234, 0.567, -0.890],
                    [0.345, -0.678, 0.901],
                ],
                "dimension": 768,
                "count": 3,
                "model_id": "qwen3-0.6b",
                "processing_time": 0.1245,
            }
        }


class SparseEmbedResponse(BaseEmbedResponse):
    """
    Response model for single/batch sparse embeddings.

    Used for /embed and /query endpoint with sparse models.

    Attributes:
        embeddings: List of generated sparse embeddings
        count: Number of embeddings returned
        model_id: Identifier of the model used
        processing_time: Time taken to process the request
    """

    embeddings: List[SparseEmbedding] = Field(
        ..., description="List of sparse embeddings"
    )
    count: int = Field(..., description="Number of embeddings", ge=1)

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [
                    {
                        "indices": [10, 25, 42],
                        "values": [0.85, 0.62, 0.91],
                        "text": "first text",
                    },
                    {
                        "indices": [15, 30, 50],
                        "values": [0.73, 0.88, 0.65],
                        "text": "second text",
                    },
                ],
                "count": 2,
                "model_id": "splade-pp-v2",
                "processing_time": 0.0892,
            }
        }


class ModelsListResponse(BaseModel):
    """
    Response model for listing available models.

    Attributes:
        models: List of available models with their info
        total: Total number of models
    """

    models: List[ModelInfo] = Field(..., description="List of available models")
    total: int = Field(..., description="Total number of models", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "id": "qwen3-0.6b",
                        "name": "Qwen/Qwen3-Embedding-0.6B",
                        "type": "embeddings",
                        "loaded": True,
                    }
                ],
                "total": 1,
            }
        }


class RootResponse(BaseModel):
    """
    Response model for root endpoint.

    Attributes:
        message: Welcome message
        version: API version
        docs_url: URL to API documentation
    """

    message: str = Field(..., description="Welcome message")
    version: str = Field(..., description="API version")
    docs_url: str = Field(..., description="Documentation URL")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Unified Embedding API - Dense & Sparse Embeddings",
                "version": "3.0.0",
                "docs_url": "/docs",
            }
        }
