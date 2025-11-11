"""
Response schemas for API endpoints.

This module defines all Pydantic models for API responses,
ensuring consistent output format across all endpoints.
"""

from typing import List, Literal
from pydantic import BaseModel, Field
from .common import ModelInfo


class BaseEmbedResponse(BaseModel):
    """
    Base class for embedding responses.

    Attributes:
        model_id: Identifier of the model used
    """

    model: str = Field(..., description="Model identifier used")


class EmbeddingObject(BaseModel):
    """Single embedding object."""

    object: Literal["embedding"] = "embedding"
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index of the embedding")


class TokenUsage(BaseModel):
    """Usage statistics."""

    prompt_tokens: int
    total_tokens: int


class DenseEmbedResponse(BaseEmbedResponse):
    """
    Response model for single/batch dense embeddings.

    Used for /embeddings endpoint dense models.

    Attributes:
        data: List of generated dense embeddings
        model: Identifier of the model used
        usage: Usage statistics

    """

    object: Literal["list"] = "list"
    data: List[EmbeddingObject]
    model: str = Field(..., description="Model identifier used")
    usage: TokenUsage = Field(..., description="Usage statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                    {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
                ],
                "model": "qwen3-0.6b",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
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
