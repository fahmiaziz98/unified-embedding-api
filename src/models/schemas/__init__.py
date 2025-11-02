"""
Pydantic schemas for API requests and responses.

This package exports all schema models for easy importing throughout
the application.
"""

from .common import (
    SparseEmbedding,
    ModelInfo,
    HealthStatus,
    ErrorResponse,
    EmbeddingOptions,
)

from .requests import BaseEmbedRequest, EmbedRequest, BatchEmbedRequest

from .responses import (
    BaseEmbedResponse,
    DenseEmbedResponse,
    SparseEmbedResponse,
    BatchDenseEmbedResponse,
    BatchSparseEmbedResponse,
    ModelsListResponse,
    RootResponse,
)

__all__ = [
    # Common
    "SparseEmbedding",
    "ModelInfo",
    "HealthStatus",
    "ErrorResponse",
    "EmbeddingOptions",
    # Requests
    "BaseEmbedRequest",
    "EmbedRequest",
    "BatchEmbedRequest",
    # Responses
    "BaseEmbedResponse",
    "DenseEmbedResponse",
    "SparseEmbedResponse",
    "BatchDenseEmbedResponse",
    "BatchSparseEmbedResponse",
    "ModelsListResponse",
    "RootResponse",
]
