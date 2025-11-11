"""
Pydantic schemas for API requests and responses.

This package exports all schema models for easy importing throughout
the application.
"""

from .common import (
    ModelInfo,
    HealthStatus,
    ErrorResponse,
    EmbeddingOptions,
)

from .requests import BaseEmbedRequest, EmbedRequest, RerankRequest

from .responses import (
    BaseEmbedResponse,
    DenseEmbedResponse,
    EmbeddingObject,
    TokenUsage,
    ModelsListResponse,
    RootResponse,
)

__all__ = [
    # Common
    "ModelInfo",
    "HealthStatus",
    "ErrorResponse",
    "EmbeddingOptions",
    # Requests
    "BaseEmbedRequest",
    "EmbedRequest",
    "RerankRequest",
    # Responses
    "BaseEmbedResponse",
    "DenseEmbedResponse",
    "EmbeddingObject",
    "TokenUsage",
    "ModelsListResponse",
    "RootResponse",
]
