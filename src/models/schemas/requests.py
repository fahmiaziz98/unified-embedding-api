"""
Request schemas for API endpoints.

This module defines all Pydantic models for incoming API requests,
with validation and documentation.
"""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from .common import EmbeddingOptions


class BaseEmbedRequest(BaseModel):
    """
    OpenAI-compatible embedding request.

    Matches the format of OpenAI's embeddings API:
    https://platform.openai.com/docs/api-reference/embeddings
    """

    model: str = Field(
        ...,
        description="Model identifier to use for embedding generation",
        examples=["qwen3-0.6b", "splade-pp-v2"],
    )

    encoding_format: Optional[Literal["float", "base64"]] = Field(
        default="float", description="Encoding format"
    )
    dimensions: Optional[int] = Field(None, description="Output dimensions")

    user: Optional[str] = Field(None, description="User identifier")

    options: Optional[EmbeddingOptions] = Field(
        None, description="Optional embedding generation parameters"
    )

    @field_validator("model")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Validate that model is not empty."""
        if not v or not v.strip():
            raise ValueError("model cannot be empty")
        return v.strip()

    model_config = ConfigDict(
        extra="allow"  # Allow extra fields for advanced users (passed as **kwargs)
    )


class EmbedRequest(BaseEmbedRequest):
    """
    Request model for single/batch text and sparse embedding.

    Used for /embeddings and /embed_sparse endpoint to process multiple texts at once.

    Attributes:
        input: Str or List of input texts to embed
    """

    input: Union[str, List[str]] = Field(
        ...,
        description="Str or List of input texts to generate embeddings for",
        min_length=1,
    )

    @field_validator("input")
    @classmethod
    def validate_texts(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate that all texts are non-empty."""
        if not v:
            raise ValueError("Input cannot be empty")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "input": [
                    "First document to embed",
                    "Second document to embed",
                    "Third document to embed",
                ],
                "model": "qwen3-0.6b",
                "options": {
                    "normalize_embeddings": True,
                },
            }
        }


class RerankRequest(BaseEmbedRequest):
    """
    Request model for document reranking.

    Attributes:
        query: The search query
        documents: List of documents to rerank
        top_k: Maximum number of documents to return

    """

    query: str = Field(..., description="Search query text")
    documents: List[str] = Field(
        ..., min_items=1, description="List of documents to rerank"
    )
    top_k: int = Field(..., description="Maximum number of results to return")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "bge-v2-m3",
                "query": "Python best programming languages for data science",
                "top_k": 4,
                "documents": [
                    "Python is a popular language for data science due to its extensive libraries.",
                    "R is widely used in statistical computing and data analysis.",
                    "Java is a versatile language used in various applications, including data science.",
                    "SQL is essential for managing and querying relational databases.",
                    "Julia is a high-performance language gaining popularity for numerical computing and data science.",
                ],
            }
        }
