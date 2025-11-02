"""
Request schemas for API endpoints.

This module defines all Pydantic models for incoming API requests,
with validation and documentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from .common import EmbeddingOptions


class BaseEmbedRequest(BaseModel):
    """
    Base class for embedding requests.

    Attributes:
        model_id: Identifier of the model to use
        prompt: Optional instruction prompt for instruction-based models
        options: Optional embedding parameters (normalize, batch_size, etc.)
    """

    model_id: str = Field(
        ...,
        description="Model identifier to use for embedding generation",
        examples=["qwen3-0.6b", "splade-pp-v2"],
    )
    prompt: Optional[str] = Field(
        None, description="Optional instruction prompt for the model", max_length=512
    )
    options: Optional[EmbeddingOptions] = Field(
        None, description="Optional embedding generation parameters"
    )

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Validate that model_id is not empty."""
        if not v or not v.strip():
            raise ValueError("model_id cannot be empty")
        return v.strip()

    model_config = ConfigDict(
        extra="allow"  # Allow extra fields for advanced users (passed as **kwargs)
    )


class EmbedRequest(BaseEmbedRequest):
    """
    Request model for single/batch text and sparse embedding.

    Used for /embed endpoint to process multiple texts at once.

    Attributes:
        texts: List of input texts to embed
        model_id: Identifier of the model to use
        prompt: Optional prompt for instruction-based models
    """

    texts: List[str] = Field(
        ...,
        description="List of input texts to generate embeddings for",
        min_length=1,
        max_length=100,
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate that all texts are non-empty."""
        if not v:
            raise ValueError("texts list cannot be empty")

        if len(v) > 100:
            raise ValueError(f"Batch size ({len(v)}) exceeds maximum (100)")

        # Validate each text
        validated = []
        for idx, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"texts[{idx}] must be a string")
            if not text.strip():
                raise ValueError(f"texts[{idx}] cannot be empty or whitespace only")
            if len(text) > 8192:
                raise ValueError(f"texts[{idx}] exceeds maximum length (8192)")
            validated.append(text)

        return validated

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "First document to embed",
                    "Second document to embed",
                    "Third document to embed",
                ],
                "model_id": "qwen3-0.6b",
                "prompt": "Represent this document for retrieval",
                "options": {
                    "normalize_embeddings": True,
                },
            }
        }
