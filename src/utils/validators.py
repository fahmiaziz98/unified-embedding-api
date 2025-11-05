"""
Input validation utilities.

This module provides validation functions for request inputs,
ensuring data quality and preventing abuse.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
from src.core.exceptions import (
    TextTooLongError,
    ValidationError,
    ModelNotFoundError,
    ModelTypeError,
)


def validate_text(text: str, max_length: int = 8192, allow_empty: bool = False) -> None:
    """
    Validate a single text input.

    Args:
        text: Input text to validate
        max_length: Maximum allowed text length
        allow_empty: Whether to allow empty strings

    Raises:
        ValidationError: If text is empty and not allowed
        TextTooLongError: If text exceeds max_length
    """
    if not allow_empty and not text.strip():
        raise ValidationError("text", "Text cannot be empty")

    if len(text) > max_length:
        raise TextTooLongError(len(text), max_length)


def ensure_model_type(config, expected_type: str, model_id: str) -> None:
    """
    Validate that the model configuration matches the expected type.

    Raises:
        HTTPException: If the model is missing or the type does not match.
    """
    if config is None:
        raise ModelNotFoundError(model_id)

    if config.type != expected_type:
        raise ModelTypeError(config, model_id, expected_type)


def validate_model_id(model_id: str, available_models: List[str]) -> None:
    """
    Validate that a model_id exists in available models.

    Args:
        model_id: Model identifier to validate
        available_models: List of available model IDs

    Raises:
        ValidationError: If model_id is invalid
    """
    if not model_id:
        raise ValidationError("model_id", "Model ID cannot be empty")

    if model_id not in available_models:
        raise ValidationError(
            "model_id",
            f"Model '{model_id}' not found. Available: {', '.join(available_models)}",
        )


def extract_embedding_kwargs(request: BaseModel) -> Dict[str, Any]:
    """
    Extract embedding kwargs from a request object.

    This function extracts both the 'options' field and any extra fields
    passed in the request, combining them into a single kwargs dict.

    Args:
        request: Pydantic request model (EmbedRequest or BatchEmbedRequest)

    Returns:
        Dictionary of kwargs to pass to embedding model

    Example:
        >>> request = EmbedRequest(
        ...     texts=["hello"],
        ...     model_id="qwen3-0.6b",
        ...     options=EmbeddingOptions(normalize_embeddings=True),
        ...     batch_size=32  # Extra field
        ... )
        >>> extract_embedding_kwargs(request)
        {'normalize_embeddings': True, 'batch_size': 32}
    """
    kwargs = {}

    # Extract from 'options' field if present
    if hasattr(request, "options") and request.options is not None:
        kwargs.update(request.options.to_kwargs())

    # Extract extra fields (excluding standard fields)
    standard_fields = {
        "input",
        "model",
        "encoding_format",
        "dimensions",
        "user",
        "options",
        "query",
        "documents",
        "top_k",
    }
    request_dict = request.model_dump()

    for key, value in request_dict.items():
        if key not in standard_fields and value is not None:
            kwargs[key] = value

    return kwargs


def estimate_tokens(text: str) -> int:
    """Estimate token count (simple approximation)."""
    # Simple heuristic: ~4 characters per token
    return max(1, len(text) // 4)


def count_tokens_batch(texts: List[str]) -> int:
    """Count tokens for batch of texts."""
    return sum(estimate_tokens(text) for text in texts)
