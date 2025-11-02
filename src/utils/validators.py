"""
Input validation utilities.

This module provides validation functions for request inputs,
ensuring data quality and preventing abuse.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
from src.core.exceptions import TextTooLongError, BatchTooLargeError, ValidationError


def validate_texts(
    texts: List[str],
    max_length: int = 8192,
    max_batch_size: int = 100,
    allow_empty: bool = False,
) -> None:
    """
    Validate a list of text inputs.

    Args:
        texts: List of texts to validate
        max_length: Maximum allowed length per text
        max_batch_size: Maximum number of texts in batch
        allow_empty: Whether to allow empty strings

    Raises:
        ValidationError: If texts list is empty or contains invalid items
        BatchTooLargeError: If batch size exceeds max_batch_size
        TextTooLongError: If any text exceeds max_length
    """
    if not texts:
        raise ValidationError("texts", "Texts list cannot be empty")

    if len(texts) > max_batch_size:
        raise BatchTooLargeError(len(texts), max_batch_size)

    # Validate each text
    for idx, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValidationError(
                f"texts[{idx}]", f"Expected string, got {type(text).__name__}"
            )

        if not allow_empty and not text.strip():
            raise ValidationError(f"texts[{idx}]", "Text cannot be empty")

        if len(text) > max_length:
            raise TextTooLongError(len(text), max_length)


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


def sanitize_text(text: str, max_length: int = 8192) -> str:
    """
    Sanitize text input by removing excessive whitespace and truncating.

    Args:
        text: Input text to sanitize
        max_length: Maximum length to truncate to

    Returns:
        Sanitized text
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace multiple whitespaces with single space
    text = " ".join(text.split())

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    return text


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
        ...     text="hello",
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
    standard_fields = {"text", "texts", "model_id", "prompt", "options"}
    request_dict = request.model_dump()

    for key, value in request_dict.items():
        if key not in standard_fields and value is not None:
            kwargs[key] = value

    return kwargs
