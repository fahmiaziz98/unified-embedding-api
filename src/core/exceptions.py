"""
Custom exceptions for the Unified Embedding API.

This module defines all custom exceptions used throughout the application,
providing clear error messages and proper HTTP status code mapping.
"""


class EmbeddingAPIException(Exception):
    """Base exception for all API errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotFoundError(EmbeddingAPIException):
    """Raised when a requested model is not found in configuration."""

    def __init__(self, model_id: str):
        message = f"Model '{model_id}' not found in configuration"
        super().__init__(message, status_code=404)
        self.model_id = model_id


class ModelNotLoadedError(EmbeddingAPIException):
    """Raised when attempting to use a model that is not loaded."""

    def __init__(self, model_id: str):
        message = f"Model '{model_id}' is not loaded. Please wait for initialization."
        super().__init__(message, status_code=503)
        self.model_id = model_id


class ModelTypeError(EmbeddingAPIException):
    """Raise when model not configuration"""

    def __init__(self, config, model_id: str, expected_type: str):
        message = (
            f"Model '{model_id}' is not a {expected_type.replace('-', ' ')} "
            f"model. Detected type: {config.type}"
        )
        super().__init__(message, status_code=404)
        self.model_id = model_id


class ModelLoadError(EmbeddingAPIException):
    """Raised when a model fails to load."""

    def __init__(self, model_id: str, reason: str):
        message = f"Failed to load model '{model_id}': {reason}"
        super().__init__(message, status_code=500)
        self.model_id = model_id
        self.reason = reason


class ConfigurationError(EmbeddingAPIException):
    """Raised when there's an error in the configuration file."""

    def __init__(self, reason: str):
        message = f"Configuration error: {reason}"
        super().__init__(message, status_code=500)
        self.reason = reason


class ValidationError(EmbeddingAPIException):
    """Raised when input validation fails."""

    def __init__(self, field: str, reason: str):
        message = f"Validation error for '{field}': {reason}"
        super().__init__(message, status_code=400)
        self.field = field
        self.reason = reason


class BatchTooLargeError(ValidationError):
    """Raised when batch size exceeds maximum allowed."""

    def __init__(self, batch_size: int, max_batch_size: int):
        reason = f"Batch size ({batch_size}) exceeds maximum ({max_batch_size})"
        super().__init__("texts", reason)
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size


class EmbeddingGenerationError(EmbeddingAPIException):
    """Raised when embedding generation fails."""

    def __init__(self, model_id: str, reason: str):
        message = f"Failed to generate embedding with model '{model_id}': {reason}"
        super().__init__(message, status_code=500)
        self.model_id = model_id
        self.reason = reason


class RerankingDocumentError(EmbeddingAPIException):
    """Raised when reranking document fails."""

    def __init__(self, model_id: str, reason: str):
        message = f"Failed to reranking document with model '{model_id}': {reason}"
        super().__init__(message, status_code=500)
        self.model_id = model_id
        self.reason = reason


class ServerNotReadyError(EmbeddingAPIException):
    """Raised when server is not fully initialized."""

    def __init__(self):
        message = "Server is not ready. Please wait for model initialization."
        super().__init__(message, status_code=503)
