"""
Abstract base classes for embedding models.

This module defines the interface that all embedding model implementations
must follow, ensuring consistency across dense and sparse embeddings.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for all embedding models.

    All embedding model implementations (dense, sparse, etc.) must inherit
    from this class and implement all abstract methods.

    Attributes:
        config: ModelConfig instance containing model metadata
        _loaded: Flag indicating if the model is currently loaded in memory
    """

    def __init__(self, config: Any):
        """
        Initialize the embedding model.

        Args:
            config: ModelConfig instance with model configuration
        """
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.

        This method should:
        - Check if already loaded (idempotent)
        - Initialize the underlying model
        - Set _loaded flag to True
        - Handle errors gracefully

        Raises:
            RuntimeError: If model fails to load
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory and free resources.

        This method should:
        - Release model from memory
        - Clear any caches
        - Set _loaded flag to False
        - Be safe to call multiple times
        """
        pass

    @abstractmethod
    def embed_query(
        self, texts: List[str], prompt: Optional[str] = None, **kwargs
    ) -> Union[List[List[float]], List[Dict[str, Any]]]:
        """
        Generate embeddings for query texts.

        Query embeddings may differ from document embeddings in some models
        (e.g., asymmetric retrieval models).

        Args:
            texts: List of query texts to embed (REQUIRED)
            prompt: Optional instruction prompt for the model
            **kwargs: Additional model-specific parameters, such as:
                - normalize_embeddings (bool): L2 normalize output vectors
                - batch_size (int): Batch size for processing
                - max_length (int): Maximum token sequence length
                - convert_to_numpy (bool): Return numpy arrays instead of lists
                - precision (str): Computation precision ('float32', 'int8', etc.)

        Returns:
            List of embeddings (format depends on model type)
            - Dense: List[List[float]]
            - Sparse: List[Dict[str, Any]] with 'indices' and 'values'

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input validation fails

        Note:
            Available kwargs depend on the underlying model implementation.
            Check sentence-transformers documentation for full parameter list.
        """
        pass

    @abstractmethod
    def embed_documents(
        self, texts: List[str], prompt: Optional[str] = None, **kwargs
    ) -> Union[List[List[float]], List[Dict[str, Any]]]:
        """
        Generate embeddings for document texts.

        Document embeddings are used for indexing and storage.

        Args:
            texts: List of document texts to embed (REQUIRED)
            prompt: Optional instruction prompt for the model
            **kwargs: Additional model-specific parameters (see embed_query for details)

        Returns:
            List of embeddings (format depends on model type)
            - Dense: List[List[float]]
            - Sparse: List[Dict[str, Any]] with 'indices' and 'values'

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input validation fails

        Note:
            Available kwargs depend on the underlying model implementation.
            Check sentence-transformers documentation for full parameter list.
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._loaded

    @property
    def model_id(self) -> str:
        """
        Get the model identifier.

        Returns:
            Model ID string
        """
        return self.config.id

    @property
    def model_type(self) -> str:
        """
        Get the model type.

        Returns:
            Model type ('embeddings' or 'sparse-embeddings')
        """
        return self.config.type

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.model_id}, "
            f"type={self.model_type}, "
            f"loaded={self.is_loaded})"
        )
