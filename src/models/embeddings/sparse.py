"""
Sparse embedding model implementation.

This module provides the SparseEmbeddingModel class for generating
sparse vector embeddings (e.g., SPLADE models).
"""

from typing import List, Optional, Dict, Any
from sentence_transformers import SparseEncoder
from loguru import logger

from src.core.base import BaseEmbeddingModel
from src.core.config import ModelConfig
from src.core.exceptions import ModelLoadError, EmbeddingGenerationError


class SparseEmbeddingModel(BaseEmbeddingModel):
    """
    Sparse embedding model wrapper.

    This class wraps sentence-transformers SparseEncoder models
    to generate sparse vector embeddings (indices + values).

    Attributes:
        config: ModelConfig instance
        model: SparseEncoder instance
        _loaded: Flag indicating if the model is loaded
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the sparse embedding model.

        Args:
            config: ModelConfig instance with model configuration
        """
        super().__init__(config)
        self.model: Optional[SparseEncoder] = None

    def load(self) -> None:
        """
        Load the sparse embedding model into memory.

        Raises:
            ModelLoadError: If model fails to load
        """
        if self._loaded:
            logger.debug(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading sparse embedding model: {self.config.name}")

        try:
            self.model = SparseEncoder(self.config.name)
            self._loaded = True
            logger.success(f"✓ Loaded sparse model: {self.model_id}")

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(f"✗ {error_msg}")
            raise ModelLoadError(self.model_id, error_msg)

    def unload(self) -> None:
        """
        Unload the model from memory and free resources.
        """
        if not self._loaded:
            logger.debug(f"Model {self.model_id} not loaded, nothing to unload")
            return

        try:
            if self.model is not None:
                del self.model
                self.model = None

            self._loaded = False
            logger.info(f"✓ Unloaded model: {self.model_id}")

        except Exception as e:
            logger.error(f"Error unloading model {self.model_id}: {e}")

    def _tensor_to_sparse_dict(self, tensor) -> Dict[str, Any]:
        """
        Convert sparse tensor to dictionary format.

        Args:
            tensor: Sparse tensor from model

        Returns:
            Dictionary with 'indices' and 'values' keys
        """
        coalesced = tensor.coalesce()
        values = coalesced.values().tolist()
        indices = coalesced.indices()[0].tolist()

        return {"indices": indices, "values": values}

    def embed_query(
        self, texts: List[str], prompt: Optional[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate sparse embeddings for query texts.

        Args:
            texts: List of query texts to embed
            prompt: Optional instruction prompt (may not be used by sparse models)
            **kwargs: Additional parameters (model-specific)

        Returns:
            List of sparse embeddings as dicts with 'indices' and 'values'

        Raises:
            RuntimeError: If model is not loaded
            EmbeddingGenerationError: If embedding generation fails
        """
        if not self._loaded or self.model is None:
            self.load()

        try:
            tensors = self.model.encode_query(texts, **kwargs)

            # Convert tensors to sparse dict format
            results = []
            for tensor in tensors:
                sparse_dict = self._tensor_to_sparse_dict(tensor)
                results.append(sparse_dict)

            return results

        except Exception as e:
            error_msg = f"Query embedding generation failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(self.model_id, error_msg)

    def embed_documents(
        self, texts: List[str], prompt: Optional[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate sparse embeddings for document texts.

        Args:
            texts: List of document texts to embed
            prompt: Optional instruction prompt (may not be used by sparse models)
            **kwargs: Additional parameters (model-specific)

        Returns:
            List of sparse embeddings as dicts with 'indices' and 'values'

        Raises:
            RuntimeError: If model is not loaded
            EmbeddingGenerationError: If embedding generation fails
        """
        if not self._loaded or self.model is None:
            self.load()

        try:
            # Encode documents
            tensors = self.model.encode_document(texts, **kwargs)

            # Convert tensors to sparse dict format
            results = []
            for tensor in tensors:
                sparse_dict = self._tensor_to_sparse_dict(tensor)
                results.append(sparse_dict)

            return results

        except Exception as e:
            error_msg = f"Document embedding generation failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(self.model_id, error_msg)
