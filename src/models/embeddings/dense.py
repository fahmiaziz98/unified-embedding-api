"""
Dense embedding model implementation.

This module provides the DenseEmbeddingModel class for generating
dense vector embeddings using sentence-transformers.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config.settings import get_settings
from src.core.base import BaseEmbeddingModel
from src.core.config import ModelConfig
from src.core.exceptions import ModelLoadError, EmbeddingGenerationError


class DenseEmbeddingModel(BaseEmbeddingModel):
    """
    Dense embedding model wrapper using sentence-transformers.

    This class wraps sentence-transformers SentenceTransformer models
    to generate dense vector embeddings for text.

    Attributes:
        config: ModelConfig instance
        model: SentenceTransformer instance
        _loaded: Flag indicating if the model is loaded
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the dense embedding model.

        Args:
            config: ModelConfig instance with model configuration
        """
        super().__init__(config)
        self.model: Optional[SentenceTransformer] = None
        self.settings = get_settings()

    def load(self) -> None:
        """
        Load the dense embedding model into memory.

        Raises:
            ModelLoadError: If model fails to load
        """
        if self._loaded:
            logger.debug(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading dense embedding model: {self.config.name}")

        try:
            self.model = SentenceTransformer(
                self.config.name,
                device=self.settings.DEVICE,
                trust_remote_code=self.settings.TRUST_REMOTE_CODE,
            )
            self._loaded = True
            logger.success(f"✓ Loaded dense model: {self.model_id}")

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(f"✗ {error_msg}")
            raise ModelLoadError(self.model_id, error_msg)

    def unload(self) -> None:
        """
        Unload the model from memory and free resources.

        This method safely releases the model and clears GPU/CPU memory.
        """
        if not self._loaded:
            logger.debug(f"Model {self.model_id} not loaded, nothing to unload")
            return

        try:
            if self.model is not None:
                # Clear model from memory
                del self.model
                self.model = None

            self._loaded = False
            logger.info(f"✓ Unloaded model: {self.model_id}")

        except Exception as e:
            logger.error(f"Error unloading model {self.model_id}: {e}")

    def embed(self, input: List[str], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for query texts.

        Args:
            input: List of query texts to embed
            **kwargs: Additional parameters for sentence-transformers:
                - normalize_embeddings (bool)
                - batch_size (int)
                - convert_to_numpy (bool)
                - etc.

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If model is not loaded
            EmbeddingGenerationError: If embedding generation fails
        """
        if not self._loaded or self.model is None:
            self.load()

        try:
            embeddings = self.model.encode(input, **kwargs)
            
            return [
                emb.tolist() if hasattr(emb, "tolist") else list(emb)
                for emb in embeddings
            ]

        except Exception as e:
            error_msg = f"Query embedding generation failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(self.model_id, error_msg)
