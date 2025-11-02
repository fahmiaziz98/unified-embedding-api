"""
Rerank model implementation.

This module provides the RerankModel class for reranking
documents using sentence-transformers.
"""

from typing import List, Optional
from sentence_transformers import CrossEncoder
from loguru import logger

from src.config.settings import get_settings
from src.core.config import ModelConfig
from src.core.exceptions import ModelLoadError, RerankingDocumentError


class RerankModel:
    """
    Cross-encoder model wrapper using sentence-transformers.

    This class wraps sentence-transformers SentenceTransformer models
    to ranking documents

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
        self.config = config
        self._loaded = False
        self.model: Optional[CrossEncoder] = None
        self.settings = get_settings()

    def load(self) -> None:
        """
        Load the cross-encoder model into memory.

        Raises:
            ModelLoadError: If model fails to load
        """
        if self._loaded:
            logger.debug(f"Model {self.model_id} already loaded")

        logger.info(f"Loading rerank model: {self.config.name}")

        try:
            self.model = CrossEncoder(
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

    def rank_document(
        self,
        query: str,
        documents: List[str],
        top_k: int,
        **kwargs,
    ) -> List[float]:
        """
        Rerank documents using the CrossEncoder model.

        Args:
            query (str): The search query string.
            documents (List[str]): List of documents to be reranked.
            top_k (int): top n documents
            **kwargs

        Returns:
            List[float]: List of relevance scores for each document.

        Raises:.
            Exception: If reranking fails.
        """
        if not self._loaded or self.model is None:
            self.load()
        try:
            scores = self.model.rank(query, documents, top_k=top_k, **kwargs)
            normalized_score = self._normalize_rerank_scores(scores)
            return normalized_score

        except Exception as e:
            error_msg = f"Reranking documents failed: {str(e)}"
            logger.error(error_msg)
            raise RerankingDocumentError(self.model_id, error_msg)

    def _normalize_rerank_scores(
        self, rankings: List[dict], target_range: tuple = (0, 1)
    ) -> List[float]:
        """
        Normalize reranking scores menggunakan berbagai metode.

        Args:
            rankings: List of ranking dictionaries dari cross-encoder
            target_range: Target range untuk minmax normalization (min, max)

        Returns:
            List of normalized scores
        """
        raw_scores = [ranking["score"] for ranking in rankings]

        # Min-Max normalization ke target range
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if max_score == min_score:
            return [target_range[1]] * len(raw_scores)  # All same score

        target_min, target_max = target_range
        normalized = [
            target_min
            + (score - min_score) * (target_max - target_min) / (max_score - min_score)
            for score in raw_scores
        ]
        return normalized

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
