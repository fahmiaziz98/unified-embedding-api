"""
Rerank model implementation.

This module provides the RerankModel class for reranking
documents using sentence-transformers.
"""

from typing import List, Optional, Dict
from sentence_transformers import CrossEncoder
from loguru import logger

from src.config.settings import get_settings
from src.core.config import ModelConfig
from src.core.exceptions import ModelLoadError, RerankingDocumentError


class RerankModel:
    """
    Cross-encoder model wrapper using sentence-transformers.

    This class wraps sentence-transformers CrossEncoder models
    for ranking documents

    Attributes:
        config: ModelConfig instance
        model: CrossEncoder instance
        _loaded: Flag indicating if the model is loaded
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the rerank model.

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
            return

        logger.info(f"Loading rerank model: {self.config.name}")

        try:
            self.model = CrossEncoder(
                self.config.name,
                device=self.settings.DEVICE,
                trust_remote_code=self.settings.TRUST_REMOTE_CODE,
            )
            self._loaded = True
            logger.success(f"✓ Loaded rerank model: {self.model_id}")

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
    ) -> List[Dict]:
        """
        Rerank documents using the CrossEncoder model.

        Args:
            query (str): The search query string.
            documents (List[str]): List of documents to be reranked.
            top_k (int): Number of top documents to return
            **kwargs: Additional arguments passed to model.rank()

        Returns:
            List[Dict]: List of ranking results with 'corpus_id' and 'score'.
                       Returns top_k results sorted by score (highest first).

        Raises:
            RerankingDocumentError: If reranking fails.
        """
        if not self._loaded or self.model is None:
            self.load()
        
        try:
            # model.rank returns List[Dict] with 'corpus_id' and 'score'
            # Already sorted by score (highest first) and limited to top_k
            ranking_results = self.model.rank(
                query, 
                documents, 
                top_k=top_k, 
                **kwargs
            )
            
            # Normalize scores to 0-1 range for consistency
            normalized_results = self._normalize_rerank_scores(ranking_results)
            
            logger.debug(
                f"Reranked {len(documents)} docs, returned top {len(normalized_results)}"
            )
            
            return normalized_results

        except Exception as e:
            error_msg = f"Reranking documents failed: {str(e)}"
            logger.error(error_msg)
            raise RerankingDocumentError(self.model_id, error_msg)

    def _normalize_rerank_scores(
        self, 
        rankings: List[Dict], 
        target_range: tuple = (0, 1)
    ) -> List[Dict]:
        """
        Normalize reranking scores using min-max normalization.

        Args:
            rankings: List of ranking dictionaries from cross-encoder
                     Format: [{'corpus_id': int, 'score': float}, ...]
            target_range: Target range for normalization (min, max)

        Returns:
            List[Dict]: Rankings with normalized scores
        """
        if not rankings:
            return []
        
        # Extract raw scores
        raw_scores = [ranking["score"] for ranking in rankings]
        
        # Min-Max normalization
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        
        # If all scores are the same, return max target value
        if max_score == min_score:
            return [
                {
                    "corpus_id": r["corpus_id"],
                    "score": target_range[1]
                }
                for r in rankings
            ]
        
        # Normalize to target range
        target_min, target_max = target_range
        normalized_rankings = []
        
        for ranking in rankings:
            score = ranking["score"]
            normalized_score = (
                target_min + 
                (score - min_score) * (target_max - target_min) / (max_score - min_score)
            )
            normalized_rankings.append({
                "corpus_id": ranking["corpus_id"],
                "score": float(normalized_score)
            })
        
        return normalized_rankings

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
            Model type ('rerank')
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