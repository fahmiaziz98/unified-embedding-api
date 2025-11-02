from typing import Any, Dict, List, Optional
from sentence_transformers import SparseEncoder
from loguru import logger

from ..src.core.config import ModelConfig


class SparseEmbeddingModel:
    """
    Sparse embedding model wrapper.

    Attributes:
        config: ModelConfig instance
        model: SparseEncoder instance
        _loaded: Flag indicating if the model is loaded
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[SparseEncoder] = None
        self._loaded = False

    def load(self) -> None:
        """Load the sparse embedding model."""
        if self._loaded:
            return

        logger.info(f"Loading sparse model: {self.config.name}")
        try:
            self.model = SparseEncoder(self.config.name)
            self._loaded = True
            logger.success(f"Loaded sparse model: {self.config.id}")
        except Exception as e:
            logger.error(f"Failed to load sparse model {self.config.id}: {e}")
            raise

    def query_embed(
        self, text: List[str], prompt: Optional[str] = None
    ) -> Dict[Any, Any]:
        """
        Generate a sparse embedding for a single text.

        Args:
            text: Input text
            prompt: Optional prompt for instruction-based models
        Returns:
            Sparse embedding as a dictionary with 'indices' and 'values' keys.
        """
        if not self._loaded:
            self.load()

        try:
            tensor = self.model.encode_query(text)

            values = tensor[0].coalesce().values().tolist()
            indices = tensor[0].coalesce().indices()[0].tolist()

            return {"indices": indices, "values": values}
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise

    def embed_documents(
        self, text: List[str], prompt: Optional[str] = None
    ) -> Dict[Any, Any]:
        """
        Generate a sparse embedding for a single text.

        Args:
            text: Input text
            prompt: Optional prompt for instruction-based models

        Returns:
            Sparse embedding as a dictionary with 'indices' and 'values' keys.
        """

        try:
            tensor = self.model.encode(text)

            values = tensor[0].coalesce().values().tolist()
            indices = tensor[0].coalesce().indices()[0].tolist()

            return {"indices": indices, "values": values}

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise

    def embed_batch(
        self, texts: List[str], prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate sparse embeddings for a batch of texts.

        Args:
            texts: List of input texts
            prompt: Optional prompt for instruction-based models

        Returns:
            List of sparse embeddings as dictionaries with 'text' and 'sparse_embedding' keys.
        """
        if not self._loaded:
            self.load()

        try:
            tensors = self.model.encode(texts)
            results = []

            for i, tensor in enumerate(tensors):
                values = tensor.coalesce().values().tolist()
                indices = tensor.coalesce().indices()[0].tolist()

                results.append(
                    {
                        "text": texts[i],
                        "sparse_embedding": {"indices": indices, "values": values},
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Sparse embedding generation failed: {e}")
            raise
