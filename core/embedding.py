from typing import List, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger

from .config import ModelConfig


class EmbeddingModel:
    """
    Embedding model wrapper for dense embeddings.
    
    attributes:
        config: ModelConfig instance
        model: SentenceTransformer instance
        _loaded: Flag indicating if the model is loaded
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self._loaded = False
    
    def load(self) -> None:
        """Load the embedding model."""
        if self._loaded:
            return
            
        logger.info(f"Loading embedding model: {self.config.name}")
        try:
            self.model = SentenceTransformer(self.config.name, device="cpu", trust_remote_code=True)
            self._loaded = True
            logger.success(f"Loaded embedding model: {self.config.id}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.config.id}: {e}")
            raise

    def query_embed(self, text: List[str], prompt: Optional[str] = None) -> List[float]:
        """
        method to generate embedding for a single text.
        
        Args:
            text: Input text
            prompt: Optional prompt for instruction-based models
        
        Returns:
            Embedding vector
        """
        if not self._loaded:
            self.load()
            
        try:
            embeddings = self.model.encode_query(text, prompt=prompt)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def embed_documents(self, texts: List[str], prompt: Optional[str] = None) -> List[List[float]]:
        """
        method to generate embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            prompt: Optional prompt for instruction-based models
        
        Returns:            
        List of embedding vectors
        """
        if not self._loaded:
            self.load()
            
        try:
            embeddings = self.model.encode_document(texts, prompt=prompt)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
