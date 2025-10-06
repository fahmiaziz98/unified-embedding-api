from loguru import logger
from typing import Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer
from sentence_transformers import SparseEncoder


class ModelConfig:
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.id = model_id
        self.name = config["name"]
        self.type = config["type"]  # "embedding" or "sparse"
        self.dimension = int(config["dimension"])
        self.max_tokens = int(config["max_tokens"])
        self.description = config["description"]
        self.language = config["language"]
        self.repository = config["repository"]

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

    def embed(self, texts: List[str], prompt: Optional[str] = None) -> List[List[float]]:
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
            embeddings = self.model.encode(texts, prompt=prompt)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

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
    
    def _format_values(self, values: List[float]) -> List[float]:
        """Format float values to a fixed precision."""
        return [round(float(v), 8) for v in values]
    
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

    def embed(self, text: str, prompt: Optional[str] = None) -> Dict[Any, Any]:
        """
        Generate a sparse embedding for a single text.
        
        Args:
            text: Input text
            prompt: Optional prompt for instruction-based models
        
        Returns:
            Sparse embedding as a dictionary with 'indices' and 'values' keys.
        """
        
        try:
            tensor = self.model.encode([text])
            
            values = tensor[0].coalesce().values().tolist()
            indices = tensor[0].coalesce().indices()[0].tolist()
            
            return {
                "indices": indices,
                "values": self._format_values(values)
            }
            

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise

    def embed_batch(self, texts: List[str], prompt: Optional[str] = None) -> List[Dict[str, Any]]:
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
                
                results.append({
                    "text": texts[i],
                    "sparse_embedding": {
                        "indices": indices,
                        "values": self._format_values(values)
                    }
                })
            
            return results
        except Exception as e:
            logger.error(f"Sparse embedding generation failed: {e}")
            raise
