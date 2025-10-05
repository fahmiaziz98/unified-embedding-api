from typing import List, Optional
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    """
    Request model for single text embedding.
    
    Attributes:
        text: The input text to embed
        model_id: Identifier of the model to use
        prompt: Optional prompt for instruction-based models
    """
    text: str
    model_id: str
    prompt: Optional[str] = None

class BatchEmbedRequest(BaseModel):
    """
    Request model for batch text embedding.
    
    Attributes:
        texts: List of input texts to embed
        model_id: Identifier of the model to use
        prompt: Optional prompt for instruction-based models
    """
    texts: List[str]
    model_id: str
    prompt: Optional[str] = None

class EmbedResponse(BaseModel):
    """
    Response model for single text embedding.
    
    Attributes:
        embedding: The generated embedding vector
        dimension: Dimensionality of the embedding
        model_id: Identifier of the model used
        processing_time: Time taken to process the request
    """
    embedding: List[float]
    dimension: int
    model_id: str
    processing_time: float

class BatchEmbedResponse(BaseModel):
    """
    Response model for batch text embedding.
    
    Attributes:
        embeddings: List of generated embedding vectors
        dimension: Dimensionality of the embeddings
        model_id: Identifier of the model used
        processing_time: Time taken to process the request
    """
    embeddings: List[List[float]]
    dimension: int
    model_id: str
    processing_time: float

class SparseEmbedding(BaseModel):
    """
    Sparse embedding model.
    
    Attributes:
        text: The input text that was embedded
        indices: Indices of non-zero elements in the sparse vector
        values: Values corresponding to the indices
    """
    text: Optional[str] = None
    indices: List[int]
    values: List[float]

class SparseEmbedResponse(BaseModel):
    """
    Sparse embedding response model.

    Attributes:
        sparse_embedding: The generated sparse embedding
        model_id: Identifier of the model used
        processing_time: Time taken to process the request
    """
    sparse_embedding: SparseEmbedding
    model_id: str
    processing_time: float

class BatchSparseEmbedResponse(BaseModel):
    """
    Batch sparse embedding response model.

    Attributes:
        embeddings: List of generated sparse embeddings
        model_id: Identifier of the model used
    """
    embeddings: List[SparseEmbedding]
    model_id: str
    processing_time: float