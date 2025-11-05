"""
Single/Batch embedding generation endpoints.

This module provides routes for generating embeddings for
multiple texts in a single request.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.models.schemas import (
    EmbedRequest,
    DenseEmbedResponse,
    EmbeddingObject,
    TokenUsage,
    SparseEmbedResponse,
    SparseEmbedding,
)
from src.core.manager import ModelManager
from src.core.exceptions import (
    ModelNotFoundError,
    ModelNotLoadedError,
    EmbeddingGenerationError,
    ValidationError,
)
from src.api.dependencies import get_model_manager
from src.utils.validators import extract_embedding_kwargs, validate_texts, count_tokens_batch
from src.config.settings import get_settings


router = APIRouter(tags=["embeddings"])


@router.post(
    "/embeddings",
    response_model=DenseEmbedResponse,
    summary="Generate single/batch embeddings",
    description="Generate embeddings for multiple texts in a single request",
)
async def create_embeddings_document(
    request: EmbedRequest,
    manager: ModelManager = Depends(get_model_manager),
    settings=Depends(get_settings),
):
    """
    Generate embeddings for multiple texts.

    Args:
        request: BatchEmbedRequest with input, model, and optional parameters
        manager: Model manager dependency
        settings: Application settings

    Returns:
        DenseEmbedResponse 
    Raises:
        HTTPException: On validation or generation errors
    """
    try:
        # Validate input
        validate_texts(
            request.input,
            max_length=settings.MAX_TEXT_LENGTH,
            max_batch_size=settings.MAX_BATCH_SIZE,
        )
        kwargs = extract_embedding_kwargs(request)

        model = manager.get_model(request.model)
        config = manager.model_configs[request.model]

        start_time = time.time()

        if config.type == "embeddings":
            embeddings = model.embed(
                input=request.input, **kwargs
            )
            processing_time = time.time() - start_time

            data = []
            for idx, embedding in enumerate(embeddings):
                data.append(
                    EmbeddingObject(
                        object="embedding",
                        embedding=embedding,
                        index=idx,
                    )
                )
            
            # Calculate token usage
            token_usage = TokenUsage(
                prompt_tokens=count_tokens_batch(request.input),
                total_tokens=count_tokens_batch(request.input),
            )

            response = DenseEmbedResponse(
                object="list",
                data=data,
                model=request.model,
                usage=token_usage,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{request.model}' is not a dense model. Type: {config.type}",
            )

        logger.info(
            f"Generated {len(request.input)} embeddings "
            f"in {processing_time:.3f}s ({len(request.input) / processing_time:.1f} texts/s)"
        )

        return response

    except (ValidationError, ModelNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingGenerationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in create_embeddings_document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch embeddings: {str(e)}",
        )


@router.post(
    "/embed_sparse",
    response_model=SparseEmbedResponse,
    summary="Generate single/batch sparse embeddings",
    description="Generate embedding for a multiple query text",
)
async def create_sparse_embedding(
    request: EmbedRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Generate a single/batch sparse embedding.

    Args:
        request: EmbedRequest with input, model, and optional parameters
        manager: Model manager dependency

    Returns:
        SparseEmbedResponse 

    Raises:
        HTTPException: On validation or generation errors
    """
    try:
        validate_texts(request.input)
        kwargs = extract_embedding_kwargs(request)

        model = manager.get_model(request.model)
        config = manager.model_configs[request.model]

        start_time = time.time()

        if config.type == "sparse-embeddings":
            sparse_results = model.embed(
                input=request.input, **kwargs
            )
            processing_time = time.time() - start_time

            sparse_embeddings = []
            for idx, sparse_result in enumerate(sparse_results):
                sparse_embeddings.append(
                    SparseEmbedding(
                        text=request.input[idx],
                        indices=sparse_result["indices"],
                        values=sparse_result["values"],
                    )
                )

            response = SparseEmbedResponse(
                embeddings=sparse_embeddings,
                count=len(sparse_embeddings),
                model=request.model
            )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{request.model}' is not a sparse model. Type: {config.type}",
            )

        logger.info(
            f"Generated {len(request.texts)} embeddings "
            f"in {processing_time:.3f}s ({len(request.texts) / processing_time:.1f} texts/s)"
        )

        return response

    except (ValidationError, ModelNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingGenerationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in create_query_embedding")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create query embedding: {str(e)}",
        )
