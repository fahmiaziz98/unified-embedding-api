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
from src.utils.validators import (
    extract_embedding_kwargs,
    count_tokens_batch,
)

router = APIRouter(tags=["embeddings"])


def _ensure_model_type(config, expected_type: str, model_id: str) -> None:
    """
    Validate that the model configuration matches the expected type.

    Raises:
        HTTPException: If the model is missing or the type does not match.
    """
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found.",
        )
    if config.type != expected_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{model_id}' is not a {expected_type.replace('-', ' ')} "
                f"model. Detected type: {config.type}"
            ),
        )


@router.post(
    "/embeddings",
    response_model=DenseEmbedResponse,
    summary="Generate single/batch embeddings",
    description="Generate embeddings for multiple texts in a single request",
)
async def create_embeddings(
    request: EmbedRequest, manager: ModelManager = Depends(get_model_manager)
):
    """
    Generate embeddings for multiple texts.

    The endpoint validates the request, checks that the requested
    model is a dense embedding model, and returns a
    :class:`DenseEmbedResponse`.

    Raises:
        HTTPException: On validation or generation errors
    """

    if isinstance(request.input, str):
        texts = [request.input]

    try:
        kwargs = extract_embedding_kwargs(request)

        model = manager.get_model(request.model)
        config = manager.model_configs.get(request.model)

        _ensure_model_type(config, "embeddings", request.model)

        start_time = time.time()

        embeddings = model.embed(input=texts, **kwargs)
        processing_time = time.time() - start_time

        data = [
            EmbeddingObject(
                object="embedding",
                embedding=embedding,
                index=idx,
            )
            for idx, embedding in enumerate(embeddings)
        ]

        token_usage = TokenUsage(
            prompt_tokens=count_tokens_batch(texts),
            total_tokens=count_tokens_batch(texts),
        )

        response = DenseEmbedResponse(
            object="list",
            data=data,
            model=request.model,
            usage=token_usage,
        )

        logger.info(
            f"Generated {len(texts)} embeddings "
            f"in {processing_time:.3f}s ({len(texts) / processing_time:.1f} texts/s)"
        )

        return response

    except (ValidationError, ModelNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingGenerationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in create_embeddings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create embeddings: {str(e)}",
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

    The endpoint validates the request, checks that the requested
    model is a sparse embedding model, and returns a
    :class:`SparseEmbedResponse`.

    Raises:
        HTTPException: On validation or generation errors
    """
    if isinstance(request.input, str):
        texts = [request.input]

    try:
        kwargs = extract_embedding_kwargs(request)

        model = manager.get_model(request.model)
        config = manager.model_configs.get(request.model)

        _ensure_model_type(config, "sparse-embeddings", request.model)

        start_time = time.time()

        sparse_results = model.embed(input=texts, **kwargs)
        processing_time = time.time() - start_time

        sparse_embeddings = [
            SparseEmbedding(
                text=texts[idx],
                indices=sparse_result["indices"],
                values=sparse_result["values"],
            )
            for idx, sparse_result in enumerate(sparse_results)
        ]

        response = SparseEmbedResponse(
            embeddings=sparse_embeddings,
            count=len(sparse_embeddings),
            model=request.model,
        )

        logger.info(
            f"Generated {len(texts)} embeddings "
            f"in {processing_time:.3f}s ({len(texts) / processing_time:.1f} texts/s)"
        )

        return response

    except (ValidationError, ModelNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except EmbeddingGenerationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in create_sparse_embedding")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create sparse embedding: {str(e)}",
        )
