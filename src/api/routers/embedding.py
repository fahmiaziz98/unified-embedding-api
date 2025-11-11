"""
Single/Batch embedding generation endpoints.

This module provides routes for generating embeddings for
multiple texts in a single request.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from src.models.schemas import (
    EmbedRequest,
    DenseEmbedResponse,
    EmbeddingObject,
    TokenUsage,
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
    ensure_model_type,
)

router = APIRouter()


@router.post(
    "/embeddings",
    response_model=DenseEmbedResponse,
    tags=["OpenAI Compatible"],
    summary="Generate single/batch embeddings",
    description="Generate embeddings for multiple texts in a single request",
)
async def create_openai_embeddings(
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

    texts = [request.input] if isinstance(request.input, str) else request.input

    if not texts or not isinstance(texts, list):
        raise ValidationError("Input must be a non-empty list or string.")

    try:
        kwargs = extract_embedding_kwargs(request)

        model = manager.get_model(request.model)
        config = manager.model_configs.get(request.model)

        ensure_model_type(config, "embeddings", request.model)

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
        logger.exception("Unexpected error in create_openai_embeddings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create embeddings: {str(e)}",
        )


@router.post(
    "/embed",
    tags=["embeddings"],
    summary="Generate single/batch dense embeddings",
    description="Generate embedding for a multiple query text",
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

    texts = [request.input] if isinstance(request.input, str) else request.input

    if not texts or not isinstance(texts, list):
        raise ValidationError("Input must be a non-empty list or string.")

    try:
        kwargs = extract_embedding_kwargs(request)

        model = manager.get_model(request.model)
        config = manager.model_configs.get(request.model)

        ensure_model_type(config, "embeddings", request.model)

        start_time = time.time()

        embeddings = model.embed(input=texts, **kwargs)
        processing_time = time.time() - start_time

        
        logger.info(
            f"Generated {len(texts)} embeddings "
            f"in {processing_time:.3f}s ({len(texts) / processing_time:.1f} texts/s)"
        )

        return JSONResponse(content=embeddings)

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
    tags=["embeddings"],
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
    texts = [request.input] if isinstance(request.input, str) else request.input

    if not texts or not isinstance(texts, list):
        raise ValidationError("Input must be a non-empty list or string.")

    try:
        kwargs = extract_embedding_kwargs(request)

        model = manager.get_model(request.model)
        config = manager.model_configs.get(request.model)

        ensure_model_type(config, "sparse-embeddings", request.model)

        start_time = time.time()

        sparse_results = model.embed(input=texts, **kwargs)
        processing_time = time.time() - start_time
        
        formatted_embeddings = [
            [{"index": i, "value": v} for i, v in zip(res["indices"], res["values"])]
            for res in sparse_results
        ]

        logger.info(
            f"Generated {len(texts)} embeddings "
            f"in {processing_time:.3f}s ({len(texts) / processing_time:.1f} texts/s)"
        )

        return JSONResponse(content=formatted_embeddings)

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
