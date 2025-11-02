"""
Single/Batch embedding generation endpoints.

This module provides routes for generating embeddings for
multiple texts in a single request.
"""

import time
from typing import Union
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.models.schemas import (
    EmbedRequest,
    DenseEmbedResponse,
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
from src.utils.validators import extract_embedding_kwargs, validate_texts
from src.config.settings import get_settings


router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/embed",
    response_model=Union[DenseEmbedResponse, SparseEmbedResponse],
    summary="Generate single/batch embeddings spesialization document",
    description="Generate embeddings for multiple texts in a single request",
)
async def create_embeddings_document(
    request: EmbedRequest,
    manager: ModelManager = Depends(get_model_manager),
    settings=Depends(get_settings),
):
    """
    Generate embeddings for multiple texts.

    This endpoint efficiently processes multiple texts in a single batch,
    reducing overhead compared to multiple single requests.

    Args:
        request: BatchEmbedRequest with texts, model_id, and optional parameters
        manager: Model manager dependency
        settings: Application settings

    Returns:
        DenseEmbedResponse or SparseEmbedResponse depending on model type

    Raises:
        HTTPException: On validation or generation errors
    """
    try:
        # Validate input
        validate_texts(
            request.texts,
            max_length=settings.MAX_TEXT_LENGTH,
            max_batch_size=settings.MAX_BATCH_SIZE,
        )

        # Extract kwargs
        kwargs = extract_embedding_kwargs(request)


        # Get model
        model = manager.get_model(request.model_id)
        config = manager.model_configs[request.model_id]

        start_time = time.time()

        # Generate embeddings based on model type
        if config.type == "sparse-embeddings":
            # Sparse batch embeddings
            sparse_results = model.embed_documents(
                texts=request.texts, prompt=request.prompt, **kwargs
            )
            processing_time = time.time() - start_time

            # Convert to SparseEmbedding objects
            sparse_embeddings = []
            for idx, sparse_result in enumerate(sparse_results):
                sparse_embeddings.append(
                    SparseEmbedding(
                        text=request.texts[idx],
                        indices=sparse_result["indices"],
                        values=sparse_result["values"],
                    )
                )

            response = SparseEmbedResponse(
                embeddings=sparse_embeddings,
                count=len(sparse_embeddings),
                model_id=request.model_id,
                processing_time=processing_time,
            )
        else:
            # Dense batch embeddings
            embeddings = model.embed_documents(
                texts=request.texts, prompt=request.prompt, **kwargs
            )
            processing_time = time.time() - start_time

            response = DenseEmbedResponse(
                embeddings=embeddings,
                dimension=len(embeddings[0]) if embeddings else 0,
                count=len(embeddings),
                model_id=request.model_id,
                processing_time=processing_time,
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
        logger.exception("Unexpected error in create_embeddings_document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch embeddings: {str(e)}",
        )


@router.post(
    "/query",
    response_model=Union[DenseEmbedResponse, SparseEmbedResponse],
    summary="Generate single/batch embeddings spesialization query",
    description="Generate embedding for a multiple query text",
)
async def create_query_embedding(
    request: EmbedRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Generate a single/batch query embedding.

    This endpoint creates embeddings optimized for search queries.
    Some models differentiate between query and document embeddings.

    Args:
        request: EmbedRequest with text, model_id, and optional parameters
        manager: Model manager dependency
        settings: Application settings

    Returns:
        DenseEmbedResponse or SparseEmbedResponse depending on model type

    Raises:
        HTTPException: On validation or generation errors
    """
    try:
        # Validate input
        validate_texts(request.texts)

        # Extract kwargs
        kwargs = extract_embedding_kwargs(request)

        # Get model
        model = manager.get_model(request.model_id)
        config = manager.model_configs[request.model_id]

        start_time = time.time()

        # Generate embedding based on model type
        if config.type == "sparse-embeddings":
            # Sparse embedding
            sparse_results = model.embed_query(
                texts=request.texts, prompt=request.prompt, **kwargs
            )
            processing_time = time.time() - start_time

            # Convert to SparseEmbedding objects
            sparse_embeddings = []
            for idx, sparse_result in enumerate(sparse_results):
                sparse_embeddings.append(
                    SparseEmbedding(
                        text=request.texts[idx],
                        indices=sparse_result["indices"],
                        values=sparse_result["values"],
                    )
                )

            response = SparseEmbedResponse(
                embeddings=sparse_embeddings,
                count=len(sparse_embeddings),
                model_id=request.model_id,
                processing_time=processing_time,
            )
        else:
            # Dense batch embeddings
            embeddings = model.embed_documents(
                texts=request.texts, prompt=request.prompt, **kwargs
            )
            processing_time = time.time() - start_time

            response = DenseEmbedResponse(
                embeddings=embeddings,
                dimension=len(embeddings[0]) if embeddings else 0,
                count=len(embeddings),
                model_id=request.model_id,
                processing_time=processing_time,
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
