"""
Rerank Endpoint Module

This module provides routes for reranking documents based on a query.
It accepts a list of documents and returns a ranked list based on relevance to the query.
"""

import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.models.schemas import RerankRequest, RerankResponse, RerankResult
from src.core.manager import ModelManager
from src.core.exceptions import (
    ModelNotFoundError,
    ModelNotLoadedError,
    RerankingDocumentError,
    ValidationError,
)
from src.api.dependencies import get_model_manager
from src.utils.validators import extract_embedding_kwargs

router = APIRouter(tags=["rerank"])


@router.post(
    "/rerank",
    response_model=RerankResponse,
    summary="Rerank documents",
    description="Reranks the provided documents based on the given query.",
)
async def rerank_documents(
    request: RerankRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> RerankResponse:
    """
    Rerank documents based on a query.

    This endpoint processes a list of documents and returns them ranked
    according to their relevance to the query.

    Args:
        request: The request object containing the query and documents to rank
        manager: The model manager dependency to access the model

    Returns:
        RerankResponse: The response containing the ranked documents and processing time

    Raises:
        HTTPException: If there are validation errors, model loading issues, or unexpected errors
    """
    # Filter out empty documents and keep original indices
    valid_docs = [
        (i, doc.strip()) for i, doc in enumerate(request.documents) if doc.strip()
    ]

    if not valid_docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid documents provided.",
        )

    try:
        # Extract kwargs but exclude rerank-specific fields
        kwargs = extract_embedding_kwargs(request)

        # Remove fields that are already passed as positional arguments
        # to avoid "got multiple values for argument" error
        kwargs.pop("query", None)
        kwargs.pop("documents", None)
        kwargs.pop("top_k", None)

        model = manager.get_model(request.model_id)
        config = manager.model_configs[request.model_id]

        if config.type != "rerank":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{request.model_id}' is not a rerank model. Type: {config.type}",
            )

        start = time.time()

        # Call rank_document with clean kwargs
        scores = model.rank_document(
            query=request.query,
            documents=[doc for _, doc in valid_docs],  # Use filtered documents
            top_k=request.top_k,
            **kwargs,
        )

        processing_time = time.time() - start

        # Sebelum memanggil rank_document, tambahkan:
        logger.debug(f"Rerank request - Query: '{request.query}'")
        logger.debug(f"Documents to rank: {len(request.documents)}")
        logger.debug(f"First document: {request.documents[-1][:100]}...")
        logger.debug(f"Top K: {request.top_k}")

        # Setelah rank_document, tambahkan:
        logger.debug(f"Ranking returned {len(scores)} scores")
        logger.debug(f"Sample scores: {scores[:5] if scores else 'None'}")

        # Build results with original indices
        original_indices, documents_list = zip(*valid_docs)
        results = []

        for i, (orig_idx, doc) in enumerate(zip(original_indices, documents_list)):
            results.append(RerankResult(text=doc, score=scores[i], index=orig_idx))

        # Sort results by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"Reranked {len(results)} documents in {processing_time:.3f}s "
            f"(model: {request.model_id})"
        )

        return RerankResponse(
            model_id=request.model_id,
            processing_time=processing_time,
            query=request.query,
            results=results,
        )

    except (ValidationError, ModelNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except RerankingDocumentError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in rerank_documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rerank documents: {str(e)}",
        )
