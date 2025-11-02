"""
Rerank Endpoint Module

This module provides routes for reranking documents based on a query.
It accepts a list of documents and returns a ranked list based on relevance to the query.
"""

import time
from typing import Union, List
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
    "/rerank", response_model=RerankResponse, summary="Rerank documents", description="Reranks the provided documents based on the given query."
)
async def rerank_documents(
    request: RerankRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> RerankResponse:
    """
    Rerank documents based on a query.

    This endpoint processes a list of documents and returns them ranked according to their relevance to the query.
    
    Args:
        request (RerankRequest): The request object containing the query and documents to rank.
        manager (ModelManager): The model manager dependency to access the model.

    Returns:
        RerankResponse: The response containing the ranked documents and processing time.

    Raises:
        HTTPException: If there are validation errors, model loading issues, or unexpected errors.
    """
    # Filter out empty documents and keep original indices
    valid_docs = [
        (i, doc.strip()) for i, doc in enumerate(request.documents) if doc.strip()
    ]

    if not valid_docs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid documents provided.")

    try:
        kwargs = extract_embedding_kwargs(request)
        model = manager.get_model(request.model_id)
        config = manager.model_configs[request.model_id]

        start = time.time()
        if config.type == "rerank":
            scores = model.rank_document(
                request.query, [doc for _, doc in valid_docs], request.top_k, **kwargs
            )
            processing_time = time.time() - start

            original_indices, documents_list = zip(*valid_docs)
            results: List[RerankResult] = []

            for i, (orig_idx, doc) in enumerate(zip(original_indices, documents_list)):
                results.append(RerankResult(text=doc, score=scores[i], index=orig_idx))

            # Sort results by score in descending order
            results.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"Reranked documents in {processing_time:.3f} seconds")
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
            detail=f"Failed to create query embedding: {str(e)}",
        )
