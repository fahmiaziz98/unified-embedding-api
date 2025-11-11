"""
Rerank Endpoint Module

This module provides routes for reranking documents based on a query.
It accepts a list of documents and returns a ranked list based on relevance to the query.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from src.models.schemas import RerankRequest
from src.core.manager import ModelManager
from src.core.exceptions import (
    ModelNotFoundError,
    ModelNotLoadedError,
    RerankingDocumentError,
    ValidationError,
)
from src.api.dependencies import get_model_manager
from src.utils.validators import extract_embedding_kwargs, ensure_model_type

router = APIRouter(prefix="/rerank", tags=["rerank"])


@router.post(
    "",
    summary="Rerank documents",
    description="Reranks the provided documents based on the given query.",
)
async def rerank_documents(
    request: RerankRequest,
    manager: ModelManager = Depends(get_model_manager),
):
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

    valid_docs = [
        (i, doc.strip()) for i, doc in enumerate(request.documents) if doc.strip()
    ]

    if not valid_docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid documents provided.",
        )

    try:
        kwargs = extract_embedding_kwargs(request)

        kwargs.pop("query", None)
        kwargs.pop("documents", None)
        kwargs.pop("top_k", None)

        model = manager.get_model(request.model)
        config = manager.model_configs[request.model]

        ensure_model_type(config, "rerank", request.model)

        start = time.time()

        documents_list = [doc for _, doc in valid_docs]

        ranking_results = model.rank_document(
            query=request.query,
            documents=documents_list,
            top_k=request.top_k,
            **kwargs,
        )

        processing_time = time.time() - start

        results = []

        for rank_result in ranking_results:
            doc_idx = rank_result.get("corpus_id", 0)
            if doc_idx < len(valid_docs):
                original_idx = valid_docs[doc_idx][0]  # Original index
                doc_text = documents_list[doc_idx]
                score = rank_result["score"]
        
                results.append({
                    "text": doc_text,
                    "score": float(score),     
                    "index": int(original_idx)
                })

        logger.info(
            f"Reranked {len(results)} documents in {processing_time:.3f}s "
            f"(model: {request.model})"
        )

        return JSONResponse(content=results)

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
