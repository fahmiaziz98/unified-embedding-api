"""
Rerank endpoint

This module provides routes for rerank documents
"""

import time
from typing import Union
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

router = APIRouter(prefix="rerank", tags=["rerank"])


@router.post(
    "/", response_model=RerankResponse, summary="reranking documents", description=""
)
async def rerank_documents(
    request: RerankRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> Union[RerankResponse, HTTPException]:
    """"""
    # Filter out empty documents and keep original indices
    valid_docs = [
        (i, doc.strip()) for i, doc in enumerate(request.documents) if doc.strip()
    ]
    try:
        kwargs = extract_embedding_kwargs(request)
        model = manager.get_model(request.model_id)
        config = manager.model_configs[request.model_id]

        start = time.time()
        if config.type == "rerank":
            scores = model.rank_document(
                request.query, request.documents, request.top_k, **kwargs
            )
            processing_time = time.time() - start

            original_indices, documents_list = zip(*valid_docs)
            results: list[RerankResult] = []

            for i, (orig_idx, doc) in enumerate(zip(original_indices, documents_list)):
                results.append(RerankResult(text=doc, score=scores[i], index=orig_idx))

            # Sort by score descending
            results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Rerank documents in {processing_time:.3f} seconds")
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
