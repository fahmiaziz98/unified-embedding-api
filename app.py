from contextlib import asynccontextmanager
import time
from typing import Union, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger

from core import ModelManager
from models import (
    EmbedRequest,
    EmbedResponse,
    BatchEmbedRequest,
    BatchEmbedResponse,
    SparseEmbedding,
    SparseEmbedResponse,
    BatchSparseEmbedResponse,
)


model_manager: Optional[ModelManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager that runs at application startup and shutdown.

    On startup, initializes a :class:`core.ModelManager` using the
    configuration file and preloads all models so the first inference call
    does not pay a cold-start penalty. On shutdown, it unloads all models to
    free memory.

    Args:
        app: FastAPI instance (passed by FastAPI runtime).

    Yields:
        None. This is an async contextmanager used by FastAPI's lifespan.
    """
    global model_manager
    
    logger.info("Starting embedding API...")
    try:
        model_manager = ModelManager("config.yaml")
        logger.info("Preloading all models...")
        model_manager.preload_all_models()
        logger.success("All models preloaded successfully!")
    except Exception:
        # log exception with traceback for easier debugging
        logger.exception("Failed to initialize models")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down embedding API...")
    if model_manager:
        model_manager.unload_all_models()



def create_app() -> FastAPI:
    """Create and return the FastAPI application instance.

    This function instantiates a temporary :class:`ModelManager` to generate
    the API description based on available models in the configuration, then
    deletes it to avoid keeping models loaded before the application lifespan
    runs.

    Returns:
        FastAPI: configured FastAPI application.
    """

    temp_manager = ModelManager("config.yaml")
    api_description = temp_manager.generate_api_description()

    # explicitly delete the temporary manager to avoid keeping models loaded
    del temp_manager

    return FastAPI(
        title="Unified Embedding API",
        description=api_description,
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

app = create_app()

@app.post("/query", response_model=Union[EmbedResponse, SparseEmbedResponse])
async def create_query(request: EmbedRequest):
    """Create a single dense or sparse query embedding for the given text.

    The request must include `model_id`. For sparse models (config type
    "sparse-embeddings") the endpoint returns a `SparseEmbedResponse`,
    otherwise a dense `EmbedResponse` is returned.

    Args:
        request: `EmbedRequest` pydantic model with text, prompt and model_id.
    Returns:
        Union[EmbedResponse, SparseEmbedResponse]: The embedding response.
    Raises:
        HTTPException: on validation or internal errors with appropriate
            HTTP status codes.
    """
    
    if not request.model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    try:
        assert model_manager is not None
        model = model_manager.get_model(request.model_id)
        start_time = time.time()

        config = model_manager.model_configs[request.model_id]

        if config.type == "sparse-embeddings":
            # Sparse embedding
            sparse_result = model.query_embed(text=[request.text], prompt=request.prompt)
            processing_time = time.time() - start_time

            if isinstance(sparse_result, dict) and "indices" in sparse_result:
                sparse_embedding = SparseEmbedding(
                    text=request.text,
                    indices=sparse_result["indices"],
                    values=sparse_result["values"],
                )
            else:
                raise ValueError(f"Unexpected sparse result format: {sparse_result}")

            return SparseEmbedResponse(
                sparse_embedding=sparse_embedding,
                model_id=request.model_id,
                processing_time=processing_time,
            )

        # Dense embedding
        embedding = model.query_embed(text=[request.text], prompt=request.prompt)[0]
        processing_time = time.time() - start_time

        return EmbedResponse(
            embedding=embedding,
            dimension=len(embedding),
            model_id=request.model_id,
            processing_time=processing_time,
        )

    except AssertionError:
        logger.exception("Model manager is not initialized")
        raise HTTPException(status_code=500, detail="Server not ready")
    except Exception:
        logger.exception("Error creating query embedding")
        raise HTTPException(status_code=500, detail="Failed to create query embedding")

@app.post("/embed", response_model=Union[EmbedResponse, SparseEmbedResponse])
async def create_embedding(request: EmbedRequest):
    """Create a single dense or sparse embedding for the given text.

    The request must include `model_id`. For sparse models (config type
    "sparse-embeddings") the endpoint returns a `SparseEmbedResponse`,
    otherwise a dense `EmbedResponse` is returned.

    Args:
        request: `EmbedRequest` pydantic model with text, prompt and model_id.

    Returns:
        Union[EmbedResponse, SparseEmbedResponse]: The embedding response.

    Raises:
        HTTPException: on validation or internal errors with appropriate
            HTTP status codes.
    """

    if not request.model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    try:
        assert model_manager is not None
        model = model_manager.get_model(request.model_id)
        start_time = time.time()

        config = model_manager.model_configs[request.model_id]

        if config.type == "sparse-embeddings":
            # Sparse embedding
            sparse_result = model.embed_documents(text=[request.text], prompt=request.prompt)
            processing_time = time.time() - start_time

            if isinstance(sparse_result, dict) and "indices" in sparse_result:
                sparse_embedding = SparseEmbedding(
                    text=request.text,
                    indices=sparse_result["indices"],
                    values=sparse_result["values"],
                )
            else:
                raise ValueError(f"Unexpected sparse result format: {sparse_result}")

            return SparseEmbedResponse(
                sparse_embedding=sparse_embedding,
                model_id=request.model_id,
                processing_time=processing_time,
            )

        # Dense embedding
        embedding = model.embed_documents(text=[request.text], prompt=request.prompt)[0]
        processing_time = time.time() - start_time

        return EmbedResponse(
            embedding=embedding,
            dimension=len(embedding),
            model_id=request.model_id,
            processing_time=processing_time,
        )

    except AssertionError:
        logger.exception("Model manager is not initialized")
        raise HTTPException(status_code=500, detail="Server not ready")
    except Exception:
        logger.exception("Error creating embedding")
        raise HTTPException(status_code=500, detail="Failed to create embedding")

@app.post(
    "/embed/batch",
    response_model=Union[BatchEmbedResponse, BatchSparseEmbedResponse],
)
async def create_batch_embedding(request: BatchEmbedRequest):
    """Create batch embeddings (dense or sparse) for a list of texts.

    Args:
        request: `BatchEmbedRequest` containing `texts`, optional `prompt`, and
            required `model_id`.

    Returns:
        Union[BatchEmbedResponse, BatchSparseEmbedResponse]: Batch embedding
            responses depending on model type.
    """

    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    if not request.model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    try:
        assert model_manager is not None
        model = model_manager.get_model(request.model_id)
        start_time = time.time()

        config = model_manager.model_configs[request.model_id]

        if config.type == "sparse-embeddings":
            # Sparse batch embedding
            sparse_embeddings_raw = model.embed_batch(request.texts, request.prompt)
            processing_time = time.time() - start_time

            sparse_embeddings = []
            for emb in sparse_embeddings_raw:
                if isinstance(emb, dict) and "sparse_embedding" in emb:
                    sparse_data = emb["sparse_embedding"]
                    text = str(emb.get("text", ""))
                    sparse_embeddings.append(
                        SparseEmbedding(
                            text=text,
                            indices=sparse_data["indices"],
                            values=sparse_data["values"],
                        )
                    )
                else:
                    raise ValueError(f"Unexpected sparse embedding format: {emb}")

            return BatchSparseEmbedResponse(
                embeddings=sparse_embeddings,
                model_id=request.model_id,
                processing_time=processing_time,
            )

        # Dense batch embedding
        embeddings = model.embed(request.texts, request.prompt)
        processing_time = time.time() - start_time

        return BatchEmbedResponse(
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            model_id=request.model_id,
            processing_time=processing_time,
        )

    except AssertionError:
        logger.exception("Model manager is not initialized")
        raise HTTPException(status_code=500, detail="Server not ready")
    except Exception:
        logger.exception("Error creating batch embeddings")
        raise HTTPException(status_code=500, detail="Failed to create batch embeddings")

@app.get("/models")
async def list_available_models():
    try:
        return model_manager.list_models()
    except Exception as e:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail="Failed to list models")

@app.get("/health")
async def health_check():
    try:
        memory_info = model_manager.get_memory_usage()
        return {
            "status": "ok",
            "total_models": len(model_manager.model_configs),
            "loaded_models": memory_info["loaded_count"],
            "memory": memory_info,
            "startup_complete": True
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "error", "error": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Unified Embedding API - Dense & Sparse Embeddings",
        "version": "3.0.0"
    }