"""
Unified Embedding API - Main Application Entry Point

A unified API for dense and sparse embeddings with support for
multiple models from Hugging Face.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config.settings import get_settings
from src.core.manager import ModelManager
from src.api import dependencies
from src.api.routers import embedding, model_list, health
from src.utils.logger import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize models and cache
    - Shutdown: Cleanup resources
    """
    settings = get_settings()

    # Setup logging
    setup_logger(
        level=settings.LOG_LEVEL, log_file=settings.LOG_FILE, log_dir=settings.LOG_DIR
    )

    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    try:
        # Initialize model manager
        logger.info("Initializing model manager...")
        manager = ModelManager(settings.MODEL_CONFIG_PATH)
        dependencies.set_model_manager(manager)

        # Preload models if enabled
        if settings.PRELOAD_MODELS:
            logger.info("Preloading models...")
            manager.preload_all_models()
        else:
            logger.info("Model preloading disabled (lazy loading)")

        logger.success("Startup complete! API is ready to serve requests.")

    except Exception:
        logger.exception("Failed to initialize application")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")

    try:
        manager = dependencies.get_model_manager()
        if manager:
            manager.unload_all_models()
            logger.info("All models unloaded")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    # Generate API description
    temp_manager = ModelManager(settings.MODEL_CONFIG_PATH)
    api_description = temp_manager.generate_api_description()
    del temp_manager

    # Create FastAPI app
    app = FastAPI(
        title=settings.APP_NAME,
        description=api_description,
        version=settings.VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        debug=settings.DEBUG,
    )

    # Add CORS middleware if enabled
    if settings.CORS_ENABLED:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(f"CORS enabled for origins: {settings.CORS_ORIGINS}")

    # Include routers
    app.include_router(health.router)  # Root and health (no prefix)
    app.include_router(embedding.router, prefix="/api/v1")
    app.include_router(model_list.router, prefix="/api/v1")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
