"""
Model manager for loading and managing embedding models.

This module provides the ModelManager class which handles model
lifecycle, configuration, and provides a unified interface for
accessing different embedding models.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
from threading import Lock
from loguru import logger

from .base import BaseEmbeddingModel
from .config import ModelConfig
from .exceptions import ModelNotFoundError, ConfigurationError, ModelLoadError
from src.models.embeddings import DenseEmbeddingModel, SparseEmbeddingModel


class ModelManager:
    """
    Manages multiple embedding models based on a configuration file.

    This class handles:
    - Loading model configurations from YAML
    - Lazy loading and preloading of models
    - Thread-safe model access
    - Model lifecycle management

    Attributes:
        models: Dictionary mapping model IDs to their instances
        model_configs: Dictionary mapping model IDs to their configurations
        _lock: Threading lock for thread-safe operations
        _preload_complete: Flag indicating if all models have been preloaded
    """

    def __init__(self, config_path: str = "src/config/models.yaml"):
        """
        Initialize the model manager.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.models: Dict[str, BaseEmbeddingModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self._lock = Lock()
        self._preload_complete = False

        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """
        Load model configurations from YAML file.

        Args:
            config_path: Path to configuration file

        Raises:
            ConfigurationError: If file not found or invalid
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not config or "models" not in config:
                raise ConfigurationError(
                    "Invalid configuration: 'models' key not found"
                )

            # Parse each model configuration
            for model_id, model_cfg in config["models"].items():
                try:
                    self.model_configs[model_id] = ModelConfig(model_id, model_cfg)
                except Exception as e:
                    logger.warning(f"Skipping invalid model config '{model_id}': {e}")
                    continue

            if not self.model_configs:
                raise ConfigurationError("No valid model configurations found")

            logger.info(f"Loaded {len(self.model_configs)} model configurations")

        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML parsing error: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _create_model(self, config: ModelConfig) -> BaseEmbeddingModel:
        """
        Factory method to create model instances based on type.

        Args:
            config: ModelConfig instance

        Returns:
            Instantiated model (DenseEmbeddingModel or SparseEmbeddingModel)
        """
        if config.type == "sparse-embeddings":
            return SparseEmbeddingModel(config)
        else:
            return DenseEmbeddingModel(config)

    def preload_all_models(self) -> None:
        """
        Preload all models defined in the configuration.

        This method loads all models at startup to avoid cold-start
        latency on first request.
        """
        if self._preload_complete:
            logger.info("Models already preloaded")
            return

        logger.info(f"Preloading {len(self.model_configs)} models...")

        successful_loads = 0
        failed_loads = []

        for model_id, config in self.model_configs.items():
            try:
                with self._lock:
                    if model_id not in self.models:
                        model = self._create_model(config)
                        model.load()
                        self.models[model_id] = model
                        successful_loads += 1

            except ModelLoadError as e:
                logger.error(f"Failed to preload {model_id}: {e.message}")
                failed_loads.append(model_id)
            except Exception as e:
                logger.error(f"Unexpected error preloading {model_id}: {e}")
                failed_loads.append(model_id)

        self._preload_complete = True

        if failed_loads:
            logger.warning(
                f"Preloaded {successful_loads}/{len(self.model_configs)} models. "
                f"Failed: {', '.join(failed_loads)}"
            )
        else:
            logger.success(f"Successfully preloaded all {successful_loads} models")

    def get_model(self, model_id: str) -> BaseEmbeddingModel:
        """
        Retrieve a model instance by its ID.

        Loads the model on-demand if not already loaded.

        Args:
            model_id: The ID of the model to retrieve

        Returns:
            The model instance

        Raises:
            ModelNotFoundError: If model_id not in configuration
            ModelLoadError: If model fails to load
        """
        if model_id not in self.model_configs:
            raise ModelNotFoundError(model_id)

        with self._lock:
            # Return if already loaded
            if model_id in self.models:
                model = self.models[model_id]
                # Double check it's actually loaded
                if model.is_loaded:
                    return model

            # Load on-demand
            logger.info(f"Loading model on-demand: {model_id}")

            try:
                config = self.model_configs[model_id]
                model = self._create_model(config)
                model.load()
                self.models[model_id] = model
                logger.success(f"Loaded: {model_id}")
                return model

            except ModelLoadError:
                raise
            except Exception as e:
                raise ModelLoadError(model_id, str(e))

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: The ID of the model

        Returns:
            Dictionary with model details and load status
        """
        if model_id not in self.model_configs:
            return {}

        config = self.model_configs[model_id]
        is_loaded = model_id in self.models and self.models[model_id].is_loaded

        return {
            "id": config.id,
            "name": config.name,
            "type": config.type,
            "loaded": is_loaded,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their configurations and load status.

        Returns:
            List of model information dictionaries
        """
        return [self.get_model_info(model_id) for model_id in self.model_configs.keys()]

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for loaded models.

        Returns:
            Dictionary with memory usage information
        """
        loaded_models = []
        for model_id, model in self.models.items():
            if model.is_loaded:
                loaded_models.append(
                    {
                        "id": model_id,
                        "type": self.model_configs[model_id].type,
                        "name": model.config.name,
                    }
                )

        return {
            "total_available": len(self.model_configs),
            "loaded_count": len(loaded_models),
            "loaded_models": loaded_models,
            "preload_complete": self._preload_complete,
        }

    def unload_all_models(self) -> None:
        """
        Unload all models and clear the model cache.

        This method safely unloads all models and frees memory.
        """
        with self._lock:
            count = len(self.models)

            for model_id, model in self.models.items():
                try:
                    model.unload()
                except Exception as e:
                    logger.error(f"Error unloading {model_id}: {e}")

            self.models.clear()
            self._preload_complete = False

            logger.info(f"Unloaded {count} models")

    def generate_api_description(self) -> str:
        """
        Generate a dynamic API description based on available models.

        Returns:
            Formatted API description string
        """
        dense_models = []
        sparse_models = []

        for model_id, config in self.model_configs.items():
            if config.type == "sparse-embeddings":
                sparse_models.append(f"**{config.name}**")
            else:
                dense_models.append(f"**{config.name}**")

        description = """
High-performance API for generating text embeddings using multiple model architectures.

"""
        if dense_models:
            description += "✅ **Dense Embedding Models:**\n"
            for model in dense_models:
                description += f"- {model}\n"
            description += "\n"

        if sparse_models:
            description += "🔤 **Sparse Embedding Models:**\n"
            for model in sparse_models:
                description += f"- {model}\n"
            description += "\n"

        description += """
🚀 **Features:**
- Single text embedding generation
- Batch text embedding processing
- Both dense and sparse vector outputs
- Automatic model type detection
- Flexible parameters (normalize, batch_size, etc.)
- List all available models with status
- Fast response times with preloading

📊 **Statistics:**
"""
        description += f"- Total configured models: **{len(self.model_configs)}**\n"
        description += f"- Dense embedding models: **{len(dense_models)}**\n"
        description += f"- Sparse embedding models: **{len(sparse_models)}**\n"
        description += """

⚠️ Note: This is a development API. For production use, deploy on cloud platforms like Hugging Face TEI, AWS, or GCP.
        """
        return description.strip()
