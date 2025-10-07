import yaml
from pathlib import Path
from loguru import logger
from typing import Dict, List, Any, Union
from threading import Lock
from .embedding import EmbeddingModel
from .sparse import SparseEmbeddingModel
from .config import ModelConfig

class ModelManager:
    """
    Manages multiple embedding models based on a configuration file.

    Attributes:
        models: Dictionary mapping model IDs to their instances.
        model_configs: Dictionary mapping model IDs to their configurations.
        default_model_id: The default model ID to use if none is specified.
        _lock: A threading lock for thread-safe operations.
        _preload_complete: Flag indicating if all models have been preloaded.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.models: Dict[str, Union[EmbeddingModel, SparseEmbeddingModel]] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self._lock = Lock()  # For thread safety
        self._preload_complete = False
        
        self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> None:
        """Load model configurations from a YAML file."""

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
            for model_id, model_cfg in config["models"].items():
                self.model_configs[model_id] = ModelConfig(model_id, model_cfg)
                
            logger.info(f"Loaded {len(self.model_configs)} model configurations")
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _create_model(self, config: ModelConfig) -> Union[EmbeddingModel, SparseEmbeddingModel]:
        """
        Factory method to create model instances based on type.
        
        Args:
            config: The ModelConfig instance.
            
        Returns:
            The created model instance.
        """
        if config.type == "sparse-embeddings":
            return SparseEmbeddingModel(config)
        else:  
            return EmbeddingModel(config)
    
    def preload_all_models(self) -> None:
        """
        Preload all models defined in the configuration.
        returns: None
        """

        if self._preload_complete:
            logger.info("Models already preloaded")
            return
            
        logger.info(f"Preloading {len(self.model_configs)} models...")
        
        successful_loads = 0
        for model_id, config in self.model_configs.items():
            try:
                with self._lock:
                    if model_id not in self.models:
                        model = self._create_model(config)
                        model.load()
                        self.models[model_id] = model
                        successful_loads += 1
                        logger.debug(f"Preloaded: {model_id}")
                        
            except Exception as e:
                logger.error(f"Failed to preload {model_id}: {e}")
        
        self._preload_complete = True
        logger.success(f"Preloaded {successful_loads}/{len(self.model_configs)} models")
    
    def get_model(self, model_id: str) -> Union[EmbeddingModel, SparseEmbeddingModel]:
        """
        Retrieve a model instance by its ID, loading it on-demand if necessary.
        
        Args:
            model_id: The ID of the model to retrieve.
            
        Returns:
            The model instance.
        """
        if model_id not in self.model_configs:
            raise ValueError(f"Model '{model_id}' not found in configuration")
            
        with self._lock:
            if model_id in self.models:
                return self.models[model_id]
            
            logger.info(f"🔄 Loading model on-demand: {model_id}")
            try:
                config = self.model_configs[model_id]
                model = self._create_model(config)
                model.load()
                self.models[model_id] = model
                logger.success(f"Loaded: {model_id}")
                return model
            except Exception as e:
                raise RuntimeError(f"Failed to load model {model_id}: {e}")
            
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: The ID of the model.
        
        Returns:
            A dictionary with model details and load status.
        """
        if model_id not in self.model_configs:
            return {}
            
        config = self.model_configs[model_id]
        is_loaded = model_id in self.models and self.models[model_id]._loaded
        
        return {
            "id": config.id,
            "name": config.name,
            "type": config.type,
            "loaded": is_loaded,
            "repository": config.repository,
        }
    
    
    def generate_api_description(self) -> str:
        """Generate a dynamic API description based on available models."""

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
        
        # Add features section
        description += """
🚀 **Features:**
- Single text embedding generation
- Batch text embedding processing
- Both dense and sparse vector outputs
- Automatic model type detection
- List all available models with status
- Fast response times with preloading

📊 **Statistics:**
"""
        description += f"- Total configured models: **{len(self.model_configs)}**\n"
        description += f"- Dense embedding models: **{len(dense_models)}**\n"
        description += f"- Sparse embedding models: **{len(sparse_models)}**\n"
        description += """
        
⚠️ Note: This is a development API. For production use, must deploy on cloud like TGI Huggingface, AWS, GCP etc
        """
        return description.strip()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their configurations and load status."""
        return [self.get_model_info(model_id) for model_id in self.model_configs.keys()]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for loaded models."""
        loaded_models = []
        for model_id, model in self.models.items():
            if model._loaded:
                loaded_models.append({
                    "id": model_id,
                    "type": self.model_configs[model_id].type,
                    "name": model.config.name
                })
        
        return {
            "total_available": len(self.model_configs),
            "loaded_count": len(loaded_models),
            "loaded_models": loaded_models,
            "preload_complete": self._preload_complete
        }
    
    def unload_all_models(self) -> None:
        """Unload all models and clear the model cache."""
        with self._lock:
            count = len(self.models)
            for model in self.models.values():
                model.unload()
            self.models.clear()
            self._preload_complete = False
            logger.info(f"Unloaded {count} models")
