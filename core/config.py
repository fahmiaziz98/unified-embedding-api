from typing import Any, Dict


class ModelConfig:
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.id = model_id
        self.name = config["name"]
        self.type = config["type"] 
        self.repository = config["repository"]