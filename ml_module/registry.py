from __future__ import annotations

from typing import List, Protocol, Optional, Dict, Any
from backend_module.database import DataBaseManager


class MLModel(Protocol):
    def process_group(self, db_manager: DataBaseManager, file_group: List[Dict[str, Any]], work_dir: str) -> Dict[str, Any]:
        ...


def get_model(model: Optional[str], model_source: Optional[str], task_type: Optional[str]) -> Optional[MLModel]:
    """Return a model adapter instance for the given job fields, or None if unsupported."""
    if model == "samurai-ulr" and model_source == "internet" and task_type == "one-shot-object-detection":
        from .model_samurai_ulr import SamuraiULRModel  # lazy import

        return SamuraiULRModel()
    return None

