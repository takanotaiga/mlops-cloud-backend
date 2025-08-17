from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from backend_module.database import DataBaseManager
from query import inference_job_progress_query as q


class StepState:
    PENDING = "pending"       # 実行前
    RUNNING = "running"       # 実行中
    COMPLETED = "completed"   # 実行完了
    FAILD = "faild"           # 失敗 (table-wide convention)


DEFAULT_STEPS: List[Dict[str, str]] = [
    {"key": "download", "label": "Download", "state": StepState.PENDING},
    {"key": "preprocess", "label": "Preprocess", "state": StepState.PENDING},
    {"key": "sam2", "label": "SAM2", "state": StepState.PENDING},
    {"key": "dataset_export", "label": "Dataset export", "state": StepState.PENDING},
    {"key": "rtdetr_train", "label": "RT-DETR train", "state": StepState.PENDING},
    {"key": "trt_export", "label": "TensorRT export", "state": StepState.PENDING},
    {"key": "rtdetr_infer", "label": "RT-DETR inference", "state": StepState.PENDING},
    {"key": "aggregate", "label": "Aggregate", "state": StepState.PENDING},
    {"key": "postprocess", "label": "Postprocess", "state": StepState.PENDING},
    {"key": "upload", "label": "Upload", "state": StepState.PENDING},
]


@dataclass
class InferenceJobProgressTracker:
    db_manager: DataBaseManager
    job_id: str

    def init_default_steps(self) -> Dict[str, Any]:
        """Initialize steps with defaults if missing."""
        return q.init_steps_if_absent(self.db_manager, self.job_id, DEFAULT_STEPS)

    def start(self, key: str) -> Dict[str, Any]:
        return q.update_step_state(self.db_manager, self.job_id, step_key=key, state=StepState.RUNNING)

    def complete(self, key: str) -> Dict[str, Any]:
        return q.update_step_state(self.db_manager, self.job_id, step_key=key, state=StepState.COMPLETED)

    def fail(self, key: str) -> Dict[str, Any]:
        return q.update_step_state(self.db_manager, self.job_id, step_key=key, state=StepState.FAILD)

    def set(self, key: str, state: str) -> Dict[str, Any]:
        return q.update_step_state(self.db_manager, self.job_id, step_key=key, state=state)
