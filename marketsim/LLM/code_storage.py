import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import importlib.util

from marketsim.LLM.utils import load_json_if_exists, save_json


class CodeStorage:
    def __init__(self, storage_dir: str = "llm_calls"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        return load_json_if_exists(self.metadata_file) or {}

    def _save_metadata(self):
        save_json(self.metadata_file, self._metadata)


    def save_code(
        self,
        code: str,
        function_name: str,
        iteration: int,
        task_description: str,
        market_data: Optional[Dict[str, Any]] = None,
        error_feedback: Optional[str] = None,
        success: bool = True,
    ) -> str:
        iteration_dir = self.storage_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        code_file = iteration_dir / f"{function_name}.py"
        with open(code_file, "w") as f:
            f.write(code)

        metadata_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "node_id": iteration,
            "function_name": function_name,
            "task_description": task_description,
            "success": success,
            "error_feedback": error_feedback,
            "code_file": str(code_file),
            "storage_kind": "tree_node",
        }

        key = f"{function_name}_iter{iteration}"
        self._metadata[key] = metadata_entry
        self._save_metadata()

        return str(code_file)

    def save_failed_attempt(
        self,
        code: str,
        function_name: str,
        *,
        step: int,
        attempt: int,
        task_description: str,
        error_feedback: Optional[str] = None,
    ) -> str:
        attempt_dir = self.storage_dir / "attempts" / f"step_{step:04d}_try_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        code_file = attempt_dir / f"{function_name}.py"
        with open(code_file, "w") as f:
            f.write(code)

        metadata_entry = {
            "timestamp": datetime.now().isoformat(),
            "function_name": function_name,
            "step": step,
            "attempt": attempt,
            "task_description": task_description,
            "success": False,
            "error_feedback": error_feedback,
            "code_file": str(code_file),
            "storage_kind": "failed_attempt",
        }

        key = f"{function_name}_step{step}_try{attempt}"
        self._metadata[key] = metadata_entry
        self._save_metadata()

        return str(code_file)






