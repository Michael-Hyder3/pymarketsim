"""
Code Storage System for Generated Functions

Handles saving, versioning, and importing dynamically generated Python code.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import importlib.util

from marketsim.LLM.utils import load_json_if_exists, save_json


class CodeStorage:
    """Manages storage and retrieval of generated code."""

    def __init__(self, storage_dir: str = "llm_calls"):
        """
        Initialize code storage.

        Args:
            storage_dir: Base directory for storing generated code
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file or create empty dict."""
        return load_json_if_exists(self.metadata_file) or {}

    def _save_metadata(self):
        """Save metadata to file."""
        save_json(self.metadata_file, self._metadata)

    @staticmethod
    def _build_market_summary(market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compact summary of market data for metadata storage."""
        return {
            "total_transactions": market_data.get("total_transactions", 0),
            "timesteps": market_data.get("timesteps", 0),
            "num_agents": len(market_data.get("agents", {})) if isinstance(market_data.get("agents"), dict) else 0,
        }

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
        """
        Save generated code with metadata.

        Args:
            code: The Python code to save
            function_name: Name of the function (used for file naming)
            iteration: Iteration number
            task_description: Description of the task
            market_data: Optional market data used for generation
            error_feedback: Optional error feedback if generation had issues
            success: Whether generation was successful

        Returns:
            Path to the saved code file
        """
        # Create iteration directory
        iteration_dir = self.storage_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        # Save code file
        code_file = iteration_dir / f"{function_name}.py"
        with open(code_file, "w") as f:
            f.write(code)

        # Create metadata entry
        metadata_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "function_name": function_name,
            "task_description": task_description,
            "success": success,
            "error_feedback": error_feedback,
            "code_file": str(code_file),
        }

        if market_data:
            metadata_entry["market_summary"] = self._build_market_summary(market_data)

        # Store metadata
        key = f"{function_name}_iter{iteration}"
        self._metadata[key] = metadata_entry
        self._save_metadata()

        return str(code_file)

    def load_code_file(self, file_path: str) -> str:
        """
        Load code from a file.

        Args:
            file_path: Path to the code file

        Returns:
            The code as a string
        """
        with open(file_path, "r") as f:
            return f.read()

    def import_function(self, file_path: str, function_name: str) -> Callable:
        """
        Dynamically import a function from a generated code file.

        Args:
            file_path: Path to the Python file
            function_name: Name of the function to import

        Returns:
            The function callable

        Raises:
            ImportError: If import fails
        """
        spec = importlib.util.spec_from_file_location("generated_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, function_name):
            raise ImportError(f"Function '{function_name}' not found in {file_path}")

        return getattr(module, function_name)

    def get_latest_function(
        self,
        function_name: str,
        successful_only: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for the latest version of a function.

        Args:
            function_name: Name of the function
            successful_only: Only return successful generations

        Returns:
            Metadata dict or None if not found
        """
        matching = [
            (key, entry)
            for key, entry in self._metadata.items()
            if entry.get("function_name") == function_name
        ]

        if not matching:
            return None

        # Sort by iteration (highest first)
        matching.sort(key=lambda x: x[1].get("iteration", 0), reverse=True)

        for key, entry in matching:
            if successful_only and not entry.get("success", False):
                continue
            return entry

        return None

    def list_functions(self) -> Dict[str, Dict[str, Any]]:
        """
        List all stored functions with their latest metadata.

        Returns:
            Dict mapping function names to their latest metadata
        """
        functions = {}
        for key, entry in self._metadata.items():
            func_name = entry.get("function_name")
            if func_name not in functions or entry.get("iteration", 0) > functions[func_name].get("iteration", 0):
                functions[func_name] = entry
        return functions

    def list_all_versions(self, function_name: str) -> list:
        """
        List all versions of a function.

        Args:
            function_name: Name of the function

        Returns:
            List of metadata dicts, sorted by iteration (newest first)
        """
        versions = [
            entry
            for entry in self._metadata.values()
            if entry.get("function_name") == function_name
        ]
        versions.sort(key=lambda x: x.get("iteration", 0), reverse=True)
        return versions


def get_code_storage(storage_dir: str = "llm_calls") -> CodeStorage:
    """
    Get a CodeStorage instance.

    Args:
        storage_dir: Directory for storing code

    Returns:
        CodeStorage instance
    """
    return CodeStorage(storage_dir)
