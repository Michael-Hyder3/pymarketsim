"""
Automated test runner for LLM-generated functions.
Runs tests and captures failure details for feedback.
"""
import unittest
import sys
import importlib.util
from pathlib import Path
from io import StringIO
from typing import Tuple, Optional, Dict, Any


def run_tests_for_function(
    function_name: str,
    test_file_path: str,
    iteration: int = 1,
) -> Tuple[bool, Optional[str]]:
    """
    Run unit tests for an LLM-generated function and capture results.
    
    Args:
        function_name: Name of the function being tested
        test_file_path: Path to the test file (e.g., "llm_calls/iteration_1/test_suggest_shading_region.py")
        iteration: Current iteration number (for importing the right function version)
    
    Returns:
        (success, error_feedback) where:
        - success: True if all tests pass
        - error_feedback: None if success, else a string describing failures
    """
    test_path = Path(test_file_path)
    if not test_path.exists():
        return True, None  # No test file, assume success
    
    # Add parent directory to path so test can import modules
    test_dir = test_path.parent
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir.parent.parent))  # Add root for relative imports
    
    # Dynamically inject the function from the current iteration
    # This ensures tests always test the latest generated code
    func_file = test_dir / f"{function_name}.py"
    if func_file.exists():
        spec = importlib.util.spec_from_file_location(function_name, func_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, function_name, None)
        if func:
            # Inject into globals so test can import it
            sys.modules[function_name] = module
    
    # Load and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern=f"test_{function_name}.py"
    )
    
    # Capture output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    # If all tests passed
    if result.wasSuccessful():
        return True, None
    
    # Build error feedback from failures
    error_lines = []
    for test, traceback in result.failures:
        error_lines.append(f"FAILED: {test}")
        error_lines.append(traceback)
    
    for test, traceback in result.errors:
        error_lines.append(f"ERROR: {test}")
        error_lines.append(traceback)
    
    error_feedback = "\n".join(error_lines)
    return False, error_feedback


def format_test_feedback(
    function_name: str,
    test_success: bool,
    error_details: Optional[str],
) -> str:
    """
    Format test results into a concise feedback message for the LLM.
    
    Args:
        function_name: Name of the function
        test_success: Whether tests passed
        error_details: Detailed error output if tests failed
    
    Returns:
        Formatted feedback string
    """
    if test_success:
        return f"✓ All tests passed for {function_name}"
    
    # Summarize first error for feedback
    if error_details:
        lines = error_details.split('\n')
        # Extract just the assertion error line
        summary_lines = []
        for i, line in enumerate(lines):
            if 'AssertionError' in line or 'FAILED' in line:
                summary_lines.append(line.strip())
                if i + 1 < len(lines):
                    summary_lines.append(lines[i + 1].strip())
                break
        
        if summary_lines:
            return "Test failures:\n" + "\n".join(summary_lines[:2])
    
    return f"Tests failed for {function_name}. See details above."
