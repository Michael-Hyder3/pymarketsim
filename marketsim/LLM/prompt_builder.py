"""
Prompt Builder for LLM-based Code Generation

Combines market description, transaction data, and error feedback
into a comprehensive prompt for the LLM to generate Python functions.
"""
import json
from typing import Dict, Any, Optional


def build_code_generation_prompt(
    market_description: str,
    transaction_data: Dict[str, Any],
    task_description: str,
    error_feedback: Optional[str] = None,
    previous_attempt: Optional[str] = None,
) -> str:
    """
    Build a comprehensive prompt for the LLM to generate Python code.

    Args:
        market_description: Natural language description of the market and how it works
        transaction_data: Generated transaction data (usually from old_pipelines/scripts/generate_transaction_data.py)
        task_description: What the LLM should generate (e.g., "a Python function that analyzes market trends")
        error_feedback: Optional error message or feedback from a previous attempt
        previous_attempt: Optional code from previous iteration to improve upon

    Returns:
        A formatted prompt string ready to send to the LLM
    """
    # Format transaction data as pretty JSON
    data_json = json.dumps(transaction_data, indent=2)

    # Build the base prompt
    prompt_parts = [
        "You are an expert Python developer tasked with generating code for financial market simulations.",
        "",
        "=" * 80,
        "MARKET DESCRIPTION",
        "=" * 80,
        market_description,
        "",
        "=" * 80,
        "MARKET DATA (Sample)",
        "=" * 80,
        data_json,
        "",
        "=" * 80,
        "YOUR TASK",
        "=" * 80,
        task_description,
        "",
    ]

    # Add previous attempt context if provided
    if previous_attempt:
        prompt_parts.extend([
            "=" * 80,
            "PREVIOUS ATTEMPT (for context/improvement)",
            "=" * 80,
            previous_attempt,
            "",
        ])

    # Add error feedback if provided
    if error_feedback:
        prompt_parts.extend([
            "=" * 80,
            "ERROR FEEDBACK FROM PREVIOUS ATTEMPT",
            "=" * 80,
            "The previous code had the following issue(s):",
            error_feedback,
            "",
            "Please fix these issues and provide a corrected version.",
            "",
        ])

    # Add code generation instructions
    prompt_parts.extend([
        "=" * 80,
        "INSTRUCTIONS",
        "=" * 80,
        "1. Generate clean, well-documented Python code.",
        "2. Ensure all imports are included at the top of the code.",
        "3. Write the code in a single, self-contained code block.",
        "4. Include docstrings explaining what the function does.",
        "5. The code should be immediately executable without external dependencies beyond standard market simulation packages.",
        "",
        "Please provide ONLY the Python code in a markdown code block (```python ... ```).",
        "Do not include explanations outside the code block.",
    ])

    return "\n".join(prompt_parts)


def build_function_generation_prompt(
    market_description: str,
    transaction_data: Dict[str, Any],
    function_name: str,
    function_purpose: str,
    input_spec: str,
    output_spec: str,
    error_feedback: Optional[str] = None,
    previous_attempt: Optional[str] = None,
) -> str:
    """
    Build a specialized prompt for generating a specific Python function.

    Args:
        market_description: Natural language description of the market
        transaction_data: Generated transaction data
        function_name: Name of the function to generate (e.g., "analyze_market_efficiency")
        function_purpose: What the function should do
        input_spec: Description of expected input parameters
        output_spec: Description of expected return value
        error_feedback: Optional error message or feedback
        previous_attempt: Optional previous code to improve upon

    Returns:
        A formatted prompt string for function generation
    """
    task_description = f"""Generate a Python function with the following specifications:

Function Name: {function_name}
Purpose: {function_purpose}

Input Specification:
{input_spec}

Output Specification:
{output_spec}

The function should analyze the provided market data and perform the specified task.
Make reasonable assumptions about market dynamics and agent behavior based on the data provided."""

    return build_code_generation_prompt(
        market_description=market_description,
        transaction_data=transaction_data,
        task_description=task_description,
        error_feedback=error_feedback,
        previous_attempt=previous_attempt,
    )
