"""
LLM-based Code Generation Pipeline

Orchestrates the process of:
1. Loading market description and transaction data
2. Building a comprehensive prompt
3. Generating code with the LLM
4. Storing the generated code with metadata
"""
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add marketsim to path if needed
try:
    from marketsim.LLM.prompt_builder import (
        build_code_generation_prompt,
        build_function_generation_prompt,
    )
    from marketsim.LLM.code_generator import CodeGenerator, CodeGenerationError
    from marketsim.LLM.code_storage import CodeStorage
except ImportError:
    # Fallback for when running from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from marketsim.LLM.prompt_builder import (
        build_code_generation_prompt,
        build_function_generation_prompt,
    )
    from marketsim.LLM.code_generator import CodeGenerator, CodeGenerationError
    from marketsim.LLM.code_storage import CodeStorage


class CodeGenerationPipeline:
    """Manages the full pipeline from data to generated code."""

    def __init__(
        self,
        market_description: str,
        data_path: Optional[str] = None,
        storage_dir: str = "llm_calls",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        max_retries: int = 1,
    ):
        """
        Initialize the pipeline.

        Args:
            market_description: Natural language description of the market
            data_path: Path to transaction data JSON file
            storage_dir: Directory for storing generated code
            llm_base_url: Base URL for LLM API (optional; if None, uses environment/default in CodeGenerator)
            llm_api_key: API key for LLM (optional; if None, uses environment/default in CodeGenerator)
            max_retries: Maximum number of LLM calls per generation (default 1 = single call, no retries)
        """
        self.market_description = market_description
        self.data_path = data_path
        self.storage = CodeStorage(storage_dir)
        self.generator = CodeGenerator(base_url=llm_base_url, api_key=llm_api_key)
        self.market_data: Optional[Dict[str, Any]] = None
        self.max_retries = max_retries

        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load transaction data from JSON file.

        Args:
            data_path: Path to JSON file

        Returns:
            Loaded data dictionary
        """
        with open(data_path, "r") as f:
            self.market_data = json.load(f)
        return self.market_data

    def generate_function(
        self,
        function_name: str,
        function_purpose: str,
        input_spec: str,
        output_spec: str,
        iteration: int = 1,
        error_feedback: Optional[str] = None,
        previous_attempt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single function.

        Args:
            function_name: Name of the function to generate
            function_purpose: What the function should do
            input_spec: Description of input parameters
            output_spec: Description of return value
            iteration: Iteration number (for versioning)
            error_feedback: Optional error to address
            previous_attempt: Optional previous code to improve

        Returns:
            Result dictionary with keys:
            - code: Generated Python code
            - success: Whether generation succeeded
            - error: Error message if failed
            - file_path: Path where code was saved
        """
        if not self.market_data:
            raise ValueError("Market data not loaded. Call load_data() first.")

        # Build the prompt
        prompt = build_function_generation_prompt(
            market_description=self.market_description,
            transaction_data=self.market_data,
            function_name=function_name,
            function_purpose=function_purpose,
            input_spec=input_spec,
            output_spec=output_spec,
            error_feedback=error_feedback,
            previous_attempt=previous_attempt,
        )

        # Generate and validate code
        code, success, error = self.generator.generate_and_validate(prompt, max_retries=self.max_retries)

        # Store the code
        file_path = self.storage.save_code(
            code=code,
            function_name=function_name,
            iteration=iteration,
            task_description=function_purpose,
            market_data=self.market_data,
            error_feedback=error_feedback,
            success=success,
        )

        return {
            "code": code,
            "success": success,
            "error": error,
            "file_path": file_path,
            "function_name": function_name,
        }

    def generate_code(
        self,
        task_description: str,
        iteration: int = 1,
        error_feedback: Optional[str] = None,
        previous_attempt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate arbitrary Python code for a task.

        Args:
            task_description: Description of what code to generate
            iteration: Iteration number (for versioning)
            error_feedback: Optional error to address
            previous_attempt: Optional previous code to improve

        Returns:
            Result dictionary with keys:
            - code: Generated Python code
            - success: Whether generation succeeded
            - error: Error message if failed
        """
        if not self.market_data:
            raise ValueError("Market data not loaded. Call load_data() first.")

        # Build the prompt
        prompt = build_code_generation_prompt(
            market_description=self.market_description,
            transaction_data=self.market_data,
            task_description=task_description,
            error_feedback=error_feedback,
            previous_attempt=previous_attempt,
        )

        # Generate and validate code
        code, success, error = self.generator.generate_and_validate(prompt, max_retries=self.max_retries)

        return {
            "code": code,
            "success": success,
            "error": error,
        }

    def list_generated_functions(self) -> Dict[str, Dict[str, Any]]:
        """
        List all generated functions.

        Returns:
            Dictionary of function names to their latest metadata
        """
        return self.storage.list_functions()

    def get_function_versions(self, function_name: str) -> list:
        """
        Get all versions of a function.

        Args:
            function_name: Name of the function

        Returns:
            List of metadata dictionaries
        """
        return self.storage.list_all_versions(function_name)


# ============================================================================
# Default Market Description
# ============================================================================
DEFAULT_MARKET_DESCRIPTION = """
You are a Zero-Intelligence (ZI) buyer agent in a continuous double auction
market trading a single asset over a fixed number of timesteps.

You have a private valuation schedule (values for units you want to buy).
Your objective is to maximize surplus by bidding below your private values.
For example, if you have a private value of 100 for a unit and you buy it 
for 90, your surplus is 10.

At each timestep, agents may submit buy orders if they are buyer agents
or sell orders if they are seller agents. Orders are matched
when the best bid meets or exceeds the best ask, producing a transaction at
a price determined by the agent who entered the market first.

You must choose a single shading strategy (a numeric interval) and use it
consistently for the entire market duration.
"""


# ============================================================================
# Example Usage & Command-Line Interface
# ============================================================================
def main():
    """Example usage of the pipeline."""
    # Find the data file - could be in results/ or data/ directory
    data_files = [
        Path("results/zi_only_transactions.json"),
        Path("../LLama 3/data/zi_only_transactions.json"),
    ]

    data_path = None
    for f in data_files:
        if f.exists():
            data_path = str(f)
            print(f"Found data file: {data_path}")
            break

    if not data_path:
        print("Error: Could not find transaction data file")
        print("Please run old_pipelines/scripts/generate_transaction_data.py first")
        return

    # Initialize the pipeline
    pipeline = CodeGenerationPipeline(
        market_description=DEFAULT_MARKET_DESCRIPTION,
        data_path=data_path,
    )

    print("\n" + "=" * 80)
    print("LLM-BASED CODE GENERATION PIPELINE")
    print("=" * 80)
    print(f"Market Description: {pipeline.market_description[:100]}...")
    print(f"Data File: {data_path}")
    print(f"Generated code will be stored in: llm_calls/")

    # Example: Generate a market analysis function
    print("\n" + "=" * 80)
    print("GENERATING: Market Analysis Function")
    print("=" * 80)

    result = pipeline.generate_function(
        function_name="analyze_market_efficiency",
        function_purpose="Analyze market efficiency metrics from transaction data",
        input_spec="""
        transactions: List of transaction dictionaries with keys:
            - time: timestep of transaction
            - price: transaction price
            - involved: whether the analyzing agent was involved
        """,
        output_spec="""
        Dictionary with keys:
            - efficiency_score: float between 0 and 1
            - volatility: price volatility metric
            - average_price: mean transaction price
            - num_transactions: total transactions
            - analysis: text description of findings
        """,
        iteration=1,
    )

    print(f"\nSuccess: {result['success']}")
    if result['success']:
        print(f"Code saved to: {result['file_path']}")
        print("\nGenerated Code Preview:")
        print("-" * 80)
        print(result['code'][:500] + "..." if len(result['code']) > 500 else result['code'])
    else:
        print(f"Error: {result['error']}")

    # List all generated functions
    print("\n" + "=" * 80)
    print("GENERATED FUNCTIONS")
    print("=" * 80)
    functions = pipeline.list_generated_functions()
    for func_name, metadata in functions.items():
        print(f"- {func_name}: iteration {metadata.get('iteration')}, success={metadata.get('success')}")


if __name__ == "__main__":
    main()
