#!/usr/bin/env python
"""
Full pipeline demo: Generate market data, then generate code from it.

This script:
1. Generates transaction data from market simulation
2. Uses the LLM to generate a Python function based on that data
3. Saves the generated code

Configuration is loaded from llm_function_configs.py
"""
import json
import sys
from pathlib import Path

# Add marketsim to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import market simulation, fallback to mock if dependencies missing
try:
    from marketsim.tests.new_setting_tests.mixed_agent_simulation import run_mixed_agent_test
    HAS_MARKET_SIM = True
except ImportError:
    HAS_MARKET_SIM = False
    print("Note: Market simulation dependencies not available, will use mock data")

from marketsim.LLM.code_generation_pipeline import CodeGenerationPipeline, DEFAULT_MARKET_DESCRIPTION
from marketsim.LLM.strategy_test_runner import run_tests_for_function, format_test_feedback
from llm_function_configs import get_config, list_configs


def create_mock_data():
    """Create mock transaction data for testing."""
    return {
        "participant_private_values": {
            "agent_id": 0,
            "buyer_values": [100.0, 95.0, 90.0, 85.0, 80.0],
            "agent_type": "buyer"
        },
        "total_transactions": 42,
        "timesteps": 100,
        "transactions": [
            {"time": i * 10, "price": 85.0 + (i % 5), "involved": i % 2 == 0}
            for i in range(42)
        ]
    }


def extract_transaction_data(results, scenario_name, participant_id=0):
    """Extract transaction data from simulation results."""
    transactions = results.get("transactions", [])
    agents = results.get("agents", {})

    # Get the participant's private values and id
    participant_agent = None
    participant_info = {}
    for agent_name, agent in agents.items():
        if agent.get_id() == participant_id:
            participant_agent = agent
            # Only include id, private values, and agent type
            if hasattr(agent.pv, "buyer_values") and len(agent.pv.buyer_values) > 0:
                participant_info = {
                    "agent_id": participant_id,
                    "buyer_values": [float(v) for v in agent.pv.buyer_values],
                    "agent_type": "buyer"
                }
            elif hasattr(agent.pv, "seller_costs") and len(agent.pv.seller_costs) > 0:
                participant_info = {
                    "agent_id": participant_id,
                    "seller_costs": [float(c) for c in agent.pv.seller_costs],
                    "agent_type": "seller"
                }
            break

    transaction_data = []
    for txn in transactions:
        involved = (txn["buyer_id"] == participant_id) or (txn["seller_id"] == participant_id)
        transaction_record = {
            "time": txn["time"],
            "price": float(txn["price"]),
            "involved": involved
        }
        transaction_data.append(transaction_record)

    return {
        "participant_private_values": participant_info,
        "total_transactions": len(transactions),
        "timesteps": results.get("timesteps", 0),
        "transactions": transaction_data,
    }


def main(config_name: str = "shading"):
    """
    Run the full pipeline to generate a function.
    
    Args:
        config_name: Name of the function configuration to use (see llm_function_configs.py)
    """
    print("\n" + "=" * 80)
    print("FULL PIPELINE: Market Data → Code Generation")
    print("=" * 80)
    
    # Load the configuration
    try:
        config = get_config(config_name)
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nAvailable configurations:")
        for name, info in list_configs().items():
            print(f"  - {name}: {info['function_name']} - {info['function_purpose']}")
        return 1
    
    print(f"\n[CONFIG] Using function configuration: {config_name}")
    print(f"  Function: {config['function_name']}")
    print(f"  Purpose: {config['function_purpose']}")

    # Step 1: Generate market data
    print("\n[STEP 1] Generating market simulation data...")
    print("-" * 80)

    # Try to load existing transaction data first
    data_files = [
        Path("results/zi_only_transactions.json"),
        Path("../LLama 3/data/zi_only_transactions.json"),
    ]
    
    zi_data = None
    for data_file in data_files:
        if data_file.exists():
            print(f"Loading existing transaction data from {data_file}...")
            with open(data_file, 'r') as f:
                zi_data = json.load(f)
            print(f"✓ Loaded transaction data:")
            print(f"  - Total transactions: {zi_data['total_transactions']}")
            print(f"  - Timesteps: {zi_data['timesteps']}")
            break
    
    # If no saved data, generate new data
    if zi_data is None:
        if HAS_MARKET_SIM:
            try:
                print("Generating new market data...")
                zi_results = run_mixed_agent_test(
                    num_zi_buy=6,
                    num_zi_sell=6,
                    num_hbl_buy=0,
                    num_hbl_sell=0,
                    timesteps=100,  # Smaller for faster testing
                    seed=42,
                )
                
                zi_data = extract_transaction_data(zi_results, "zi_only_market")
                print(f"✓ Generated market data:")
                print(f"  - Total transactions: {zi_data['total_transactions']}")
                print(f"  - Timesteps: {zi_data['timesteps']}")
                print(f"  - Sample transactions: {len(zi_data['transactions'])} recorded")

            except Exception as e:
                print(f"✗ Error generating market data: {e}")
                print("Using mock data instead...")
                zi_data = create_mock_data()
        else:
            print("Market simulation dependencies not available")
            print("Using mock data instead...")
            zi_data = create_mock_data()

    # Step 2: Initialize the pipeline
    print("\n[STEP 2] Initializing code generation pipeline...")
    print("-" * 80)

    pipeline = CodeGenerationPipeline(
        market_description=DEFAULT_MARKET_DESCRIPTION,
    )
    
    # Manually set the market data
    pipeline.market_data = zi_data
    print("✓ Pipeline initialized")
    print(f"  - Market description loaded")
    print(f"  - Transaction data loaded ({zi_data['total_transactions']} transactions)")

    # Step 3: Generate code with automatic test-driven refinement
    print("\n[STEP 3] Generating Python code with LLM...")
    print("-" * 80)

    function_name = config["function_name"]
    max_iterations = config.get("max_iterations", 3)
    iteration = 1
    error_feedback = None
    
    while iteration <= max_iterations:
        print(f"\n  Iteration {iteration}/{max_iterations}:")
        
        try:
            result = pipeline.generate_function(
                function_name=function_name,
                function_purpose=config["function_purpose"],
                input_spec=config["input_spec"],
                output_spec=config["output_spec"],
                iteration=iteration,
                error_feedback=error_feedback,
            )

            if not result['success']:
                print(f"  ✗ Code generation failed: {result['error']}")
                return 1

            print(f"  ✓ Code generated: {result['file_path']}")
            
            # Step 3a: Run tests on generated code
            print(f"  ✓ Running tests...")
            test_file = f"llm_calls/iteration_{iteration}/test_{function_name}.py"
            test_success, test_error = run_tests_for_function(function_name, test_file, iteration=iteration)
            
            if test_success:
                print(f"  ✓ All tests passed!")
                print(f"\n  Generated code saved to: {result['file_path']}")
                print(f"\n  Code preview:")
                print("  " + "-" * 76)
                code_lines = result['code'].split('\n')
                for line in code_lines[:20]:  # Show first 20 lines
                    print(f"  {line}")
                if len(code_lines) > 20:
                    print(f"  ... ({len(code_lines) - 20} more lines)")
                print("  " + "-" * 76)
                break
            else:
                # Extract first few lines of error for feedback
                error_lines = test_error.split('\n')
                feedback_lines = [l for l in error_lines if 'AssertionError' in l or 'FAILED' in l][:2]
                error_feedback = "Test failed: " + " ".join(feedback_lines)
                print(f"  ✗ Tests failed. Feedback for next iteration: {error_feedback}")
                iteration += 1

        except Exception as e:
            print(f"  ✗ Error during code generation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    if iteration > max_iterations:
        print(f"\n✗ Max iterations ({max_iterations}) reached without passing tests")
        return 1

    # Step 4: List generated functions
    print("\n[STEP 4] Generated functions inventory...")
    print("-" * 80)

    functions = pipeline.list_generated_functions()
    for func_name, metadata in functions.items():
        print(f"✓ {func_name}")
        print(f"  - Iteration: {metadata.get('iteration')}")
        print(f"  - Success: {metadata.get('success')}")
        print(f"  - File: {metadata.get('code_file')}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the generated code")
    print("2. Test it with your market agents")
    print("3. If improvements needed, run with error_feedback parameter")
    print("\n")

    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the LLM-based code generation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="shading",
        help="Configuration name to use (see llm_function_configs.py for available options)"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available configurations"
    )
    
    args = parser.parse_args()
    
    if args.list_configs:
        print("\nAvailable configurations:")
        print("-" * 80)
        configs = list_configs()
        for name, info in configs.items():
            print(f"  {name:20} - {info['function_name']}")
            print(f"  {'':20}   Purpose: {info['function_purpose'][:60]}...")
        print("-" * 80)
        sys.exit(0)
    
    sys.exit(main(config_name=args.config))
