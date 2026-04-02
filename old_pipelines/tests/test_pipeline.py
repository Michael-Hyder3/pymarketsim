"""
Test pipeline with mock data (no dependencies on numpy/torch)

This allows testing the pipeline without the full environment setup.
"""
import json
import sys
import ast
import importlib.util
from pathlib import Path

# Import directly without going through marketsim __init__
llm_dir = Path(__file__).parent / "marketsim" / "LLM"
sys.path.insert(0, str(llm_dir.parent))

# Import modules directly
spec = importlib.util.spec_from_file_location("prompt_builder", llm_dir / "prompt_builder.py")
prompt_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_builder)

spec = importlib.util.spec_from_file_location("code_generator", llm_dir / "code_generator.py")
code_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_generator)

spec = importlib.util.spec_from_file_location("code_storage", llm_dir / "code_storage.py")
code_storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_storage)

build_function_generation_prompt = prompt_builder.build_function_generation_prompt
CodeGenerator = code_generator.CodeGenerator
CodeStorage = code_storage.CodeStorage


def create_mock_market_data() -> dict:
    """Create mock transaction data for testing."""
    return {
        "participant_private_values": {
            "agent_id": 0,
            "buyer_values": [100.0, 95.0, 90.0, 85.0, 80.0],
            "agent_type": "buyer"
        },
        "total_transactions": 42,
        "timesteps": 300,
        "transactions": [
            {"time": 10, "price": 87.5, "involved": True},
            {"time": 25, "price": 88.2, "involved": False},
            {"time": 35, "price": 89.1, "involved": True},
            {"time": 42, "price": 87.8, "involved": False},
            {"time": 55, "price": 90.5, "involved": True},
        ]
    }


def test_prompt_builder():
    """Test the prompt builder."""
    print("\n" + "=" * 80)
    print("TEST 1: Prompt Builder")
    print("=" * 80)

    market_description = """
    Continuous double auction with ZI agents.
    Buyers have private values, sellers have costs.
    """

    mock_data = create_mock_market_data()

    prompt = build_function_generation_prompt(
        market_description=market_description,
        transaction_data=mock_data,
        function_name="calculate_surplus",
        function_purpose="Calculate agent surplus from transactions",
        input_spec="agent_id: int, private_values: list, transactions: list",
        output_spec="float representing total surplus",
    )

    print("✓ Prompt built successfully")
    print(f"\nPrompt length: {len(prompt)} characters")
    print("\nPrompt preview (first 500 chars):")
    print("-" * 80)
    print(prompt[:500] + "...")
    print("-" * 80)

    return True


def test_code_storage():
    """Test the code storage system."""
    print("\n" + "=" * 80)
    print("TEST 2: Code Storage System")
    print("=" * 80)

    storage = CodeStorage(storage_dir="llm_calls_test")

    # Test saving code
    sample_code = '''
def analyze_market(transactions):
    """Analyze market from transactions."""
    prices = [t["price"] for t in transactions]
    return {
        "mean": sum(prices) / len(prices),
        "min": min(prices),
        "max": max(prices),
    }
'''

    file_path = storage.save_code(
        code=sample_code,
        function_name="analyze_market",
        iteration=1,
        task_description="Test market analysis function",
        market_data=create_mock_market_data(),
        success=True,
    )

    print(f"✓ Code saved to: {file_path}")

    # Test listing functions
    functions = storage.list_functions()
    print(f"✓ Listed functions: {list(functions.keys())}")

    # Test getting versions
    versions = storage.list_all_versions("analyze_market")
    print(f"✓ Found {len(versions)} version(s) of analyze_market")

    # Test loading code
    loaded_code = storage.load_code_file(file_path)
    print(f"✓ Loaded code length: {len(loaded_code)} characters")

    return True


def test_code_validation():
    """Test code validation."""
    print("\n" + "=" * 80)
    print("TEST 3: Code Validation")
    print("=" * 80)

    valid_code = '''
def hello():
    return "world"
'''

    invalid_code = '''
def hello(
    return "world"
'''

    valid_result, valid_error = CodeGenerator.validate_code(valid_code)
    invalid_result, invalid_error = CodeGenerator.validate_code(invalid_code)

    print(f"✓ Valid code check: {valid_result} (error: {valid_error})")
    print(f"✓ Invalid code check: {invalid_result} (error: {invalid_error})")

    assert valid_result is True, "Valid code should pass"
    assert invalid_result is False, "Invalid code should fail"

    return True


def test_code_extraction():
    """Test Python code extraction from markdown."""
    print("\n" + "=" * 80)
    print("TEST 4: Code Extraction from LLM Response")
    print("=" * 80)

    # Test proper markdown code block
    llm_response = """
    Here's the function you requested:

    ```python
    def analyze_trades(transactions):
        '''Analyze trading patterns'''
        prices = [t['price'] for t in transactions]
        return {
            'count': len(transactions),
            'avg_price': sum(prices) / len(prices) if prices else 0,
        }
    ```

    This function calculates basic statistics.
    """

    extracted = CodeGenerator._extract_python_code(llm_response)
    print(f"✓ Code extracted successfully")
    print("\nExtracted code:")
    print("-" * 80)
    print(extracted)
    print("-" * 80)

    # Test with different markdown format (no language specifier)
    llm_response2 = """
    Here's code:

    ```
    def simple_func():
        return 42
    ```
    """

    extracted2 = CodeGenerator._extract_python_code(llm_response2)
    print(f"✓ Code extracted from generic code block")

    return True


def test_integration():
    """Test integration of components."""
    print("\n" + "=" * 80)
    print("TEST 5: Integration Test")
    print("=" * 80)

    market_description = "Continuous double auction with autonomous traders"
    mock_data = create_mock_market_data()

    # Build prompt
    prompt = build_function_generation_prompt(
        market_description=market_description,
        transaction_data=mock_data,
        function_name="test_func",
        function_purpose="Test function generation",
        input_spec="x: int",
        output_spec="int",
    )

    print("✓ Prompt built")
    print(f"  - Market data included: {len(json.dumps(mock_data))} bytes")
    print(f"  - Total prompt size: {len(prompt)} characters")

    # Create storage
    storage = CodeStorage(storage_dir="llm_calls_integration_test")
    print("✓ Storage initialized")

    # Simulate generated code
    test_code = '''
def test_func(x):
    """Test function."""
    return x * 2
'''

    # Validate
    is_valid, error = CodeGenerator.validate_code(test_code)
    assert is_valid, f"Test code should be valid: {error}"
    print("✓ Code validated")

    # Store
    file_path = storage.save_code(
        code=test_code,
        function_name="test_func",
        iteration=1,
        task_description="Integration test",
        market_data=mock_data,
        success=is_valid,
    )
    print(f"✓ Code stored at {file_path}")

    # List
    functions = storage.list_functions()
    assert "test_func" in functions, "Function should be listed"
    print(f"✓ Function listed successfully")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PIPELINE VALIDATION TESTS (Mock Data - No LLM Calls)")
    print("=" * 80)

    tests = [
        ("Prompt Builder", test_prompt_builder),
        ("Code Storage", test_code_storage),
        ("Code Validation", test_code_validation),
        ("Code Extraction", test_code_extraction),
        ("Integration", test_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {name} PASSED")
            else:
                failed += 1
                print(f"\n✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed == 0:
        print("\n✓ All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run generate_transaction_data.py to create market data")
        print("2. Use the pipeline with the generated data")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
