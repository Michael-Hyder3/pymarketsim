"""
Configuration templates for LLM function generation.

Each configuration defines:
- function_name: Name of the function to generate
- function_purpose: What the function should do
- input_spec: Description of input parameters
- output_spec: Description of return value
- max_iterations: Maximum refinement iterations
"""

# Configuration for shading region function
SUGGEST_SHADING_REGION = {
    "function_name": "suggest_shading_region",
    "function_purpose": "Return a simple shading region for a ZI buyer: always [0, X] for some positive X.",
    "input_spec": """
    private_values: list of float, the agent's private valuations (highest to lowest)
    """,
    "output_spec": """
    Return a Python list [0, X], where X is a positive float. Do not return anything else.
    """,
    "max_iterations": 3,
}

# Configuration for another potential function (template for future use)
ANALYZE_MARKET_EFFICIENCY = {
    "function_name": "analyze_market_efficiency",
    "function_purpose": "Analyze market efficiency metrics from transaction data",
    "input_spec": """
    transactions: List of transaction dictionaries with keys:
        - time: timestep of transaction
        - price: transaction price
        - involved: whether the analyzing agent was involved
    """,
    "output_spec": """
    Dictionary with keys:
        - efficiency_score: float between 0 and 1
        - volatility: price volatility metric
        - average_price: mean transaction price
        - num_transactions: total transactions
        - analysis: text description of findings
    """,
    "max_iterations": 2,
}

# Configuration for price prediction function (template)
PREDICT_NEXT_PRICE = {
    "function_name": "predict_next_price",
    "function_purpose": "Predict the next transaction price based on recent market history",
    "input_spec": """
    recent_transactions: List of recent transaction prices (last 10-20 trades)
    fundamental_value: Current fundamental value estimate
    """,
    "output_spec": """
    Return a single float representing the predicted next transaction price.
    """,
    "max_iterations": 3,
}

# Configuration for optimal bid function (template)
CALCULATE_OPTIMAL_BID = {
    "function_name": "calculate_optimal_bid",
    "function_purpose": "Calculate an optimal bid price for a buyer given their valuation",
    "input_spec": """
    private_value: float, the buyer's private valuation for this unit
    best_ask: float, the current best ask in the market (or infinity if none)
    market_history: dict with keys:
        - recent_prices: list of recent transaction prices
        - volatility: estimated price volatility
    """,
    "output_spec": """
    Return a single float representing the optimal bid price.
    """,
    "max_iterations": 3,
}

# Configuration for timestep-based scheduling strategy
CREATE_TIMESTEP_SCHEDULE = {
    "function_name": "create_timestep_schedule",
    "function_purpose": "Create a timestep-based shading schedule for an agent.",
    "input_spec": """
    private_values: list of float, the agent's private valuations (highest to lowest)
    total_timesteps: int, total number of timesteps in the market simulation
    """,
    "output_spec": """
    Return a list of entries where each entry contains only:
    - timestep: int
    - shading_region: list [low, high] with 0 <= low <= high
    The list must cover all timesteps from 0 to total_timesteps-1.
    """,
    "max_iterations": 3,
}

# All available configurations
AVAILABLE_CONFIGS = {
    "shading": SUGGEST_SHADING_REGION,
    "efficiency": ANALYZE_MARKET_EFFICIENCY,
    "prediction": PREDICT_NEXT_PRICE,
    "optimal_bid": CALCULATE_OPTIMAL_BID,
    "timestep_schedule": CREATE_TIMESTEP_SCHEDULE,
}


def get_config(config_name: str) -> dict:
    """
    Get configuration by name.
    
    Args:
        config_name: Key in AVAILABLE_CONFIGS (e.g., "shading", "efficiency")
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If config_name not found
    """
    if config_name not in AVAILABLE_CONFIGS:
        available = ", ".join(AVAILABLE_CONFIGS.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    return AVAILABLE_CONFIGS[config_name]


def list_configs() -> dict:
    """List all available function configurations."""
    return {
        name: {
            "function_name": config["function_name"],
            "function_purpose": config["function_purpose"],
        }
        for name, config in AVAILABLE_CONFIGS.items()
    }
