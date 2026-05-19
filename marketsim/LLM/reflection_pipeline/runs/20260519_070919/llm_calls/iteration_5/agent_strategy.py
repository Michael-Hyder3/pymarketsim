def agent_strategy(private_values, market_history, market_state):
    # Calculate the average private value
    avg_private_value = sum(private_values) / len(private_values)

    # Calculate the average market price
    avg_market_price = sum(market_history) / len(market_history)

    # Calculate the shading level as the difference between the average private value and the average market price
    shading_level = (avg_private_value - avg_market_price) / (max(private_values) - min(private_values))

    # Ensure the shading level is within the range [0, 1]
    shading_level = max(0, min(1, shading_level))

    return shading_level