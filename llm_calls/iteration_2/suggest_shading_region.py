def suggest_shading_region(private_values):
    """
    Suggest a simple shading region for a ZI buyer: always [0, X] for some positive X.

    Parameters:
    private_values (list of float): The agent's private valuations (highest to lowest)

    Returns:
    list: A Python list [0, X], where X is a positive float.
    """
    # Calculate the minimum private value
    min_private_value = min(private_values)
    
    # Calculate the shading region as [0, X], where X is the minimum private value
    shading_region = [0, abs(min_private_value)]
    
    return shading_region