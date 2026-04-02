def suggest_shading_region(private_values):
    """
    Suggest a simple shading region for a ZI buyer: always [0, X] for some positive X.

    Parameters:
    private_values (list of float): The agent's private valuations (highest to lowest)

    Returns:
    list: A Python list [0, X], where X is a positive float
    """
    # Calculate the average of the private values
    average_private_value = sum(private_values) / len(private_values)
    
    # Calculate the standard deviation of the private values
    std_dev = (sum((x - average_private_value) ** 2 for x in private_values) / len(private_values)) ** 0.5
    
    # Calculate the shading region as 0 to 1.5 standard deviations below the average
    shading_region = [0, average_private_value - 1.5 * std_dev]
    
    return shading_region