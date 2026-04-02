import numpy as np

def create_timestep_schedule(private_values, total_timesteps):
    """
    Create a timestep-based shading schedule for an agent.

    Parameters:
    private_values (list of float): the agent's private valuations (highest to lowest)
    total_timesteps (int): total number of timesteps in the market simulation

    Returns:
    list of entries where each entry contains:
    - timestep (int)
    - shading_region (list [low, high] with 0 <= low <= high)
    The list covers all timesteps from 0 to total_timesteps-1.
    """
    # Calculate the shading region for each private value
    shading_regions = [(i / (len(private_values) - 1), (i + 1) / (len(private_values) - 1)) for i in range(len(private_values))]

    # Create the timestep schedule
    schedule = []
    for timestep in range(total_timesteps):
        # Calculate the shading region for this timestep
        low = np.interp(timestep, [0, total_timesteps - 1], [0, 1])
        high = np.interp(timestep, [0, total_timesteps - 1], [0, 1])
        shading_region = [min(low, high), max(low, high)]
        schedule.append({'timestep': timestep, 'shading_region': shading_region})

    return schedule