def agent_strategy(private_values, fundamental_estimate, current_pv):
    private_values = list(private_values or [])
    fundamental_estimate = float(fundamental_estimate or 0.0)
    current_pv = float(current_pv or 0.0)

    pv_mean = sum(private_values) / len(private_values)
    pv_max = max(private_values)
    units_available = len(private_values)

    if current_pv > fundamental_estimate:
        shading_level = (pv_mean - fundamental_estimate) / (pv_max - fundamental_estimate)
    else:
        shading_level = (pv_max - fundamental_estimate) / (pv_max - pv_mean + 1e-6)

    if shading_level > 0.5:
        shading_level = 1 - shading_level + 0.001
    else:
        shading_level = shading_level + 0.001

    shading_level = max(0, min(1, shading_level))

    if units_available > 5:
        shading_level = shading_level * 0.85
    else:
        shading_level = shading_level * 0.75

    if current_pv > fundamental_estimate:
        shading_level = 1 - shading_level + 0.002
    else:
        shading_level = shading_level + 0.002

    # Add a small variation to the shading level
    shading_level += 0.0001 * (len(private_values) % 10)

    # Refine the shade map under the chosen menu pair
    if current_pv > fundamental_estimate:
        shading_level = 1 - shading_level + 0.001
    else:
        shading_level = shading_level + 0.001

    return max(0, min(1, shading_level))
