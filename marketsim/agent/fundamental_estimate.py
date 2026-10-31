from __future__ import annotations

from typing import Any, Optional

import numpy as np


def absolute_fundamental_estimate(
    market: Any,
    *,
    obs_noise_var: float = 0.0,
    np_rng: Optional[Any] = None,
    torch_generator: Optional[Any] = None,
) -> float:
    mean, r, T = market.get_info()
    t = market.get_time()
    val = float(market.get_fundamental_value())
    rho = (1.0 - float(r)) ** (float(T) - float(t))
    estimate = (1.0 - rho) * float(mean) + rho * val
    noise_var = float(obs_noise_var)
    if noise_var > 0.0:
        sigma = float(np.sqrt(noise_var))
        if torch_generator is not None:
            import torch

            estimate += float(torch.randn((), generator=torch_generator).item()) * sigma
        elif np_rng is not None:
            estimate += float(np_rng.normal(0.0, sigma))
        else:
            estimate += float(np.random.normal(0.0, sigma))
    return float(estimate)
