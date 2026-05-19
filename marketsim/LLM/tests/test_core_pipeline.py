from marketsim.LLM.core_episode_runner import run_action_episode


def test_run_action_episode_smoke():
    result = run_action_episode(
        action=0.5,
        num_zi_buy=2,
        num_zi_sell=2,
        timesteps=20,
        q_max=3,
        pv_var=1e4,
        pv_mu=0.0,
        shade_range=[0, 50],
        fundamental_value=1000.0,
        mean_reversion_rate=0.05,
        shock_var=1e4,
        obs_noise_var=1e4,
        arrival_rate=0.2,
        random_seed=42,
        use_dummy_fundamental=True,
    )

    assert "realized_surplus" in result
    assert "baseline_surplus" in result
    assert "equilibrium_price" in result
    assert isinstance(result["realized_surplus"], float)
    assert isinstance(result["baseline_surplus"], float)
    assert result["action"] == 0.5
