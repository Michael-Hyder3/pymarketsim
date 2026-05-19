import tempfile
from pathlib import Path

from marketsim.fourheap.constants import BUY
from marketsim.LLM.intra_sim_pipeline import run_intra_sim_adaptation_episode


def _simple_callback(_payload, _previous_code, segment_index):
    shade = 120 + (segment_index % 3) * 20
    return f'''
def agent_strategy(private_values, market_history, market_state):
    """Deterministic smoke-test strategy for segment {segment_index}."""
    return ({shade}, 1)
'''


def test_intra_sim_updates_happen_at_boundaries():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_intra_sim_adaptation_episode(
            llm_order_type=BUY,
            num_zi_buy=2,
            num_zi_sell=2,
            timesteps=50,
            q_max=5,
            fixed_seed=123,
            update_cadence=10,
            storage_dir=str(Path(tmpdir) / "llm_calls_intra_test"),
            strategy_update_callback=_simple_callback,
        )

    assert "intra_sim" in result
    intra = result["intra_sim"]
    assert intra["num_updates"] == 4
    assert len(intra["updates"]) == 4
    for idx, upd in enumerate(intra["updates"], start=1):
        assert upd["segment_index"] == idx
        assert upd["strategy_updated"] is True
        assert upd["segment_end"] == idx * 10
