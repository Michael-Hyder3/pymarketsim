"""Q-refresh node selection: newest N + highest-Q fill to top_k."""

from marketsim.LLM.reflection_pipeline.pipeline import select_q_refresh_node_ids


def test_select_q_refresh_under_cap_returns_all():
    ids = [1, 2, 5, 7]
    q = {1: 10.0, 2: 20.0, 5: 5.0, 7: 1.0}
    assert select_q_refresh_node_ids(ids, q_by_id=q, top_k=30, always_recent=3) == [
        1,
        2,
        5,
        7,
    ]


def test_select_q_refresh_always_recent_plus_top_q():
    # 40 nodes; always keep 38,39,40; fill 27 more from 1..37 by Q
    ids = list(range(1, 41))
    q = {n: float(n) for n in ids}  # higher id => higher q among older too
    selected = select_q_refresh_node_ids(
        ids, q_by_id=q, top_k=30, always_recent=3
    )
    assert selected[:3] == [38, 39, 40]
    assert len(selected) == 30
    # Remaining should be highest Q among 1..37 => 37 down to 11
    assert selected[3:] == list(range(37, 10, -1))
    assert 1 not in selected
    assert 10 not in selected
    assert 11 in selected


def test_select_q_refresh_excludes_recent_from_q_fill():
    ids = list(range(1, 11))
    # Make an old node have huge Q; recent still forced in
    q = {n: 0.0 for n in ids}
    q[1] = 1000.0
    q[9] = -50.0
    q[10] = -100.0
    selected = select_q_refresh_node_ids(
        ids, q_by_id=q, top_k=5, always_recent=2
    )
    assert selected[:2] == [9, 10]
    assert 1 in selected  # highest Q among non-recent
    assert len(selected) == 5


def test_select_q_refresh_empty():
    assert select_q_refresh_node_ids([], q_by_id={}) == []
