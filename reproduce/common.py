from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marketsim.LLM.reflection_pipeline.config import (  # noqa: E402
    build_id_holdout_config_space,
    build_ood_holdout_config_space,
    sample_id_holdout_episode_configs,
    sample_ood_holdout_episode_configs,
)
from marketsim.LLM.reflection_pipeline.evaluation import evaluate_batch  # noqa: E402
from marketsim.LLM.reflection_pipeline.run_holdout import (  # noqa: E402
    _dummy_strategy,
    infer_strategy_type,
    load_strategy_from_file,
)

HERE = Path(__file__).resolve().parent
PROGRAMS = HERE / "programs"
RESULTS = HERE / "results"

EPISODES = 5000
BASE_SEED = 12345
SAMPLE_SEED = {"ood": 12345, "id": 54321}
EVAL_STEP = 9999
HBL_L = 6
ZIP_INIT_MU = 0.0015
ZIP_WARMUP = 50
WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 1))

REGIMES = {
    "pv": {
        "background": {"background_agent_kind": "zi"},
        "seats": {
            "ZI": {"special_agent_kind": "zi"},
            "ZIP": {"special_agent_kind": "zip"},
            "HBL": {"special_agent_kind": "hbl_fast", "hbl_l_param": HBL_L},
        },
        "observation": {},
    },
    "sp": {
        "background": {
            "background_agent_kind": "zip_noisy",
            "spread_input_mode": "book",
            "background_zip_initial_mu": ZIP_INIT_MU,
            "background_zip_zi_warmup_steps": ZIP_WARMUP,
        },
        "seats": {
            "ZI": {"special_agent_kind": "zi"},
            "ZIP": {"special_agent_kind": "zip_noisy"},
            "HBL": {"special_agent_kind": "hbl", "hbl_l_param": HBL_L},
        },
        "observation": {"spread_input_mode": "book"},
    },
    "lob": {
        "background": {
            "background_agent_kind": "hbl_fast",
            "lob_input_mode": "book_orders",
        },
        "seats": {
            "ZI": {"special_agent_kind": "zi"},
            "ZIP": {"special_agent_kind": "zip"},
            "HBL": {"special_agent_kind": "hbl_fast", "hbl_l_param": HBL_L},
        },
        "observation": {"lob_input_mode": "book_orders"},
    },
}


def holdout_configs(holdout: str, n: int = EPISODES):
    rng = random.Random(SAMPLE_SEED[holdout])
    if holdout == "ood":
        configs = sample_ood_holdout_episode_configs(
            EPISODES, rng, space=build_ood_holdout_config_space()
        )
    else:
        configs = sample_id_holdout_episode_configs(
            EPISODES, rng, space=build_id_holdout_config_space()
        )
    return configs[:n]


def run_seat(configs, background, seat=None, program=None, step=EVAL_STEP, base_seed=BASE_SEED):
    kwargs = dict(
        batch_configs=configs,
        episodes_per_config=1,
        step_index=step,
        base_seed=base_seed,
        reward_mode="raw_surplus",
        include_batch_data=False,
        parallel_episodes=True,
        parallel_workers=WORKERS,
        **background,
    )
    if program is not None:
        fn = load_strategy_from_file(program)
        kwargs.update(
            strategy_fn=fn,
            strategy_type=infer_strategy_type(fn),
            special_agent_kind="llm",
            strategy_output_mode="shade",
            strategy_code=program.read_text(encoding="utf-8"),
        )
    else:
        kwargs.update(strategy_fn=_dummy_strategy, **seat)
    return evaluate_batch(**kwargs)


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")
