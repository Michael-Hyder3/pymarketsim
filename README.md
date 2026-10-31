# PyMarketSim

Market simulation plus an LLM reflection pipeline that creates CDA shade programs (PV, Sp, LOB).

Layout:

- `marketsim/` — simulator and reflection pipeline
- `scripts/` — creation run entrypoints
- `reproduce/` — holdout / table helpers (separate from the pipeline package; no result dumps in git)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install openai
```

LLM endpoint (OpenAI-compatible), via env vars only:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional; code default is a local Llama-3 instruct id)

## Creation runs

From the repo root:

```bash
./scripts/create_pv.sh
./scripts/create_sp.sh
./scripts/create_lob.sh
```

These call `python -m marketsim.LLM.reflection_pipeline` with the paper recipes.
Only the **standard** prompt stack is available (`--prompt-profile standard`).

## Holdout / tables (no LLM)

See `reproduce/README.md`. Short path:

```bash
python reproduce/holdout_eval.py {pv,sp,lob} {ood,id}
python reproduce/best_constant.py {pv,sp,lob}
python reproduce/stats.py
```
