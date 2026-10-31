# PyMarketSim

Market simulation plus an LLM reflection pipeline that creates CDA shade programs (PV, Sp, LOB).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install openai
```

LLM endpoint (OpenAI-compatible):

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

## Creation runs

```bash
./scripts/create_pv.sh
./scripts/create_sp.sh
./scripts/create_lob.sh
```

## Holdout / tables

```bash
python reproduce/holdout_eval.py {pv,sp,lob} {ood,id}
python reproduce/best_constant.py {pv,sp,lob}
python reproduce/stats.py
```

See `reproduce/README.md` for details.
