# PyMarketSim (LLM Feedback-Loop Pipeline)

This repo contains market simulation code plus an LLM-driven feedback loop that iteratively generates and evaluates trading strategies.

## What you need

- Python 3.10+
- A virtual environment
- Dependencies from `requirements.txt`
- `openai` Python client (used by `marketsim/LLM/code_generator.py`)
- Access to an OpenAI-compatible LLM endpoint (local or remote)

## Quick setup

```bash
cd pymarketsim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install openai
```

## LLM path / endpoint

The pipeline uses OpenAI-compatible API settings from environment variables:

- `OPENAI_BASE_URL` (example: `http://127.0.0.1:8000/v1`)
- `OPENAI_API_KEY` (any non-empty value for local vLLM setups)
- `OPENAI_MODEL` (default in code: `models/Meta-Llama-3-8B-Instruct`)

In this workspace, model files are located at:

`LLama 3/models/Meta-Llama-3-8B-Instruct`

If your serving stack uses a different model ID, set `OPENAI_MODEL` to that exact served name.

## Run the current pipeline

Run from repo root:

```bash
cd pymarketsim
source .venv/bin/activate
python marketsim/LLM/tests/test_feedback_loop.py --episodes 10 --pv-mode fixed --seed 100
```

Outputs are written to:

- `llm_calls/` (generated code + metadata)
- `results/baseline_only_test.json` (run summary)

## Notes

- The active simulation backend module is `marketsim/LLM/mixed_market_simulator.py`.
- If imports fail, make sure you are running from the repository root.