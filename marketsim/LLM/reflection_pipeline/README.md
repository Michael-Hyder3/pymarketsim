# Reflection Pipeline

Implements the batch-UCB reflection pipeline described in the LaTeX spec:
- UCB selection over a refinement tree
- LLM generation + verification
- Batch evaluation on a fixed config set
- Critic reflection and tree update

## Quick start (smoke)
```bash
python -m marketsim.LLM.reflection_pipeline --smoke
```

## Typical run
```bash
python -m marketsim.LLM.reflection_pipeline \
  --batch-size 2 \
  --steps 5 \
  --episodes-per-config 3 \
  --ucb-constant 1.0 \
  --reward-mode advantage
```

## Outputs
Each run is saved under `marketsim/LLM/reflection_pipeline/runs/<timestamp>/` with:
- `reflection_pipeline_summary.json`: run summary and tree snapshot
- `llm_calls/`: generated strategies and metadata
