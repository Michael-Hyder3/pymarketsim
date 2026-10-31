#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python3}"
DIR="${RUN_DIR:-runs/pv_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$DIR"

"$PY" -m marketsim.LLM.reflection_pipeline \
  --recipe simple_pv_zi_advantage_ucb_principle \
  --init-types pv \
  --background-agent-kind zi \
  --strategy-output-mode shade \
  --steps 150 \
  --episodes-per-config 50 \
  --batch-size 3 \
  --home-set-size 3 \
  --menu-sweep-on-home-set \
  --resample-every 25 \
  --q-suite-size 3 \
  --elite-parent-bias-prob 0.0 \
  --hypothesis-jump-prob 0.0 \
  --codegen-recovery-mode legacy \
  --suite-aggregate mean_min \
  --suite-aggregate-lambda 0.5 \
  --plateau-non-improve-threshold 8 \
  --parallel-episodes \
  --seed 123 \
  --run-dir "$DIR" \
  "$@" 2>&1 | tee "$DIR/run.log"
