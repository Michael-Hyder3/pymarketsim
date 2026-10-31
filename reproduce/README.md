# Reproduce helpers

Top-level folder next to `marketsim/` and `scripts/` — not mixed into the
pipeline package. Scripts only (no bundled holdout dumps). Run from the repo root.

| Path | Role |
|------|------|
| `programs/{pv,sp,lob}.py` | Reported shade programs |
| `holdout_eval.py` | Score Ours + ZI/ZIP/HBL on ID or OOD |
| `best_constant.py` | Pick C* then score both holdouts (needed for Cs / `check_subset`) |
| `stats.py` | Rebuild table stats once `reproduce/results/*.json` exist |
| `check_subset.py` | Exact-prefix check vs local results |
| `common.py` | Shared holdout/config helpers |

```bash
# creation entrypoints (LLM)
./scripts/create_pv.sh
./scripts/create_sp.sh
./scripts/create_lob.sh

# holdouts (no LLM)
python reproduce/holdout_eval.py pv ood --episodes 100
python reproduce/best_constant.py pv --episodes 100
python reproduce/stats.py
python reproduce/check_subset.py --episodes 100
```

Outputs go under `reproduce/results/` (gitignored).
