# Reproduce

| Path | Role |
|------|------|
| `programs/{pv,sp,lob}.py` | Reported shade programs |
| `holdout_eval.py` | Score Ours + ZI/ZIP/HBL on ID or OOD |
| `best_constant.py` | Select and score C* |
| `stats.py` | Rebuild table stats from `results/` |
| `check_subset.py` | Exact-prefix check vs local results |
| `common.py` | Shared helpers |

```bash
python reproduce/holdout_eval.py pv ood --episodes 100
python reproduce/best_constant.py pv --episodes 100
python reproduce/stats.py
python reproduce/check_subset.py --episodes 100
```
