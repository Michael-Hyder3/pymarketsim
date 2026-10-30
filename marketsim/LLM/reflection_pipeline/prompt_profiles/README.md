# Prompt profiles

Two intact prompt stacks for the reflection pipeline:

| Profile | Location | Role |
|---------|----------|------|
| **standard** (default) | live `prompts.py` + `pipeline.py` system prompts; copy under `standard/` | Zipparity / pre-slim prompts — **default** |
| **slim** | `slim/prompts.py` + `slim/pipeline.py` | Sep 18 narrowed data→hypothesis→refine stack — **preserved intact** |

## Switch

```bash
# default — recovered standard / zipparity-era prompts
python -m marketsim.LLM.reflection_pipeline ... 

# slim option (does not delete standard sources)
python -m marketsim.LLM.reflection_pipeline --prompt-profile slim ...
```

`--prompt-profile slim` monkeypatches builders from `slim/` at process start.
`--prompt-profile standard` uses the live/default sources.

## Do not delete

`slim/` is the preserved slim experiment. `standard/` mirrors the recovered default.
Extra copies also exist under `../prompt_snapshots/`.
