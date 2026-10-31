from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from marketsim.LLM.reflection_pipeline.config import (
    build_default_config_space,
    CRITIC_MODES,
    Q_REFRESH_ALWAYS_RECENT,
    Q_REFRESH_EPISODE_WORKERS,
    Q_REFRESH_TOP_K,
)
from marketsim.LLM.reflection_pipeline.learning_focused import (
    DEFAULT_LEARNING_PARENT_MAX_CHILDREN,
)
from marketsim.LLM.reflection_pipeline.pipeline import (
    DEFAULT_CODEGEN_RECOVERY_MODE,
    DEFAULT_MODEL_ALIAS,
    ELITE_PARENT_BIAS_PROB,
    HYPOTHESIS_JUMP_PROB,
    MODEL_ALIASES,
    PLATEAU_NON_IMPROVE_THRESHOLD,
    ReflectionPipeline,
    resolve_model_name,
)
from marketsim.LLM.reflection_pipeline.prompts import list_principle_menu_presets
from marketsim.LLM.reflection_pipeline.recipes import get_recipe, list_recipes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reflection pipeline")
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        choices=list_recipes(),
        help=(
            "Frozen experiment recipe (Phase 3). Applies locked PV defaults; "
            "explicit CLI flags still override recipe values."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes-per-config", type=int, default=None)
    parser.add_argument("--ucb-constant", type=float, default=None)
    parser.add_argument("--max-debug-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--storage-dir", type=str, default="llm_calls_reflection_pipeline")
    parser.add_argument("--output", type=str, default="results/reflection_pipeline_summary.json")
    parser.add_argument(
        "--reward-mode",
        type=str,
        default=None,
        choices=["advantage", "surplus", "consumer_surplus"],
    )
    parser.add_argument(
        "--strategy-output-mode",
        type=str,
        default=None,
        choices=["x_cap", "shade"],
    )
    parser.add_argument(
        "--init-types",
        type=str,
        default=None,
        help="Comma-separated init strategy types (pv,spread,lob)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--disable-critic",
        action="store_true",
        help=(
            "Skip the critic LLM entirely. The generator receives parent code and "
            "batch performance metrics only (no reflection text)."
        ),
    )
    parser.add_argument(
        "--no-critic-data",
        action="store_true",
        help="Omit batch metrics and episode trajectories from the critic prompt (code + scalar reward only)",
    )
    parser.add_argument(
        "--critic-per-config-metrics",
        action="store_true",
        help="Show the critic per-config mean/variance metrics instead of only pooled batch totals",
    )
    parser.add_argument(
        "--generator-critic-data",
        action="store_true",
        help=(
            "Give the generator the parent's batch metrics and median-episode LOB "
            "trajectory (same data the critic would receive). "
            "Automatically enabled when --q-reward-mode is zi_beat_fraction or zi_advantage."
        ),
    )
    parser.add_argument(
        "--generator-sibling",
        action="store_true",
        help=(
            "Give the generator the parent code plus the most recent sibling "
            "strategy from the same parent, if one exists. "
            "Automatically enabled when --q-reward-mode is zi_beat_fraction or zi_advantage."
        ),
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--q-eval-mode",
        type=str,
        default=None,
        choices=["incremental", "full_refresh"],
        help="Q evaluation mode (default from config.py DEFAULT_Q_EVAL_MODE)",
    )
    parser.add_argument(
        "--q-update-mode",
        type=str,
        default=None,
        choices=["replace", "average"],
        help="How to update node q values after each eval (default from config.py)",
    )
    parser.add_argument(
        "--q-reward-mode",
        type=str,
        default=None,
        choices=["advantage", "zi_beat_fraction", "zi_advantage"],
        help=(
            "Q reward signal: advantage (default); zi_beat_fraction (beat_count/10 on "
            "identical seeds); zi_advantage (same ZI benchmark eval and unit tests, but "
            "Q reward = batch advantage for finer parent selection)"
        ),
    )
    parser.add_argument(
        "--suite-aggregate",
        type=str,
        default=None,
        choices=["mean", "mean_min", "min"],
        help=(
            "LLM-only suite scalar that feeds Q (never applied to ZI/HBL/ZIP seats): "
            "mean (default); mean_min = (1-λ)·mean + λ·min; min = worst market only. "
            "HBL beat diagnostics still use plain mean surplus."
        ),
    )
    parser.add_argument(
        "--suite-aggregate-lambda",
        type=float,
        default=None,
        help=(
            "λ for --suite-aggregate mean_min (default 0.5). "
            "Ignored for mean/min. LLM training signal only."
        ),
    )
    parser.add_argument(
        "--parent-selection",
        type=str,
        default=None,
        choices=["auto", "ucb", "newest"],
        help=(
            "Parent selection mode. UCB is the default for all Q reward modes "
            "(including discrete ZI beat fraction). newest: single-lineage "
            "refinement on the latest node. auto is an alias for ucb."
        ),
    )
    parser.add_argument(
        "--disable-inactive-gate",
        action="store_true",
        help="Do not freeze clearly-worse nodes after the inactive-gate threshold",
    )
    parser.add_argument(
        "--inactive-gate-min-nodes",
        type=int,
        default=None,
        help="Start gating only after this many evaluable nodes exist (default 10)",
    )
    parser.add_argument(
        "--inactive-surplus-gap",
        type=float,
        default=None,
        help="Advantage/surplus path: inactive if mean surplus is this far below best",
    )
    parser.add_argument(
        "--inactive-beat-gap",
        type=int,
        default=None,
        help="zi_beat_fraction path: inactive if beat_count is this far below best",
    )
    parser.add_argument(
        "--q-suite-size",
        type=int,
        default=3,
        help="Q suite size passed through to the pipeline. Default 3.",
    )
    parser.add_argument(
        "--menu-sweep-on-home-set",
        action="store_true",
        help=(
            "Before the main --steps loop: force every active menu pair once on the "
            "stratified --home-set-size markets (train=Q=homeset; full LOB menu = 25 "
            "pairs). Compatible with --resample-every (sweep homeset becomes resample "
            "slot 0). Uses the same parallel episode workers as the main loop."
        ),
    )
    parser.add_argument(
        "--home-set-size",
        type=int,
        default=0,
        help=(
            "Use K stratified markets (buyer-heavy / seller-heavy / balanced) for "
            "train + Q + critic. Alone: freeze the same K for the whole run. "
            "With --resample-every N: redraw a new stratified K-set every N steps. "
            "Critic/generator see opaque per-market advantage rows. Compatible with "
            "--menu-sweep-on-home-set. 0 = off."
        ),
    )
    parser.add_argument(
        "--resample-every",
        type=int,
        default=0,
        help=(
            "Hold the current train markets for N steps (train=Q), then redraw. "
            "Alone: hold 1 sampled market. With --home-set-size K: hold a "
            "stratified K-set and redraw every N steps. Tree/Q persist across "
            "redraws; holdout ood_cells are excluded. 0 = off. Compatible with "
            "--menu-sweep-on-home-set."
        ),
    )
    parser.add_argument(
        "--critic-mode",
        type=str,
        default=None,
        choices=sorted(CRITIC_MODES),
        help=(
            "Critic prompt style: default (5-part essay), structured_zi_curriculum "
            "(shade-direction fields), structured_principle_form "
            "(principle + form_class menus), or structured_hypothesis_loop "
            "(current hypothesis → what worked/failed → next hypothesis; menus as "
            "naming aids). Overrides recipe critic_mode when set."
        ),
    )
    parser.add_argument(
        "--plateau-escape-mode",
        type=str,
        default=None,
        choices=["revision", "combination"],
        help=(
            "Plateau escape behavior: revision (new form_class hypothesis) or "
            "combination (gated merge of top two families)."
        ),
    )
    parser.add_argument(
        "--plateau-non-improve-threshold",
        type=int,
        default=None,
        help=(
            "Parent non-improve streak that triggers plateau escape "
            "(default 8 in main pipeline)."
        ),
    )
    parser.add_argument(
        "--principle-menu-preset",
        type=str,
        default=None,
        choices=list_principle_menu_presets(),
        help=(
            "Optional reduced menu preset: lob_e2, spread_e5, or min2 "
            "(2 principles × 2 form_classes per type)."
        ),
    )
    parser.add_argument(
        "--disable-menu-exploration",
        action="store_true",
        help=(
            "Disable critic menu-exploration nudge (repeat hypothesis pairs allowed). "
            "Matches pre-20260830 PV runs without forced menu coverage."
        ),
    )
    parser.add_argument(
        "--elite-parent-bias-prob",
        type=float,
        default=None,
        help=(
            "Probability of selecting the global elite (highest-surplus) parent "
            f"instead of UCB (default {ELITE_PARENT_BIAS_PROB})."
        ),
    )
    parser.add_argument(
        "--hypothesis-jump-prob",
        type=float,
        default=None,
        help=(
            "On UCB-selected parents, probability of forcing a new hypothesis "
            "with a different principle and/or form_class (default 0)."
        ),
    )
    parser.add_argument(
        "--codegen-recovery-mode",
        type=str,
        default=None,
        choices=["plan_guided", "legacy"],
        help=(
            "Codegen debug recovery style: legacy (July-style local parent edits, no "
            "diagnosis gate; frozen default) or plan_guided (implement Parent Reflection "
            "DIAGNOSIS + diagnosis gate)."
        ),
    )
    parser.add_argument(
        "--prompt-profile",
        type=str,
        default="standard",
        choices=["standard"],
        help="Prompt stack.",
    )
    parser.add_argument(
        "--lob-input-mode",
        type=str,
        default=None,
        choices=[
            "raw",
            "hbl_parity",
            "book_orders",
            "features",
            "features_full",
            "features_f3",
            "features_f1",
        ],
        help=(
            "LOB strategy input: raw nested history; hbl_parity (HBL information-set "
            "schema lock); book_orders (Spread-like BBO/PV/F̂ + L-window orders, "
            "L not exposed); features/features_full (F9 scalars); features_f3 "
            "(spread_rel, log_pv, imbalance); features_f1 (log_pv only)."
        ),
    )
    parser.add_argument(
        "--spread-input-mode",
        type=str,
        default=None,
        choices=["classic", "zip_parity", "book"],
        help=(
            "Spread strategy input: classic (noisy fundamental_estimate + shouts), "
            "zip_parity (ZIP info-set with 'fundamental' + shouts), or book "
            "(BBO/inventory/PV/F̂ only — NO shout tape; recommended for new runs). "
            "All modes use noisy F̂ for input and pricing (never latent true F_t). "
            "Default classic; does not change PV/LOB."
        ),
    )
    parser.add_argument(
        "--spread-omit-fundamental",
        action="store_true",
        help=(
            "Omit F̂ from Spread market_state / schema / seed / prompts for this run "
            "(classic+book: fundamental_estimate; zip_parity: fundamental). "
            "Pricing still uses noisy F̂ internally. Optional; off by default."
        ),
    )
    parser.add_argument(
        "--background-agent-kind",
        type=str,
        default=None,
        choices=["zi", "zip", "zip_noisy", "zip_fixed", "hbl", "hbl_fast"],
        help=(
            "Background population in each episode: zi (default Zero-Intelligence), "
            "zip (Cliff-Bruten ZIP; coerced to zip_noisy when strategy_type=spread), "
            "zip_noisy (ZIP with estimate_fundamental), "
            "zip_fixed (noisy-F ZIP with |μ| frozen at market shade_hi, "
            "or at --background-zip-fixed-mu when set), "
            "hbl (History-Based Learning), or hbl_fast "
            "(behavior-identical faster HBL). Counts still come from "
            "num_zi_buy/num_zi_sell. Default zi."
        ),
    )
    parser.add_argument(
        "--background-zip-fixed-mu",
        type=float,
        default=None,
        help=(
            "When --background-agent-kind=zip_fixed, freeze all background "
            "ZIP seats at this absolute |μ| (e.g. 0.0015 = top competitive "
            "ladder rung). Default: use each market's shade_hi."
        ),
    )
    parser.add_argument(
        "--background-zip-zi-warmup-steps",
        type=int,
        default=25,
        help=(
            "For ZIP backgrounds (zip / zip_noisy / zip_fixed): first N "
            "timesteps each background ZIP seat posts ZI limits while μ still "
            "learns from shouts (HBL-style warm-start, time-gated). After N "
            "steps the seat shouts with the updated ZIP margin; zip_fixed then "
            "freezes μ. When --background-zip-initial-mu is set, μ is snapped "
            "to that |μ| at the end of the ZI window (no μ train during ZI). "
            "Set 0 to disable. Default 25."
        ),
    )
    parser.add_argument(
        "--background-zip-initial-mu",
        type=float,
        default=None,
        help=(
            "For adaptive ZIP backgrounds (zip / zip_noisy): start (and snap "
            "after ZI warm-start) at this absolute |μ| (e.g. 0.0015), then "
            "continue learning. Ignored for zip_fixed (use "
            "--background-zip-fixed-mu). Default: random μ_init_range."
        ),
    )
    parser.add_argument(
        "--parallel-episodes",
        action="store_true",
        help=(
            "Run creation/holdout LLM episodes in a process pool (cross-episode). "
            "Completes all LLM episodes before any ZI benchmark cohort or Q refresh "
            "so workers are not shared across stages. Default off."
        ),
    )
    parser.add_argument(
        "--parallel-episode-workers",
        type=int,
        default=None,
        help=(
            "Process workers for --parallel-episodes "
            "(default min(60, cpu_count, n_episodes)). Ignored unless "
            "--parallel-episodes is set."
        ),
    )
    parser.add_argument(
        "--skip-zi-benchmarks",
        action="store_true",
        help=(
            "Skip ZI shade-cohort diagnostics on creation/seed eval even when "
            "q_reward_mode is zi_advantage or zi_beat_fraction. Q uses advantage "
            "only; saves the large ZI process pool."
        ),
    )
    parser.add_argument(
        "--q-refresh-top-k",
        type=int,
        default=None,
        help=(
            "Max nodes to re-evaluate on each full Q refresh (default 30). "
            "Uses --q-refresh-always-recent newest nodes plus highest-Q fill."
        ),
    )
    parser.add_argument(
        "--q-refresh-always-recent",
        type=int,
        default=None,
        help=(
            "Always refresh this many newest node ids each Q refresh "
            "(default 3), then fill remaining top-k slots by q_value."
        ),
    )
    parser.add_argument(
        "--q-refresh-episode-workers",
        type=int,
        default=None,
        help=(
            "Process workers per refreshed node for episode fan-out "
            "(default 2). Outer pool size equals the number of refreshed nodes."
        ),
    )
    parser.add_argument(
        "--lob-seed-scale",
        type=str,
        default=None,
        choices=["classic", "zi_scale", "hbl", "hbl_scaffold", "basic"],
        help=(
            "LOB order-memory seed: classic (default high shade), "
            "zi_scale (trade-capable ~0.01–0.08 band), hbl (compact "
            "HBL-like belief+grid), hbl_scaffold (methodology o_LOB foothold: "
            "BBO/PV/F̃/books/orders, fixed shade 0.01), or basic "
            "(same book_orders foothold when lob_input_mode=book_orders). "
            "Features-mode seeds unchanged."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ALIAS,
        choices=sorted(MODEL_ALIASES.keys()),
        help=(
            "Which already-running vLLM model to call: "
            "llama (models/Meta-Llama-3-8B-Instruct) or "
            "qwen (models/Qwen2.5-Coder-32B-Instruct). "
            "Start the matching server yourself on :8000."
        ),
    )
    return parser.parse_args()


def _resolve_pipeline_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    recipe = get_recipe(args.recipe) if args.recipe else None

    def pick(cli_value: Any, recipe_value: Any, fallback: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if recipe_value is not None:
            return recipe_value
        return fallback

    batch_size = pick(args.batch_size, recipe.batch_size if recipe else None, 2)
    steps = pick(args.steps, recipe.steps if recipe else None, 3)
    episodes = pick(
        args.episodes_per_config,
        recipe.episodes_per_config if recipe else None,
        2,
    )
    ucb = pick(args.ucb_constant, recipe.ucb_constant if recipe else None, 3.0)
    reward_mode = pick(args.reward_mode, recipe.reward_mode if recipe else None, "advantage")
    strategy_output_mode = pick(
        args.strategy_output_mode,
        recipe.strategy_output_mode if recipe else None,
        "x_cap",
    )
    init_types_raw = pick(
        args.init_types,
        ",".join(recipe.init_types) if recipe else None,
        "pv,spread,lob",
    )
    init_types = [t.strip() for t in str(init_types_raw).split(",") if t.strip()]

    critic_enabled = True
    if args.disable_critic:
        critic_enabled = False
    elif recipe is not None:
        critic_enabled = recipe.critic_enabled

    critic_include_data = True
    if args.no_critic_data:
        critic_include_data = False
    elif recipe is not None:
        critic_include_data = recipe.critic_include_data

    critic_per_config = args.critic_per_config_metrics or (
        recipe.critic_per_config_metrics if recipe else False
    )

    generator_critic_data = args.generator_critic_data
    if not generator_critic_data and recipe is not None and recipe.generator_include_critic_data is not None:
        generator_critic_data = recipe.generator_include_critic_data

    generator_sibling = args.generator_sibling
    if not generator_sibling and recipe is not None and recipe.generator_include_sibling is not None:
        generator_sibling = recipe.generator_include_sibling

    q_reward_mode = pick(args.q_reward_mode, recipe.q_reward_mode if recipe else None, None)
    parent_selection = pick(
        args.parent_selection,
        recipe.parent_selection_mode if recipe else None,
        None,
    )
    q_eval_mode = pick(args.q_eval_mode, recipe.q_eval_mode if recipe else None, None)
    q_update_mode = pick(args.q_update_mode, recipe.q_update_mode if recipe else None, None)

    if args.disable_inactive_gate:
        inactive_gate_enabled = False
    elif recipe is not None:
        inactive_gate_enabled = recipe.inactive_gate_enabled
    else:
        inactive_gate_enabled = True

    if args.smoke:
        batch_size = 1
        steps = 1
        episodes = 1
        critic_enabled = False

    q_suite_size = max(1, int(args.q_suite_size))
    menu_sweep_on_home_set = bool(args.menu_sweep_on_home_set)
    home_set_size = max(0, int(args.home_set_size))
    resample_every = max(0, int(args.resample_every))
    if menu_sweep_on_home_set and home_set_size <= 0:
        raise SystemExit("--menu-sweep-on-home-set requires --home-set-size > 0.")

    return {
        "batch_size": batch_size,
        "steps": steps,
        "episodes_per_config": episodes,
        "exploration_constant": ucb,
        "max_debug_retries": 1 if args.smoke else args.max_debug_retries,
        "storage_dir": args.storage_dir,
        "output_path": args.output,
        "seed": args.seed,
        "strategy_output_mode": strategy_output_mode,
        "reward_mode": reward_mode,
        "suite_aggregate": pick(args.suite_aggregate, None, "mean"),
        "suite_aggregate_lambda": pick(args.suite_aggregate_lambda, None, 0.5),
        "fixed_strategy_code": None,
        "critic_enabled": critic_enabled,
        "critic_include_data": critic_include_data,
        "critic_per_config_metrics": critic_per_config,
        "generator_include_critic_data": generator_critic_data,
        "generator_include_sibling": generator_sibling,
        "run_dir": args.run_dir,
        "initial_strategy_types": init_types,
        "q_eval_mode": q_eval_mode,
        "q_update_mode": q_update_mode,
        "q_reward_mode": q_reward_mode,
        "parent_selection_mode": parent_selection,
        "recipe_id": recipe.recipe_id if recipe else None,
        "inactive_gate_enabled": inactive_gate_enabled,
        "inactive_gate_min_nodes": (
            args.inactive_gate_min_nodes
            if args.inactive_gate_min_nodes is not None
            else 10
        ),
        "inactive_surplus_gap": (
            args.inactive_surplus_gap
            if args.inactive_surplus_gap is not None
            else 300.0
        ),
        "inactive_beat_gap": (
            args.inactive_beat_gap if args.inactive_beat_gap is not None else 3
        ),
        "critic_mode": pick(
            args.critic_mode,
            recipe.critic_mode if recipe is not None else None,
            "default",
        ),
        "plateau_escape_mode": pick(
            args.plateau_escape_mode,
            None,
            "revision",
        ),
        "plateau_non_improve_threshold": pick(
            args.plateau_non_improve_threshold,
            None,
            PLATEAU_NON_IMPROVE_THRESHOLD,
        ),
        "principle_menu_preset": args.principle_menu_preset,
        "menu_exploration_enabled": not args.disable_menu_exploration,
        "elite_parent_bias_prob": pick(
            args.elite_parent_bias_prob,
            None,
            ELITE_PARENT_BIAS_PROB,
        ),
        "hypothesis_jump_prob": pick(
            args.hypothesis_jump_prob,
            None,
            HYPOTHESIS_JUMP_PROB,
        ),
        "codegen_recovery_mode": pick(
            args.codegen_recovery_mode,
            None,
            DEFAULT_CODEGEN_RECOVERY_MODE,
        ),
        "prompt_profile": args.prompt_profile,
        "lob_input_mode": pick(
            args.lob_input_mode,
            None,
            "raw",
        ),
        "spread_input_mode": pick(
            args.spread_input_mode,
            None,
            "classic",
        ),
        "spread_omit_fundamental": bool(args.spread_omit_fundamental),
        "background_agent_kind": pick(
            args.background_agent_kind,
            None,
            "zi",
        ),
        "background_zip_fixed_mu": (
            float(args.background_zip_fixed_mu)
            if args.background_zip_fixed_mu is not None
            else None
        ),
        "background_zip_initial_mu": (
            float(args.background_zip_initial_mu)
            if args.background_zip_initial_mu is not None
            else None
        ),
        "background_zip_zi_warmup_steps": int(
            args.background_zip_zi_warmup_steps
        ),
        "parallel_episodes": bool(args.parallel_episodes),
        "parallel_episode_workers": args.parallel_episode_workers,
        "skip_zi_benchmarks": bool(args.skip_zi_benchmarks),
        "q_refresh_top_k": pick(args.q_refresh_top_k, None, Q_REFRESH_TOP_K),
        "q_refresh_always_recent": pick(
            args.q_refresh_always_recent, None, Q_REFRESH_ALWAYS_RECENT
        ),
        "q_refresh_episode_workers": pick(
            args.q_refresh_episode_workers, None, Q_REFRESH_EPISODE_WORKERS
        ),
        "lob_seed_scale": pick(args.lob_seed_scale, None, "classic"),
        "learning_falsify_zero_trade": False,
        "learning_falsify_shade_collapse": False,
        "learning_reject_noop_edits": False,
        "learning_parent_max_children": DEFAULT_LEARNING_PARENT_MAX_CHILDREN,
        "learning_lesson_memory": False,
        "learning_structural_recovery": False,
        "learning_force_menu_replace": False,
        "learning_outcome_first": False,
        "model_name": resolve_model_name(args.model),
        "market_block_size": 0,
        "q_suite_size": q_suite_size,
        "menu_sweep_markets": 0,
        "menu_sweep_on_home_set": menu_sweep_on_home_set,
        "home_set_size": home_set_size,
        "resample_every": resample_every,
        "zi_benchmark_shade_ranges": None,
        "forbid_numpy_random": False,
        "config_pool": None,
    }


def main() -> None:
    args = parse_args()
    kwargs = _resolve_pipeline_kwargs(args)
    config_pool = kwargs.pop("config_pool", None)

    if config_pool is not None:
        pipeline = ReflectionPipeline(
            config_pool=config_pool,
            **kwargs,
        )
    else:
        config_space = build_default_config_space()
        pipeline = ReflectionPipeline(
            config_space=config_space,
            **kwargs,
        )
    pipeline.run()


if __name__ == "__main__":
    main()
