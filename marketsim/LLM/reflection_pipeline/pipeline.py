from __future__ import annotations

import json
import os
import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from marketsim.LLM.code_generator import CodeGenerator
from marketsim.LLM.code_storage import CodeStorage
from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig, build_default_config_pool
from marketsim.LLM.reflection_pipeline.evaluation import evaluate_batch
from marketsim.LLM.reflection_pipeline.offline_example import generate_offline_example, load_offline_example
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_debug_prompt,
    build_reflection_prompt,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.feedback_loop import DEFAULT_MARKET_DESCRIPTION
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier

DEFAULT_STRATEGY_CODE = ""


@dataclass
class GenerationResult:
    code: Optional[str]
    strategy_fn: Optional[Any]
    parent_id: int
    success: bool
    error: Optional[str]
    code_file: Optional[str]


class StrategyGenerator:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._generator = CodeGenerator(base_url=base_url, api_key=api_key, model_name=model_name)

    def generate(
        self,
        *,
        parent_code: str,
        parent_reflection: str,
        batch_context: List[Dict[str, Any]],
        strategy_output_mode: str,
        offline_example: Optional[Dict[str, Any]],
    ) -> str:
        prompt = build_generation_prompt(
            parent_code=parent_code,
            parent_reflection=parent_reflection,
            batch_context=batch_context,
            strategy_output_mode=strategy_output_mode,
            market_description=DEFAULT_MARKET_DESCRIPTION,
            offline_example=offline_example,
        )
        return self._generator.generate_code(prompt)

    def debug(self, *, code: str, error: str) -> str:
        prompt = build_debug_prompt(code=code, error=error)
        return self._generator.generate_code(prompt)


class ReflectionCritic:
    DEFAULT_BASE_URL = "http://aries.cs.rutgers.edu:8000/v1"
    DEFAULT_API_KEY = "dummy"
    DEFAULT_MODEL_NAME = "models/Meta-Llama-3-8B-Instruct"
    FALLBACK_BASE_URLS = [
        "http://aries.cs.rutgers.edu:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
    ]

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or self.DEFAULT_API_KEY
        self._base_url_candidates = [resolved_base_url] + [
            u for u in self.FALLBACK_BASE_URLS if u != resolved_base_url
        ]
        self._api_key = resolved_api_key
        self._model_name = model_name or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL_NAME
        self._client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    def generate(self, *, code: str, batch_summary: Dict[str, Any], reward: float) -> str:
        prompt = build_reflection_prompt(code=code, batch_summary=batch_summary, reward=reward)
        last_error = None
        for candidate_url in self._base_url_candidates:
            try:
                self._client = OpenAI(base_url=candidate_url, api_key=self._api_key)
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": "Return a concise 3-part reflection."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_error = exc
                if "Connection error" in str(exc):
                    continue
                raise
        if last_error is not None:
            raise last_error
        return "Environment: unknown\nPerformance: unknown\nFix: unknown"


class ReflectionPipeline:
    def __init__(
        self,
        *,
        config_pool: Optional[List[ReflectionMarketConfig]] = None,
        batch_size: int = 2,
        steps: int = 3,
        episodes_per_config: int = 2,
        exploration_constant: float = 1.0,
        max_debug_retries: int = 3,
        storage_dir: str = "llm_calls_reflection_pipeline",
        output_path: str = "results/reflection_pipeline_summary.json",
        seed: int = 123,
        strategy_output_mode: str = "x_cap",
        reward_mode: str = "advantage",
        fixed_strategy_code: Optional[str] = None,
        critic_enabled: bool = True,
        run_dir: Optional[str] = None,
    ) -> None:
        self.config_pool = config_pool or build_default_config_pool()
        self.batch_size = max(1, int(batch_size))
        self.steps = max(1, int(steps))
        self.episodes_per_config = max(1, int(episodes_per_config))
        self.exploration_constant = float(exploration_constant)
        self.max_debug_retries = max(1, int(max_debug_retries))
        self.run_dir = run_dir
        resolved_storage_dir, resolved_output_path = self._resolve_run_paths(
            storage_dir=storage_dir,
            output_path=output_path,
        )
        self.storage_dir = resolved_storage_dir
        self.output_path = resolved_output_path
        self.seed = seed
        self.strategy_output_mode = strategy_output_mode
        self.reward_mode = reward_mode
        self.fixed_strategy_code = fixed_strategy_code
        self.critic_enabled = critic_enabled

        self._rng = random.Random(seed)
        self._tree = RefinementTree(root_code=DEFAULT_STRATEGY_CODE)
        self._total_visits = 1
        self._storage = CodeStorage(self.storage_dir)
        self._verifier = StrategyVerifier()
        self._generator = StrategyGenerator()
        self._critic = ReflectionCritic() if critic_enabled else None
        self._code_iteration = 0
        self._offline_example = None
        self._offline_example_path = str(Path(self.output_path).parent / "offline_zi_example.json")
        self._prepare_offline_example()

    def _resolve_run_paths(self, *, storage_dir: str, output_path: str) -> Tuple[str, str]:
        if self.run_dir:
            base_dir = Path(self.run_dir)
        else:
            default_storage = storage_dir == "llm_calls_reflection_pipeline"
            default_output = output_path == "results/reflection_pipeline_summary.json"
            if default_storage and default_output:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_dir = Path(__file__).resolve().parent / "runs" / run_id
            else:
                return storage_dir, output_path

        base_dir.mkdir(parents=True, exist_ok=True)
        resolved_storage = str(base_dir / "llm_calls")
        resolved_output = str(base_dir / "reflection_pipeline_summary.json")
        return resolved_storage, resolved_output

    def _prepare_offline_example(self) -> None:
        self._offline_example = load_offline_example(self._offline_example_path)
        if self._offline_example is not None:
            return
        if not self.config_pool:
            return
        example_config = self.config_pool[0]
        self._offline_example = generate_offline_example(
            config=example_config,
            seed=self.seed,
            output_path=self._offline_example_path,
            max_events=30,
        )

    def _select_batch(self) -> List[ReflectionMarketConfig]:
        if self.batch_size >= len(self.config_pool):
            return list(self.config_pool)
        return self._rng.sample(self.config_pool, self.batch_size)

    def _save_code(
        self,
        *,
        code: str,
        success: bool,
        error: Optional[str],
        task_description: str,
    ) -> str:
        self._code_iteration += 1
        return self._storage.save_code(
            code=code,
            function_name="agent_strategy",
            iteration=self._code_iteration,
            task_description=task_description,
            error_feedback=error,
            success=success,
        )

    def generate_and_verify(self, *, batch_context: List[Dict[str, Any]]) -> GenerationResult:
        parent = self._tree.select_ucb(
            exploration=self.exploration_constant,
            total_visits=self._total_visits,
        )
        if self.fixed_strategy_code is not None:
            code = self.fixed_strategy_code
            ok, err, fn = self._verifier.verify(code)
            code_file = self._save_code(
                code=code,
                success=bool(ok),
                error=err,
                task_description="fixed strategy",
            )
            return GenerationResult(
                code=code,
                strategy_fn=fn,
                parent_id=parent.node_id,
                success=bool(ok),
                error=err,
                code_file=code_file,
            )

        code = self._generator.generate(
            parent_code=parent.code,
            parent_reflection=parent.reflection,
            batch_context=batch_context,
            strategy_output_mode=self.strategy_output_mode,
            offline_example=self._offline_example,
        )

        last_error = None
        strategy_fn = None
        success = False
        code_file = None

        for attempt in range(self.max_debug_retries):
            ok, err, fn = self._verifier.verify(code)
            code_file = self._save_code(
                code=code,
                success=bool(ok),
                error=err,
                task_description=f"reflection step attempt {attempt + 1}",
            )
            if ok:
                success = True
                strategy_fn = fn
                last_error = None
                break
            last_error = err or "Verification failed"
            if attempt < self.max_debug_retries - 1:
                code = self._generator.debug(code=code, error=last_error)

        return GenerationResult(
            code=code,
            strategy_fn=strategy_fn,
            parent_id=parent.node_id,
            success=success,
            error=last_error,
            code_file=code_file,
        )

    def _fallback_reflection(self, reward: float, batch_summary: Dict[str, Any]) -> str:
        top_configs = batch_summary.get("per_config", {})
        summary = ", ".join(
            f"{k}={v:.3f}" for k, v in list(top_configs.items())[:3]
        )
        return "\n".join(
            [
                "Environment: mixed-market batch evaluation.",
                f"Performance: reward={reward:.4f}; per_config={summary}",
                "Fix: tighten shading when surplus lags, relax when ahead.",
            ]
        )

    def reflect_and_update(
        self,
        *,
        code: str,
        reward: float,
        batch_summary: Dict[str, Any],
        parent_id: int,
    ) -> str:
        if self._critic is not None:
            reflection = self._critic.generate(code=code, batch_summary=batch_summary, reward=reward)
        else:
            reflection = self._fallback_reflection(reward, batch_summary)

        self._tree.add_node(parent_id=parent_id, code=code, reflection=reflection, reward=reward)
        self._tree.update_ancestors(start_node_id=parent_id, reward=reward)
        self._total_visits += 1
        return reflection

    def run(self) -> Dict[str, Any]:
        batch_configs = self._select_batch()
        summary: List[Dict[str, Any]] = []

        for step_index in range(self.steps):
            batch_context = [cfg.observable_state() for cfg in batch_configs]
            gen = self.generate_and_verify(batch_context=batch_context)
            if not gen.success or gen.strategy_fn is None:
                summary.append(
                    {
                        "step": step_index + 1,
                        "success": False,
                        "error": gen.error,
                        "parent_id": gen.parent_id,
                        "code_file": gen.code_file,
                    }
                )
                continue

            eval_result = evaluate_batch(
                strategy_fn=gen.strategy_fn,
                batch_configs=batch_configs,
                episodes_per_config=self.episodes_per_config,
                step_index=step_index,
                base_seed=self.seed,
                reward_mode=self.reward_mode,
                max_events_per_episode=40,
                strategy_output_mode=self.strategy_output_mode,
            )
            batch_summary = {
                "per_config": eval_result.per_config_rewards,
                "reward": eval_result.reward,
            }
            reflection = self.reflect_and_update(
                code=gen.code,
                reward=eval_result.reward,
                batch_summary=batch_summary,
                parent_id=gen.parent_id,
            )

            summary.append(
                {
                    "step": step_index + 1,
                    "success": True,
                    "parent_id": gen.parent_id,
                    "reward": eval_result.reward,
                    "per_config_rewards": eval_result.per_config_rewards,
                    "reflection": reflection,
                    "code_file": gen.code_file,
                    "batch_data": eval_result.batch_data,
                }
            )

        payload = {
            "params": {
                "batch_size": self.batch_size,
                "steps": self.steps,
                "episodes_per_config": self.episodes_per_config,
                "exploration_constant": self.exploration_constant,
                "max_debug_retries": self.max_debug_retries,
                "seed": self.seed,
                "strategy_output_mode": self.strategy_output_mode,
                "reward_mode": self.reward_mode,
            },
            "run_dir": str(Path(self.output_path).parent),
            "batch_configs": [cfg.observable_state() for cfg in batch_configs],
            "summary": summary,
            "tree": self._tree.to_dict(),
        }

        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


def run_reflection_pipeline(**kwargs: Any) -> Dict[str, Any]:
    pipeline = ReflectionPipeline(**kwargs)
    return pipeline.run()
