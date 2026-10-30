import ast
import json
import re
import os
from typing import Tuple, Optional
from openai import OpenAI


class CodeGenerationError(Exception):
    pass


class CodeGenerator:
    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
    DEFAULT_API_KEY = "dummy"
    DEFAULT_MODEL_NAME = "models/Meta-Llama-3-8B-Instruct"
    MAX_PROMPT_CHARS = 16000
    FALLBACK_BASE_URLS = [
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
    ]

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or self.DEFAULT_API_KEY

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url
        self._base_url_candidates = [resolved_base_url] + [
            u for u in self.FALLBACK_BASE_URLS if u != resolved_base_url
        ]
        self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
        self._model_name = model_name or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL_NAME

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            try:
                models = self.client.models.list()
                if models.data:
                    self._model_name = models.data[0].id
                else:
                    raise CodeGenerationError("No models available from LLM endpoint")
            except Exception as e:
                raise CodeGenerationError(f"Failed to detect model: {e}")
        return self._model_name

    def generate_code(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        if len(prompt) > self.MAX_PROMPT_CHARS:
            prompt = self._truncate_prompt_preserving_reflection(prompt)

        try:
            response = None
            last_error = None
            for candidate_url in self._base_url_candidates:
                try:
                    self.client = OpenAI(base_url=candidate_url, api_key=self.api_key)
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert Python developer. Generate clean, production-ready code.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self.base_url = candidate_url
                    break
                except Exception as inner_error:
                    last_error = inner_error
                    if "Connection error" in str(inner_error):
                        continue
                    raise

            if response is None and last_error is not None:
                raise last_error

            response_text = response.choices[0].message.content
            code = self._extract_strategy_payload(response_text)

            return code

        except Exception as e:
            raise CodeGenerationError(f"LLM call failed: {e}")

    def _truncate_prompt_preserving_reflection(self, prompt: str) -> str:
        budget = self.MAX_PROMPT_CHARS
        original_len = len(prompt)
        refl_headers = (
            "## Parent Hypothesis (DIAGNOSIS — implement this experiment)",
            "## Prior Parent Reflection",
            "## Parent reflection",
        )
        refl_start = -1
        refl_header = None
        for header in refl_headers:
            idx = prompt.find(header)
            if idx >= 0:
                refl_start = idx
                refl_header = header
                break

        if refl_start >= 0:
            after = prompt.find("\n## ", refl_start + len(refl_header or ""))
            refl_end = len(prompt) if after < 0 else after
            reflection = prompt[refl_start:refl_end]
            head_budget = max(2000, (budget - len(reflection)) // 2)
            tail_budget = max(2000, budget - len(reflection) - head_budget - 120)
            head = prompt[:head_budget]
            cut = head.rfind("\n## ")
            if cut > head_budget // 2:
                head = prompt[:cut]
            tail = prompt[-tail_budget:]
            cut_t = tail.find("\n## ")
            if 0 <= cut_t < 200:
                tail = tail[cut_t + 1 :]
            out = (
                f"{head.rstrip()}\n\n"
                "[...prompt truncated for context window safety "
                "(case cards / filler); parent reflection kept...]\n\n"
                f"{reflection.strip()}\n\n"
                f"{tail.lstrip()}"
            )
            if len(out) > budget:
                keep_refl = reflection
                room = budget - len(keep_refl) - 100
                out = (
                    f"{prompt[: max(500, room // 3)].rstrip()}\n\n"
                    "[...prompt truncated; parent reflection kept...]\n\n"
                    f"{keep_refl.strip()}\n\n"
                    f"{prompt[-(room - room // 3):].lstrip()}"
                )
        else:
            head = prompt[:7000]
            tail = prompt[-8500:]
            out = (
                f"{head}\n\n"
                "[...prompt truncated for context window safety...]\n\n"
                f"{tail}"
            )

        print(
            f"[codegen] WARNING: generator prompt truncated "
            f"({original_len}->{len(out)} chars, budget={budget}). "
            f"Reflection preserved={refl_start >= 0}.",
            flush=True,
        )
        return out

    @staticmethod
    def _extract_strategy_payload(response_text: str) -> str:
        try:
            from marketsim.LLM.reflection_pipeline.code_edit import (
                extract_code_or_diff_from_response,
            )

            return extract_code_or_diff_from_response(response_text)
        except ValueError as exc:
            raise CodeGenerationError(str(exc)) from exc

    _extract_python_code = _extract_strategy_payload




