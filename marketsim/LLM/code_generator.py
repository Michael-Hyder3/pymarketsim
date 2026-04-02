"""
Code Generator for LLM-based Function Generation

Handles calling the LLM, parsing Python code from responses,
validating syntax, and returning clean executable code.
"""
import ast
import json
import re
import os
from typing import Tuple, Optional
from openai import OpenAI


class CodeGenerationError(Exception):
    """Raised when code generation fails."""
    pass


class CodeGenerator:
    """Generates Python code using an LLM."""

    DEFAULT_BASE_URL = "http://ra6kb1.cs.rutgers.edu:8000/v1"
    DEFAULT_API_KEY = "dummy"
    DEFAULT_MODEL_NAME = "models/Meta-Llama-3-8B-Instruct"
    MAX_PROMPT_CHARS = 16000
    FALLBACK_BASE_URLS = [
        "http://ra6kb1.cs.rutgers.edu:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
    ]

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the code generator.

        Args:
            base_url: Base URL for the LLM API
            api_key: API key for the LLM
            model_name: Specific model name to use (if None, will detect first available)
        """
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
        """Get the model name, detecting it if necessary."""
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
        """
        Call the LLM and extract Python code from the response.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Creativity parameter (0-1, lower = more deterministic)
            max_tokens: Maximum tokens in the response

        Returns:
            Clean Python code as a string

        Raises:
            CodeGenerationError: If code generation fails
        """
        if len(prompt) > self.MAX_PROMPT_CHARS:
            head = prompt[:7000]
            tail = prompt[-8500:]
            prompt = (
                f"{head}\n\n"
                "[...prompt truncated for context window safety...]\n\n"
                f"{tail}"
            )

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
            code = self._extract_python_code(response_text)

            return code

        except Exception as e:
            raise CodeGenerationError(f"LLM call failed: {e}")

    @staticmethod
    def _extract_python_code(response_text: str) -> str:
        """
        Extract Python code from an LLM response.

        Looks for code blocks marked with ```python ... ```.

        Args:
            response_text: The raw response text from the LLM

        Returns:
            The extracted Python code

        Raises:
            CodeGenerationError: If no Python code block is found
        """
        # Try to find markdown code block with python language specifier
        # Pattern: ```python followed by optional whitespace, then code, then ```
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try to find any code block without language specifier
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            # Return the first match that looks like valid Python
            for match in matches:
                stripped = match.strip()
                # Basic heuristic: if it starts with def, import, or class, it's probably Python
                if stripped and (stripped.startswith(('def ', 'import ', 'from ', 'class '))):
                    return stripped
            # If none look like Python, return the first one anyway
            return matches[0].strip()

        # If no code block found, raise error
        raise CodeGenerationError(
            "No Python code block found in LLM response. "
            "Expected format: ```python ... ``` or ``` ... ```"
        )

    @staticmethod
    def validate_code(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that code is syntactically correct Python.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is None
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax Error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation Error: {str(e)}"

    def generate_and_validate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Generate code and validate it, with automatic retry on validation failure.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Creativity parameter
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retries if validation fails

        Returns:
            Tuple of (code, success, error_message)
            If successful: (code, True, None)
            If failed after retries: (last_attempt_code, False, error_message)
        """
        last_code = ""
        last_error = None

        for attempt in range(max_retries):
            try:
                code = self.generate_code(prompt, temperature, max_tokens)
                last_code = code

                is_valid, error = self.validate_code(code)
                if is_valid:
                    return code, True, None

                # Validation failed, prepare for retry
                last_error = error
                if attempt < max_retries - 1:
                    # Update prompt to include error feedback
                    prompt = (
                        f"{prompt}\n\n"
                        f"The previous code had a {error}\n"
                        f"Please fix this error and provide corrected code."
                    )

            except CodeGenerationError as e:
                last_error = str(e)

        # All retries exhausted
        return last_code, False, last_error


def generate_function(
    prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> Tuple[str, bool, Optional[str]]:
    """
    Convenience function to generate code in one call.

    Args:
        prompt: The prompt for code generation
        base_url: LLM endpoint URL
        api_key: API key
        temperature: Creativity parameter
        max_retries: Number of retries on validation failure

    Returns:
        Tuple of (code, success, error_message)
    """
    generator = CodeGenerator(base_url=base_url, api_key=api_key)
    return generator.generate_and_validate(prompt, temperature=temperature, max_retries=max_retries)
