"""GPT-4o API Client wrapper with token tracking."""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class TokenUsage:
    """Track token usage and costs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    # GPT-4o pricing (as of 2024, per 1M tokens)
    prompt_price_per_1m: float = 2.50
    completion_price_per_1m: float = 10.00

    @property
    def estimated_cost(self) -> float:
        """Calculate estimated cost in USD."""
        prompt_cost = (self.prompt_tokens / 1_000_000) * self.prompt_price_per_1m
        completion_cost = (self.completion_tokens / 1_000_000) * self.completion_price_per_1m
        return prompt_cost + completion_cost

    def add(self, prompt: int, completion: int) -> None:
        """Add token usage."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.api_calls += 1

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Combine two TokenUsage instances."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            api_calls=self.api_calls + other.api_calls,
        )


class GPT4oClient:
    """Wrapper for OpenAI GPT-4o API with token tracking."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        """Initialize the GPT-4o client."""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = TokenUsage()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> tuple[str, TokenUsage]:
        """
        Make a chat completion API call.

        Returns:
            tuple: (response_text, token_usage)
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs
            )

            content = response.choices[0].message.content

            # Track usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            call_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                api_calls=1
            )
            self.usage += call_usage

            return content, call_usage

        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage across all calls."""
        return self.usage

    def reset_usage(self) -> None:
        """Reset token usage tracking."""
        self.usage = TokenUsage()


def format_prompt_for_vulnerability_detection(code: str) -> List[Dict[str, str]]:
    """
    Format prompt for binary vulnerability detection.

    Args:
        code: The source code to analyze

    Returns:
        List of message dictionaries for ChatCompletion API
    """
    system_prompt = """You are a security expert specialized in identifying vulnerabilities in source code.

Your task is to analyze the given code and determine whether it contains a security vulnerability.

Respond with ONLY one word:
- "VULNERABLE" if the code contains a security vulnerability
- "SAFE" if the code does not contain any security vulnerability

Do not provide any explanation or additional text."""

    user_prompt = f"""Analyze the following code for security vulnerabilities:

```c
{code}
```

Is this code VULNERABLE or SAFE?"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def format_prompt_for_cwe_classification(code: str) -> List[Dict[str, str]]:
    """
    Format prompt for CWE classification.

    Args:
        code: The source code to analyze

    Returns:
        List of message dictionaries for ChatCompletion API
    """
    system_prompt = """You are a security expert specialized in classifying vulnerabilities in source code according to CWE (Common Weakness Enumeration).

Your task is to identify the CWE type of the vulnerability in the given code.

Common CWE types include:
- CWE-119: Buffer Errors (buffer overflow, off-by-one, etc.)
- CWE-20: Input Validation
- CWE-125: Out-of-bounds Read
- CWE-787: Out-of-bounds Write
- CWE-190: Integer Overflow or Wraparound
- CWE-476: NULL Pointer Dereference
- CWE-22: Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
- CWE-79: Cross-site Scripting (XSS)
- CWE-89: SQL Injection
- CWE-200: Information Exposure

Respond with ONLY the CWE identifier (e.g., "CWE-119")."""

    user_prompt = f"""Classify the vulnerability in the following code:

```c
{code}
```

What is the CWE type of this vulnerability?"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
