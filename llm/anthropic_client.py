"""Anthropic API Client wrapper with token tracking."""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from anthropic import Anthropic

from .gpt4o_client import TokenUsage


class AnthropicClient:
    """Wrapper for Anthropic API with token tracking."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = "glm-4.7",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        """Initialize the Anthropic client."""
        self.client = Anthropic(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = TokenUsage()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_logprobs: Optional[int] = None,
        **kwargs
    ) -> tuple[str, TokenUsage]:
        """
        Make a chat completion API call.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_logprobs: Number of top log probabilities to return

        Returns:
            tuple: (response_text, token_usage)
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        # Convert OpenAI-style messages to Anthropic format
        system_message = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Build messages for Anthropic (requires alternating user/assistant)
        anthropic_messages = []
        for msg in user_messages:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "system": system_message if system_message else None,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Add top_logprobs if requested (for confidence scoring)
            if top_logprobs is not None:
                api_params["extra_headers"] = {"X-Api-Key": self.client.api_key}

            response = self.client.messages.create(**api_params)

            # Extract text from response (handle both TextBlock and ThinkingBlock)
            content = ""
            logprobs_info = {}

            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
                elif hasattr(block, 'thinking'):
                    content += block.thinking

            if not content:
                content = str(response.content[0])

            # Try to extract logprobs if available
            # Check if response has logprobs information
            if hasattr(response, 'model_extra') or hasattr(response, 'extra'):
                # GLM/MiniMax may return logprobs in different format
                try:
                    # Try to get token logprobs from response
                    if hasattr(response, 'usage') and hasattr(response.usage, 'extra'):
                        # Check for logprobs in usage.extra
                        logprobs_info = response.usage.extra
                except:
                    pass

            # Track usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            call_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                api_calls=1
            )
            self.usage += call_usage

            # Return with logprobs info if available
            if top_logprobs is not None:
                return content, (call_usage, logprobs_info)
            else:
                return content, call_usage

        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage across all calls."""
        return self.usage

    def reset_usage(self) -> None:
        """Reset token usage tracking."""
        self.usage = TokenUsage()


def format_prompt_for_vulnerability_detection_anthropic(code: str) -> List[Dict[str, str]]:
    """
    Format prompt for binary vulnerability detection (Anthropic version).

    Args:
        code: The source code to analyze

    Returns:
        List of message dictionaries
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


def format_prompt_for_cwe_classification_anthropic(code: str) -> List[Dict[str, str]]:
    """
    Format prompt for CWE classification (Anthropic version).

    Args:
        code: The source code to analyze

    Returns:
        List of message dictionaries
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
