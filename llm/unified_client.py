"""
Unified LLM Client Interface

Supports both Anthropic-compatible APIs (GLM, MiniMax) and OpenAI APIs (GPT-4).

For logprobs support:
- OpenAI API (GPT-4): Full logprobs support
- Anthropic-compatible: Limited support (depends on provider)

Usage:
    # For GLM/MiniMax (current)
    client = UnifiedClient(provider="anthropic", ...)

    # For GPT-4 (when ready to switch)
    client = UnifiedClient(provider="openai", ...)
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import os

from .gpt4o_client import GPT4oClient, TokenUsage

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class UnifiedClient:
    """
    Unified LLM client supporting multiple providers.

    Provider Support:
    - "openai": OpenAI API (GPT-4, supports logprobs)
    - "anthropic": Anthropic-compatible API (GLM, MiniMax)
    """

    def __init__(
        self,
        provider: str = "anthropic",  # or "openai"
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        enable_logprobs: bool = True,  # Request logprobs if supported
    ):
        """
        Initialize unified client.

        Args:
            provider: "openai" or "anthropic"
            api_key: API key
            base_url: API base URL
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            enable_logprobs: Whether to request logprobs (for confidence scoring)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_logprobs = enable_logprobs
        self.usage = TokenUsage()

        if provider == "openai":
            # Use OpenAI client (for GPT-4 with logprobs)
            self.client = GPT4oClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self._supports_logprobs = True

        elif provider == "anthropic":
            # Use Anthropic-compatible client (GLM, MiniMax)
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed")
            self.client = Anthropic(api_key=api_key, base_url=base_url)
            self._supports_logprobs = False  # GLM/MiniMax don't support logprobs

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_logprobs: Optional[int] = None,
        **kwargs
    ) -> tuple[str, Union[TokenUsage, tuple]]:
        """
        Make a chat completion API call.

        Returns:
            tuple: (response_text, token_usage or (token_usage, logprobs_info))
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        if self.provider == "openai":
            # OpenAI: Full logprobs support
            import openai
            response = self.client.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True if self.enable_logprobs else False,
                top_logprobs=top_logprobs if self.enable_logprobs else None,
            )

            content = response.choices[0].message.content

            # Extract logprobs if requested
            if self.enable_logprobs and response.choices[0].logprobs:
                logprobs = response.choices[0].logprobs.content
                if logprobs and len(logprobs) > 0:
                    top_token = logprobs[0].token
                    top_logprob = logprobs[0].logprob

                    if len(logprobs) > 1:
                        second_logprob = logprobs[1].logprob
                    else:
                        second_logprob = -float('inf')

                    # Convert to probabilities
                    import math
                    top_prob = math.exp(max(top_logprob, -100))
                    second_prob = math.exp(max(second_logprob, -100))

                    confidence_score = top_prob - second_prob

                    usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        api_calls=1
                    )

                    logprobs_info = {
                        "confidence_score": confidence_score,
                        "top_token": top_token,
                        "top_probability": top_prob,
                        "second_probability": second_prob,
                        "has_logprobs": True
                    }

                    self.usage += usage
                    return content, (usage, logprobs_info)

            # Fallback without logprobs
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                api_calls=1
            )
            self.usage += usage
            return content, usage

        elif self.provider == "anthropic":
            # Anthropic-compatible: No logprobs support for GLM/MiniMax
            # Convert messages format
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

            response = self.client.messages.create(
                model=self.model,
                system=system_message if system_message else None,
                messages=user_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract content
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
                elif hasattr(block, 'thinking'):
                    # GLM-4.7 uses reasoning_content
                    if hasattr(block, 'reasoning_content'):
                        content += block.reasoning_content

            if not content:
                content = str(response.content[0])

            # No logprobs available
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                api_calls=1
            )
            self.usage += usage

            # Return with placeholder logprobs info
            if self.enable_logprobs:
                logprobs_info = {
                    "confidence_score": 0.0,  # Placeholder
                    "top_token": "N/A",
                    "top_probability": 0.0,
                    "second_probability": 0.0,
                    "has_logprobs": False,
                    "note": "GLM/MiniMax does not support logprobs via this API"
                }
                return content, (usage, logprobs_info)
            else:
                return content, usage

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage across all calls."""
        return self.usage

    def reset_usage(self) -> None:
        """Reset token usage tracking."""
        self.usage = TokenUsage()

    def supports_logprobs(self) -> bool:
        """Check if current provider supports logprobs."""
        return self._supports_logprobs


def create_client_from_config(config) -> UnifiedClient:
    """
    Create UnifiedClient from Config object.

    Usage:
        from config import Config
        config = Config()  # Loads from .env
        client = create_client_from_config(config)
    """
    return UnifiedClient(
        provider=config.api_provider,  # "openai" or "anthropic"
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        enable_logprobs=True  # Always enable for confidence scoring
    )
