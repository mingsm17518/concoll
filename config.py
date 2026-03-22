"""Configuration for Streamed Dreaming Vulnerability Detection."""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Literal, Optional

# Load .env file (override=True means .env takes precedence over system env)
load_dotenv(override=True)


@dataclass
class Config:
    """Configuration class for the project."""

    # API Provider: "openai" or "anthropic"
    api_provider: Literal["openai", "anthropic"] = os.getenv("API_PROVIDER", "anthropic")

    # OpenAI API Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Anthropic API Configuration
    anthropic_api_key: str = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
    anthropic_base_url: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    # Model Configuration
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "MiniMax-M2.1")
    openai_model: str = os.getenv("OPENAI_MODEL", os.getenv("MODEL", "gpt-4o"))

    @property
    def model(self) -> str:
        """Get the appropriate model based on provider."""
        if self.api_provider == "anthropic":
            return self.anthropic_model
        return self.openai_model

    # PrimeVul Dataset Configuration
    dataset_name: str = "ASSERT-KTH/PrimeVul"
    dataset_split: str = "test_paired"

    # Experiment Configuration
    num_candidates: int = 5  # N for Standard Dreaming
    max_samples: int = None  # None = use all samples, set to N for testing
    random_seed: int = 42

    # API Parameters
    temperature: float = 0.0  # Deterministic for vulnerability detection
    max_tokens: int = 2048

    # Output Configuration
    results_dir: str = "results"
    verbose: bool = True

    # Methods
    method: Literal["direct", "dreaming", "streamed"] = "direct"

    @property
    def api_key(self) -> str:
        """Get the appropriate API key based on provider."""
        if self.api_provider == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key

    @property
    def base_url(self) -> str:
        """Get the appropriate base URL based on provider."""
        if self.api_provider == "anthropic":
            return self.anthropic_base_url
        return self.openai_base_url

    def validate(self) -> None:
        """Validate configuration (call explicitly when needed)."""
        if self.api_provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_AUTH_TOKEN not found! Please set it in .env file or environment variable.\n"
                    "Create a .env file with: ANTHROPIC_AUTH_TOKEN=your_key_here"
                )
        else:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found! Please set it in .env file or environment variable.\n"
                    "Create a .env file with: OPENAI_API_KEY=your_key_here"
                )

    @classmethod
    def from_args(cls, **kwargs) -> "Config":
        """Create config from keyword arguments."""
        return cls(**{k: v for k, v in kwargs.items() if v is not None})
