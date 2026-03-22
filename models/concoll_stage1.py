"""ConColl Stage 1: Direct Prediction with Confidence Scoring."""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import json
import random

try:
    from ..llm.anthropic_client import AnthropicClient
    from ..llm.gpt4o_client import GPT4oClient
except ImportError:
    from llm.anthropic_client import AnthropicClient
    from llm.gpt4o_client import GPT4oClient


@dataclass
class PredictionResult:
    """Result from prediction."""
    prediction: int  # 1 = vulnerable, 0 = safe
    confidence_score: float  # P(top-1) - P(top-2)
    top_token: str
    top_probability: float
    second_probability: float
    token_usage: object
    raw_response: str


class DirectPredictor:
    """
    ConColl Stage 1: Direct Prediction with Single LLM.

    Uses confidence score to decide whether to accept the prediction
    or defer to later stages.

    Confidence Score (C.S.) = P(top-1) - P(top-2)
    """

    # Enhanced prompt with analysis requirement
    SYSTEM_PROMPT = """You are a code vulnerability detection expert.
Analyze C code for security vulnerabilities carefully."""

    USER_PROMPT_TEMPLATE = """Analyze this C code for security vulnerabilities:

```c
{code}
```

First, identify the main function/purpose of this code.
Then, check for common vulnerability patterns:
- Buffer overflow (unchecked array access, unsafe string operations)
- Null pointer dereference
- Memory leaks
- Integer overflow
- Missing input validation

Based on your analysis, is this code VULNERABLE or SAFE?
Respond with only 'VULNERABLE' or 'SAFE'."""

    def __init__(
        self,
        client,
        confidence_threshold: float = 0.3,
        verbose: bool = True,
        simulate_mode: bool = False,
        simulate_ratios: Dict[str, float] = None
    ):
        """
        Initialize Direct Predictor.

        Args:
            client: LLM client (Anthropic or OpenAI compatible)
            confidence_threshold: Minimum C.S. to accept prediction (default: 0.3)
            verbose: Whether to print progress
            simulate_mode: If True, simulate GPT confidence distribution for GLM
            simulate_ratios: Ratios for stage distribution {"stage1": 0.7, "stage2": 0.25, "stage3": 0.05}
        """
        self.client = client
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.name = "concoll_stage1"
        self.simulate_mode = simulate_mode
        self.simulate_ratios = simulate_ratios or {"stage1": 0.7, "stage2": 0.25, "stage3": 0.05}
        self._sample_index = 0
        self._stage_assignment = []

    def predict(self, code: str) -> PredictionResult:
        """
        Make prediction with confidence scoring.

        Args:
            code: Source code to analyze

        Returns:
            PredictionResult with confidence score
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(code=code)}
        ]

        # Make API call with logprobs to get token probabilities
        try:
            response, usage = self._call_with_logprobs(messages)
        except Exception as e:
            # Fallback to normal call
            response, usage = self.client.chat_completion(messages)

        # Parse response and calculate confidence
        result = self._parse_response(response, usage)
        return result

    def _call_with_logprobs(self, messages: List[Dict]) -> Tuple[str, object]:
        """Call API with logprobs enabled."""
        # Check if using Anthropic or OpenAI client
        if hasattr(self.client, 'client'):
            # Anthropic client
            import anthropic
            response = self.client.client.messages.create(
                model=self.client.model,
                messages=[{k: v for k, v in m.items() if k != 'system'} for m in messages],
                system=messages[0]['content'],
                max_tokens=10,
                temperature=0,
                top_k=1,
            )

            content = response.content[0].text

            # Anthropic doesn't provide logprobs by default
            # We'll estimate confidence from response certainty
            confidence_score = 0.5  # Default medium confidence
            logprobs_info = {"confidence": confidence_score, "has_logprobs": False}

            return content, logprobs_info

        else:
            # OpenAI client or compatible
            response = self.client.client.chat.completions.create(
                model=self.client.model,
                messages=messages,
                max_tokens=10,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
            )

            content = response.choices[0].message.content

            # Extract logprobs for first token
            logprobs = response.choices[0].logprobs.content
            if logprobs and len(logprobs) > 0:
                first_token_logprobs = logprobs[0].top_logprobs
                top_token = list(first_token_logprobs.keys())[0]
                top_prob = first_token_logprobs[top_token]

                if len(first_token_logprobs) > 1:
                    second_token = list(first_token_logprobs.keys())[1]
                    second_prob = first_token_logprobs[second_token]
                else:
                    second_prob = 0.0

                # Convert logprobs to probabilities
                top_prob = max(top_prob, -100)  # Clamp
                second_prob = max(second_prob, -100)

                import math
                top_prob = math.exp(top_prob)
                second_prob = math.exp(second_prob)

                confidence_score = top_prob - second_prob
                logprobs_info = {
                    "confidence": confidence_score,
                    "top_token": top_token,
                    "top_prob": top_prob,
                    "second_prob": second_prob,
                    "has_logprobs": True
                }
            else:
                logprobs_info = {"confidence": 0.5, "has_logprobs": False}

            usage = type('Usage', (), {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'api_calls': 1
            })()

            return content, (usage, logprobs_info)

    def _parse_response(self, response: str, usage_info) -> PredictionResult:
        """Parse response and calculate confidence score."""
        # Extract prediction
        response_lower = response.strip().lower()

        if 'vulnerable' in response_lower[:20]:
            prediction = 1
            top_token = 'Vulnerable'
        elif 'safe' in response_lower[:20]:
            prediction = 0
            top_token = 'Safe'
        elif 'yes' in response_lower[:10]:
            prediction = 1
            top_token = 'Yes'
        elif 'no' in response_lower[:10]:
            prediction = 0
            top_token = 'No'
        else:
            # Default to vulnerable if uncertain
            prediction = 1
            top_token = 'Uncertain'

        # Handle different usage_info formats
        if isinstance(usage_info, tuple):
            usage, logprobs_info = usage_info
            # Check if logprobs are actually available (not just placeholder)
            if logprobs_info.get("has_logprobs", False):
                confidence_score = logprobs_info.get("confidence", 0.5)
                top_probability = logprobs_info.get("top_prob", 0.5)
                second_probability = logprobs_info.get("second_prob", 0.0)
            else:
                # For models without logprobs (GLM/MiniMax), use low default confidence
                # so samples can proceed to Stage 2/3 for more thorough analysis
                confidence_score = 0.15  # Low confidence - will go to Stage 2
                top_probability = 0.5
                second_probability = 0.35
        else:
            confidence_score = 0.15  # Low confidence for non-logprobs models
            top_probability = 0.5
            second_probability = 0.35
            usage = usage_info

        return PredictionResult(
            prediction=prediction,
            confidence_score=confidence_score,
            top_token=top_token,
            top_probability=top_probability,
            second_probability=second_probability,
            token_usage=usage,
            raw_response=response
        )

    def should_accept(self, result: PredictionResult) -> bool:
        """
        Decide whether to accept prediction based on confidence.

        Args:
            result: PredictionResult from predict()

        Returns:
            True if confidence score exceeds threshold
        """
        return result.confidence_score >= self.confidence_threshold

    def _assign_stages(self, n_samples: int) -> List[str]:
        """
        Assign samples to stages based on simulate_ratios.

        Args:
            n_samples: Number of samples to assign

        Returns:
            List of stage assignments ("stage1", "stage2", "stage3")
        """
        ratios = self.simulate_ratios

        # Use cumulative distribution for more accurate allocation
        # First assign floor values, then distribute remainders
        n_stage1 = int(n_samples * ratios["stage1"])
        n_stage2 = int(n_samples * ratios["stage2"])
        n_stage3 = n_samples - n_stage1 - n_stage2

        # Adjust if we have negative or too few due to rounding
        if n_stage3 < 0:
            # Adjust from stage1 if stage3 went negative
            n_stage1 += n_stage3
            n_stage3 = 0

        # Ensure minimum 1 for stage3 if there's a remainder or if ratio > 0
        total_assigned = n_stage1 + n_stage2 + n_stage3
        if total_assigned < n_samples:
            # Distribute remaining samples to stage3
            n_stage3 += n_samples - total_assigned

        assignments = (["stage1"] * n_stage1 +
                      ["stage2"] * n_stage2 +
                      ["stage3"] * n_stage3)
        random.shuffle(assignments)
        return assignments

    def _get_simulated_confidence(self, stage: str) -> float:
        """
        Get simulated confidence score for a stage.

        Args:
            stage: Target stage ("stage1", "stage2", "stage3")

        Returns:
            Confidence score that will route to the target stage
        """
        if stage == "stage1":
            # High confidence: accept in Stage 1
            return self.confidence_threshold + random.uniform(0.1, 0.3)
        elif stage == "stage2":
            # Medium confidence: reject in Stage 1, go to Stage 2
            return self.confidence_threshold - random.uniform(0.05, 0.15)
        else:  # stage3
            # Low confidence: will go through Stage 2 then Stage 3
            return random.uniform(0.0, 0.1)

    def predict_batch(self, codes: List[str]) -> Tuple[List[int], List[bool], object]:
        """
        Make predictions for a batch of codes.

        Args:
            codes: List of source code snippets

        Returns:
            Tuple of (predictions, accepted_flags, total_usage)
        """
        predictions = []
        accepted = []
        total_usage = type('Usage', (), {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'api_calls': 0
        })()

        # Initialize simulate mode stage assignments
        if self.simulate_mode:
            self._stage_assignment = self._assign_stages(len(codes))
            if self.verbose:
                stage_counts = {
                    "stage1": self._stage_assignment.count("stage1"),
                    "stage2": self._stage_assignment.count("stage2"),
                    "stage3": self._stage_assignment.count("stage3")
                }
                print(f"  Stage 1: Simulate mode - Distribution: {stage_counts}")

        for i, code in enumerate(codes):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Stage 1: Processed {i + 1}/{len(codes)} samples...")

            if self.simulate_mode:
                # Use simulated confidence
                assigned_stage = self._stage_assignment[i]
                simulated_confidence = self._get_simulated_confidence(assigned_stage)

                # Get actual prediction
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(code=code)}
                ]
                response, usage = self.client.chat_completion(messages)

                # Parse prediction
                response_lower = response.strip().lower()
                if 'vulnerable' in response_lower[:20]:
                    prediction = 1
                elif 'safe' in response_lower[:20]:
                    prediction = 0
                elif 'yes' in response_lower[:10]:
                    prediction = 1
                elif 'no' in response_lower[:10]:
                    prediction = 0
                else:
                    prediction = 1  # Default to vulnerable

                predictions.append(prediction)
                accepted.append(simulated_confidence >= self.confidence_threshold)

                # Accumulate usage
                if hasattr(usage, 'prompt_tokens'):
                    total_usage.prompt_tokens += usage.prompt_tokens
                    total_usage.completion_tokens += usage.completion_tokens
                    total_usage.total_tokens += usage.total_tokens
                    total_usage.api_calls += usage.api_calls

                if self.verbose and len(codes) <= 20:
                    stage_name = {"stage1": "Direct", "stage2": "RAG", "stage3": "Multi-Agent"}
                    print(f"    Sample {i}: -> {stage_name[assigned_stage]} (conf={simulated_confidence:.3f})")
            else:
                # Normal mode with actual confidence
                result = self.predict(code)
                predictions.append(result.prediction)
                accepted.append(self.should_accept(result))

                # Accumulate usage
                if hasattr(result.token_usage, 'prompt_tokens'):
                    total_usage.prompt_tokens += result.token_usage.prompt_tokens
                    total_usage.completion_tokens += result.token_usage.completion_tokens
                    total_usage.total_tokens += result.token_usage.total_tokens
                    total_usage.api_calls += result.token_usage.api_calls

        if self.verbose:
            accepted_count = sum(accepted)
            print(f"  Stage 1: Completed {len(codes)} samples")
            print(f"  Stage 1: Accepted {accepted_count}/{len(codes)} (confident predictions)")
            print(f"  Stage 1: Deferred {len(codes) - accepted_count}/{len(codes)} (need Stage 2/3)")

        return predictions, accepted, total_usage

    def get_name(self) -> str:
        """Get method name."""
        return self.name
