"""ConColl Stage 2: RAG with External Examples."""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
import json

try:
    from ..llm.anthropic_client import AnthropicClient
    from ..llm.gpt4o_client import GPT4oClient
    from ..llm.gpt4o_client import TokenUsage
except ImportError:
    from llm.anthropic_client import AnthropicClient
    from llm.gpt4o_client import GPT4oClient
    from llm.gpt4o_client import TokenUsage


@dataclass
class RAGExample:
    """Retrieved example for RAG."""
    code: str
    label: int  # 1 = vulnerable, 0 = safe
    cwe: str
    description: str


class RAGRetriever:
    """
    Retriever for finding similar vulnerability examples.

    Uses simple similarity metrics (can be enhanced with embeddings).
    """

    def __init__(
        self,
        examples: List[RAGExample],
        top_k: int = 3,
        random_seed: int = 42
    ):
        """
        Initialize RAG Retriever.

        Args:
            examples: List of example cases (from training set)
            top_k: Number of examples to retrieve
            random_seed: Random seed for sampling
        """
        self.examples = examples
        self.top_k = top_k
        self.random_seed = random_seed

    def retrieve(self, query_code: str, query_label: Optional[int] = None) -> List[RAGExample]:
        """
        Retrieve similar examples for the query code.

        Args:
            query_code: The code to find similar examples for
            query_label: Optional label to match (for balanced retrieval)

        Returns:
            List of retrieved examples
        """
        # Balanced retrieval: always return mixed examples (both vulnerable and safe)
        # This prevents model bias from seeing only one type of example

        vuln_examples = [e for e in self.examples if e.label == 1]
        safe_examples = [e for e in self.examples if e.label == 0]

        selected = []
        # Select balanced examples
        num_vuln = self.top_k // 2
        num_safe = self.top_k - num_vuln

        if len(vuln_examples) >= num_vuln:
            selected.extend(random.sample(vuln_examples, num_vuln))
        else:
            selected.extend(vuln_examples)

        if len(safe_examples) >= num_safe:
            selected.extend(random.sample(safe_examples, num_safe))
        else:
            selected.extend(safe_examples)

        # If we still don't have enough, fill from all
        while len(selected) < self.top_k and len(self.examples) > len(selected):
            remaining = [e for e in self.examples if e not in selected]
            selected.extend(random.sample(remaining, min(self.top_k - len(selected), len(remaining))))

        # Shuffle to avoid order bias
        random.shuffle(selected)

        return selected


class RAGPredictor:
    """
    ConColl Stage 2: RAG-based Prediction.

    Uses retrieved examples to provide context for better predictions.
    """

    SYSTEM_PROMPT = """You are a code vulnerability detection expert.
Your task is to analyze C code by comparing it with similar examples to determine if it contains security vulnerabilities.

Key vulnerability indicators:
- Buffer operations without bounds checking
- Unsafe string functions (strcpy, strcat, sprintf)
- Missing null termination checks
- Integer overflow/underflow risks
- Use-after-free patterns
- Missing input validation

IMPORTANT: Be balanced in your assessment. Use the examples as reference but make your own judgment."""

    USER_PROMPT_TEMPLATE = """Your task is to assess whether the provided code contains any security vulnerabilities. \n\n Here are some examples: \n\n {examples} \n\n Is the following code vulnerable? Respond with only 'Yes' or 'No'. \n\n {code}"""

    def __init__(
        self,
        client,
        retriever: RAGRetriever,
        verbose: bool = True
    ):
        """
        Initialize RAG Predictor.

        Args:
            client: LLM client
            retriever: RAGRetriever for finding examples
            verbose: Whether to print progress
        """
        self.client = client
        self.retriever = retriever
        self.verbose = verbose
        self.name = "concoll_stage2"

    def predict(
        self,
        code: str,
        true_label: Optional[int] = None
    ) -> Tuple[int, object, List["RAGExample"]]:
        """
        Make prediction with RAG context.

        Args:
            code: Source code to analyze
            true_label: Optional true label for better retrieval

        Returns:
            Tuple of (prediction, token_usage, examples)
        """
        # Retrieve similar examples
        examples = self.retriever.retrieve(code, true_label)

        # Format examples into prompt
        examples_text = self._format_examples(examples)

        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(
                code=code,
                examples=examples_text
            )}
        ]

        # Make prediction
        response, usage = self.client.chat_completion(messages)

        # Parse response
        response_lower = response.strip().lower()
        if 'yes' in response_lower[:10]:
            prediction = 1
        else:
            prediction = 0

        return prediction, usage, examples

    def _format_examples(self, examples: List[RAGExample]) -> str:
        """Format examples into prompt text."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            label_str = "VULNERABLE" if ex.label == 1 else "SAFE"
            formatted.append(f"""
Example {i} ({label_str}, {ex.cwe}):
{ex.description}

```c
{ex.code[:500]}...
```
""")
        return "\n".join(formatted)

    def predict_batch(
        self,
        codes: List[str],
        labels: List[int],
        indices: List[int] = None
    ) -> Tuple[List[int], object, List[List["RAGExample"]]]:
        """
        Make predictions for a batch of codes.

        Args:
            codes: List of source code snippets
            labels: True labels (for retrieval)
            indices: Indices of codes to process (None = all)

        Returns:
            Tuple of (predictions, total_usage, examples_list)
        """
        if indices is None:
            indices = range(len(codes))

        predictions = [None] * len(codes)
        examples_list = [[] for _ in range(len(codes))]
        total_usage = TokenUsage()

        for i, idx in enumerate(indices):
            if self.verbose and (i + 1) % 5 == 0:
                print(f"  Stage 2: Processed {i + 1}/{len(indices)} samples...")

            code = codes[idx]
            label = labels[idx] if idx < len(labels) else None

            prediction, usage, examples = self.predict(code, label)
            predictions[idx] = prediction
            examples_list[idx] = examples
            # Handle both simple usage and nested (usage, logprobs_info) tuple
            if isinstance(usage, tuple):
                actual_usage = usage[0] if hasattr(usage[0], 'prompt_tokens') else usage
            else:
                actual_usage = usage
            total_usage += actual_usage

        if self.verbose:
            print(f"  Stage 2: Completed {len(indices)} samples")

        return predictions, total_usage, examples_list

    def get_name(self) -> str:
        """Get method name."""
        return self.name


def create_rag_examples_from_samples(samples) -> List[RAGExample]:
    """
    Create RAG examples from PrimeVul samples.

    Args:
        samples: List of VulnerabilitySample objects

    Returns:
        List of RAGExample objects
    """
    examples = []

    # Add vulnerable examples
    for s in samples:
        examples.append(RAGExample(
            code=s.vulnerable_code[:1000],  # Truncate for context
            label=1,
            cwe=s.cwe,
            description=f"This code contains a {s.cwe} vulnerability."
        ))

    # Add safe examples (fixed code)
    for s in samples:
        examples.append(RAGExample(
            code=s.fixed_code[:1000],
            label=0,
            cwe=s.cwe,
            description=f"This code has been fixed and is now safe."
        ))

    return examples
