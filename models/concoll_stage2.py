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
    Retriever for finding similar vulnerability examples using semantic embeddings.

    Uses sentence-transformers to compute embeddings and cosine similarity
    for semantic retrieval of relevant examples.
    """

    def __init__(
        self,
        examples: List[RAGExample],
        top_k: int = 3,
        random_seed: int = 42,
        embedding_model: str = "microsoft/graphcodebert-base",
        use_fallback: bool = True,
        use_semantic: bool = True
    ):
        """
        Initialize RAG Retriever with semantic embeddings.

        Args:
            examples: List of example cases (from training set)
            top_k: Number of examples to retrieve
            random_seed: Random seed for sampling
            embedding_model: Model name for sentence-transformers
            use_fallback: If True, fall back to random sampling if embedding fails
            use_semantic: If True, try to use semantic retrieval; if False, use random
        """
        self.examples = examples
        self.top_k = top_k
        self.random_seed = random_seed
        self.embedding_model_name = embedding_model
        self.use_fallback = use_fallback
        self.use_semantic = use_semantic
        self._model = None
        self._example_embeddings = None
        self._embedding_failed = False

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None and not self._embedding_failed:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                if self.use_fallback:
                    print(f"[RAG] Warning: Failed to load embedding model '{self.embedding_model_name}': {e}")
                    print("[RAG] Falling back to random sampling for retrieval.")
                    self._embedding_failed = True
                else:
                    raise ImportError(
                        f"sentence-transformers is required for semantic RAG. "
                        f"Install it with: pip install sentence-transformers. Error: {e}"
                    )
        return self._model

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        if self._embedding_failed or self._model is None:
            return None
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def _build_example_embeddings(self):
        """Pre-compute embeddings for all examples (done once)."""
        if self._example_embeddings is None:
            texts = [ex.code for ex in self.examples]
            self._example_embeddings = self._get_embeddings(texts)

    def retrieve(self, query_code: str, query_label: Optional[int] = None) -> List[RAGExample]:
        """
        Retrieve semantically similar examples for the query code.

        Uses cosine similarity between query embedding and example embeddings
        to find the most relevant examples.

        Args:
            query_code: The code to find similar examples for
            query_label: Optional label to match (for balanced retrieval)

        Returns:
            List of retrieved examples sorted by relevance
        """
        # Skip semantic retrieval if disabled
        if not self.use_semantic:
            return self._retrieve_random(query_code, query_label)

        # Fall back to random sampling if embedding failed
        if self._embedding_failed or self._model is None:
            return self._retrieve_random(query_code, query_label)

        # Build example embeddings if not already done
        self._build_example_embeddings()

        # Check if embeddings were built successfully
        if self._example_embeddings is None:
            return self._retrieve_random(query_code, query_label)

        # Get query embedding
        query_embedding = self._get_embeddings([query_code])
        if query_embedding is None:
            return self._retrieve_random(query_code, query_label)
        query_embedding = query_embedding[0]

        # Compute similarities for all examples
        similarities = []
        for i, ex in enumerate(self.examples):
            sim = self._compute_cosine_similarity(query_embedding, self._example_embeddings[i])
            similarities.append((i, sim, ex))

        # Separate by label for balanced retrieval
        vuln_candidates = [(i, sim, ex) for i, sim, ex in similarities if ex.label == 1]
        safe_candidates = [(i, sim, ex) for i, sim, ex in similarities if ex.label == 0]

        # Sort by similarity (descending)
        vuln_candidates.sort(key=lambda x: x[1], reverse=True)
        safe_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select balanced examples (half vulnerable, half safe)
        selected = []
        num_vuln = self.top_k // 2
        num_safe = self.top_k - num_vuln

        # Add top similar vulnerable examples
        for i in range(min(num_vuln, len(vuln_candidates))):
            selected.append(vuln_candidates[i][2])

        # Add top similar safe examples
        for i in range(min(num_safe, len(safe_candidates))):
            selected.append(safe_candidates[i][2])

        # If we don't have enough, fill from remaining (sorted by similarity)
        if len(selected) < self.top_k:
            all_selected_idx = set(id(ex) for ex in selected)
            remaining = [c for c in similarities if id(c[2]) not in all_selected_idx]
            remaining.sort(key=lambda x: x[1], reverse=True)

            for _, _, ex in remaining[:self.top_k - len(selected)]:
                selected.append(ex)

        # Shuffle to avoid order bias
        random.shuffle(selected)

        return selected

    def _retrieve_random(self, query_code: str, query_label: Optional[int] = None) -> List[RAGExample]:
        """
        Fallback random retrieval (used when embedding model fails).

        Args:
            query_code: The code to find similar examples for (ignored)
            query_label: Optional label to match (for balanced retrieval)

        Returns:
            List of randomly selected examples
        """
        vuln_examples = [e for e in self.examples if e.label == 1]
        safe_examples = [e for e in self.examples if e.label == 0]

        selected = []
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

        while len(selected) < self.top_k and len(self.examples) > len(selected):
            remaining = [e for e in self.examples if e not in selected]
            selected.extend(random.sample(remaining, min(self.top_k - len(selected), len(remaining))))

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
        confidence_threshold: float = 0.3,
        verbose: bool = True
    ):
        """
        Initialize RAG Predictor.

        Args:
            client: LLM client
            retriever: RAGRetriever for finding examples
            confidence_threshold: Threshold for Stage 2 acceptance (th2)
            verbose: Whether to print progress
        """
        self.client = client
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.name = "concoll_stage2"

    def predict(
        self,
        code: str,
        true_label: Optional[int] = None
    ) -> Tuple[int, object, List["RAGExample"], bool]:
        """
        Make prediction with RAG context.

        Args:
            code: Source code to analyze
            true_label: Optional true label for better retrieval

        Returns:
            Tuple of (prediction, token_usage, examples, should_accept)
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

        # Make prediction with logprobs
        response, usage = self.client.chat_completion(
            messages,
            max_tokens=10,
            temperature=0,
            top_logprobs=5
        )

        # Parse response
        response_lower = response.strip().lower()
        top_token = "Yes" if 'yes' in response_lower[:10] else "No"
        if 'yes' in response_lower[:10]:
            prediction = 1
        else:
            prediction = 0

        # Get confidence info if available
        confidence_score = 0.15  # Default low confidence for non-logprobs
        top_probability = 0.5
        second_probability = 0.35
        if isinstance(usage, tuple) and len(usage) > 1:
            logprobs_info = usage[1]
            if isinstance(logprobs_info, dict):
                confidence_score = logprobs_info.get("confidence_score", logprobs_info.get("confidence", 0.15))
                top_probability = logprobs_info.get("top_probability", 0.5)
                second_probability = logprobs_info.get("second_probability", 0.35)
                # Clean and validate top_token
                api_top_token = logprobs_info.get("top_token", "")
                cleaned_token = api_top_token.strip().strip('"\'').lower()
                if cleaned_token in ["yes", "no"]:
                    top_token = cleaned_token.capitalize()

        # Check if should accept (according to paper: C.S. >= th2 and top-1 is Yes/No)
        should_accept = self.should_accept(confidence_score, top_token)

        return prediction, usage, examples, should_accept

    def should_accept(self, confidence_score: float, top_token: str) -> bool:
        """
        Decide whether to accept prediction based on confidence.

        According to the paper: "If the score exceeds a second predefined threshold
        (th2) and the top-1 prediction corresponds to a valid class label
        ('Yes' or 'No'), the model's decision is approved."

        Args:
            confidence_score: Confidence score from logprobs
            top_token: The top predicted token

        Returns:
            True if confidence score exceeds threshold AND top-1 is Yes/No
        """
        # Check threshold
        if confidence_score < self.confidence_threshold:
            return False

        # Check if top-1 token is Yes or No (case-insensitive)
        top_token_lower = top_token.lower() if top_token else ""
        if top_token_lower in ["yes", "no"]:
            return True

        return False

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
    ) -> Tuple[List[int], object, List[List["RAGExample"]], List[bool], List[dict]]:
        """
        Make predictions for a batch of codes.

        Args:
            codes: List of source code snippets
            labels: True labels (for retrieval)
            indices: Indices of codes to process (None = all)

        Returns:
            Tuple of (predictions, total_usage, examples_list, accepted_list, confidence_details)
        """
        if indices is None:
            indices = range(len(codes))

        # Return results corresponding to indices, not full codes list
        predictions = []
        examples_list = []
        accepted_list = []
        confidence_details = []  # Store Stage 2 confidence
        total_usage = TokenUsage()

        for i, idx in enumerate(indices):
            if self.verbose and (i + 1) % 5 == 0:
                print(f"  Stage 2: Processed {i + 1}/{len(indices)} samples...")

            code = codes[idx]
            label = labels[idx] if idx < len(labels) else None

            prediction, usage, examples, should_accept = self.predict(code, label)
            predictions.append(prediction)
            examples_list.append(examples)
            accepted_list.append(should_accept)

            # Store confidence details
            confidence_score = 0.15
            top_token_val = "Unknown"
            if isinstance(usage, tuple) and len(usage) > 1:
                logprobs_info = usage[1]
                if isinstance(logprobs_info, dict):
                    confidence_score = logprobs_info.get("confidence_score", 0.15)
                    top_token_val = logprobs_info.get("top_token", "Unknown")

            confidence_details.append({
                "confidence_score": confidence_score,
                "top_token": top_token_val,
                "prediction": prediction,
                "accepted": should_accept
            })

            # Handle both simple usage and nested (usage, logprobs_info) tuple
            if isinstance(usage, tuple):
                actual_usage = usage[0] if hasattr(usage[0], 'prompt_tokens') else usage
            else:
                actual_usage = usage
            if actual_usage is not None:
                total_usage += actual_usage

        if self.verbose:
            print(f"  Stage 2: Completed {len(indices)} samples")

        return predictions, total_usage, examples_list, accepted_list, confidence_details

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
