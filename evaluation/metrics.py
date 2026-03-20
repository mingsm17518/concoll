"""Evaluation metrics for vulnerability detection."""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


@dataclass
class BinaryMetrics:
    """Metrics for binary classification."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def print(self) -> None:
        """Print metrics in a formatted way."""
        print("\n=== Binary Classification Metrics ===")
        print(f"Accuracy:  {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"F1 Score:  {self.f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP={self.tp} | FN={self.fn}")
        print(f"  FP={self.fp} | TN={self.tn}")


@dataclass
class CostMetrics:
    """Token usage and cost metrics."""
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_api_calls: int
    estimated_cost_usd: float

    @classmethod
    def from_token_usage(cls, usage, api_calls: int = 1) -> "CostMetrics":
        """Create from TokenUsage object."""
        return cls(
            total_prompt_tokens=usage.prompt_tokens,
            total_completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            total_api_calls=api_calls,
            estimated_cost_usd=usage.estimated_cost
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def print(self) -> None:
        """Print cost metrics."""
        print("\n=== Cost Metrics ===")
        print(f"Prompt Tokens:     {self.total_prompt_tokens:,}")
        print(f"Completion Tokens: {self.total_completion_tokens:,}")
        print(f"Total Tokens:      {self.total_tokens:,}")
        print(f"API Calls:         {self.total_api_calls:,}")
        print(f"Estimated Cost:    ${self.estimated_cost_usd:.4f}")


@dataclass
class ExperimentResult:
    """Complete experiment results."""
    method: str
    binary_metrics: BinaryMetrics
    cost_metrics: CostMetrics
    num_samples: int
    num_candidates: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "num_samples": self.num_samples,
            "num_candidates": self.num_candidates,
            "binary_metrics": self.binary_metrics.to_dict(),
            "cost_metrics": self.cost_metrics.to_dict(),
        }

    def save(self, output_dir: str, filename: Optional[str] = None) -> Path:
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{self.method}_results.json"

        filepath = output_path / filename
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def print(self) -> None:
        """Print complete results."""
        print(f"\n{'='*50}")
        print(f"Method: {self.method}")
        print(f"Samples: {self.num_samples}")
        print(f"Candidates: {self.num_candidates}")
        self.binary_metrics.print()
        self.cost_metrics.print()
        print(f"{'='*50}")


def compute_binary_metrics(
    predictions: List[int],
    labels: List[int]
) -> BinaryMetrics:
    """
    Compute binary classification metrics.

    Args:
        predictions: List of predicted labels (0 or 1)
        labels: List of ground truth labels (0 or 1)

    Returns:
        BinaryMetrics object
    """
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases where only one class is predicted
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (1, 1):
            if predictions[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]

    return BinaryMetrics(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn)
    )


def parse_prediction(response: str) -> int:
    """
    Parse LLM response to binary label.

    Args:
        response: LLM response text

    Returns:
        1 if vulnerable detected, 0 otherwise
    """
    response_lower = response.strip().lower()

    # Check for vulnerability indicators
    vulnerable_keywords = ["vulnerable", "yes", "true", "1"]
    safe_keywords = ["safe", "no", "false", "0"]

    for keyword in vulnerable_keywords:
        if keyword in response_lower:
            return 1

    for keyword in safe_keywords:
        if keyword in response_lower:
            return 0

    # Default: if uncertain, treat as not vulnerable
    return 0


def majority_vote(predictions: List[int]) -> int:
    """
    Return majority vote from predictions.

    Args:
        predictions: List of binary predictions

    Returns:
        Majority label (0 or 1)
    """
    if not predictions:
        return 0
    return 1 if sum(predictions) > len(predictions) / 2 else 0


class MultiClassMetrics:
    """Metrics for multi-class classification (e.g., CWE types)."""

    def __init__(self, predictions: List[str], labels: List[str]):
        """
        Initialize with predictions and labels.

        Args:
            predictions: List of predicted class labels
            labels: List of ground truth class labels
        """
        self.predictions = predictions
        self.labels = labels

    def top1_accuracy(self) -> float:
        """Compute top-1 accuracy."""
        return accuracy_score(self.labels, self.predictions)

    def top_k_accuracy(self, k: int = 3) -> float:
        """
        Compute top-k accuracy (if predictions contain top-k suggestions).

        Args:
            k: Top-k value

        Returns:
            Top-k accuracy
        """
        # For now, just return top-1
        # This can be extended if we return top-k predictions
        return self.top1_accuracy()

    def print_report(self) -> None:
        """Print classification report."""
        print("\n=== Multi-Class Classification Report ===")
        print(classification_report(self.labels, self.predictions, zero_division=0))
