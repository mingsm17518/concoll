#!/usr/bin/env python3
"""
ConColl: Confidence- and Collaboration-based Decision Making
Sequential multi-stage approach for code vulnerability detection.

Based on EMNLP 2025 paper: "A Sequential Multi-Stage Approach for
Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making"
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Union
import json

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data.local_loader import LocalPrimeVulLoader
from data.primevul_loader import create_test_samples
from llm.unified_client import UnifiedClient, create_client_from_config
from models.concoll_stage1 import DirectPredictor
from models.concoll_stage2 import RAGPredictor, create_rag_examples_from_samples
from models.concoll_stage3 import MultiAgentCollaboration
from evaluation.metrics import compute_binary_metrics, CostMetrics, ExperimentResult
from llm.gpt4o_client import TokenUsage

Client = UnifiedClient  # Unified interface for both providers


class ConCollFramework:
    """
    ConColl Sequential Framework.

    Three-stage sequential approach:
    1. Direct Prediction with confidence scoring
    2. RAG with external examples
    3. Multi-Agent Collaboration
    """

    def __init__(
        self,
        client: Client,
        confidence_threshold: float = 0.3,
        stage2_threshold: float = 0.2,
        rag_examples: int = 50,
        verbose: bool = True,
        force_stages: bool = False,
        use_stage3: bool = True,  # Enable Stage 3 by default
        simulate_mode: bool = False,
        simulate_ratios: dict = None
    ):
        """
        Initialize ConColl framework.

        Args:
            client: LLM client
            confidence_threshold: Threshold for Stage 1 acceptance (th1)
            stage2_threshold: Threshold for Stage 2 acceptance (th2)
            rag_examples: Number of examples for RAG retrieval
            verbose: Whether to print progress
            force_stages: If True, force all samples through all stages
            simulate_mode: If True, simulate GPT confidence distribution
            simulate_ratios: Ratios for stage distribution
        """
        self.client = client
        self.confidence_threshold = confidence_threshold
        self.stage2_threshold = stage2_threshold
        self.rag_examples = rag_examples
        self.verbose = verbose
        self.force_stages = force_stages
        self.use_stage3 = use_stage3  # Whether to use Stage 3
        self.simulate_mode = simulate_mode
        self.simulate_ratios = simulate_ratios or {"stage1": 0.7, "stage2": 0.25, "stage3": 0.05}

        # Stages will be initialized after data loading
        self.stage1 = None
        self.stage2 = None
        self.stage3 = None

    def setup_stages(self, training_samples):
        """
        Initialize all three stages.

        Args:
            training_samples: Samples from training set for RAG examples
        """
        # Stage 1: Direct Prediction
        self.stage1 = DirectPredictor(
            client=self.client,
            confidence_threshold=self.confidence_threshold,
            verbose=self.verbose,
            simulate_mode=self.simulate_mode,
            simulate_ratios=self.simulate_ratios
        )

        # Stage 2: RAG (needs examples)
        from models.concoll_stage2 import RAGRetriever
        examples = create_rag_examples_from_samples(training_samples[:self.rag_examples])
        retriever = RAGRetriever(examples, top_k=3)
        self.stage2 = RAGPredictor(
            client=self.client,
            retriever=retriever,
            confidence_threshold=self.stage2_threshold,  # th2 (independent from th1)
            verbose=self.verbose
        )

        # Stage 3: Multi-Agent Collaboration
        self.stage3 = MultiAgentCollaboration(
            client=self.client,
            agents=["security_analyst", "penetration_tester", "software_security_engineer"],
            voting_strategy="majority",
            verbose=self.verbose
        )

    def predict_batch(
        self,
        codes: List[str],
        labels: List[int]
    ) -> tuple:
        """
        Run predictions through the sequential framework.

        Args:
            codes: List of source code snippets
            labels: True labels for RAG retrieval

        Returns:
            Tuple of (predictions, stage_stats)
        """
        total_samples = len(codes)
        predictions = [None] * total_samples
        total_usage = TokenUsage()

        # Stage statistics
        stage_stats = {
            "stage1_accepted": 0,
            "stage2_used": 0,
            "stage3_used": 0,
            "stage1_cost": 0,
            "stage2_cost": 0,
            "stage3_cost": 0
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print("ConColl Sequential Framework")
            print(f"{'='*60}")
            print(f"Total samples: {total_samples}")
            print(f"Stage 1 threshold (th1): {self.confidence_threshold}")
            print(f"Stage 2 threshold (th2): {self.stage2_threshold}")
            print(f"RAG examples: {self.rag_examples}")
            print(f"\nStarting sequential prediction...")
            print(f"{'='*60}\n")

        # Stage 1: Direct Prediction for all samples
        if self.verbose:
            print("Stage 1: Direct Prediction with Confidence Scoring")

        stage1_preds, stage1_accepted, stage1_usage = self.stage1.predict_batch(codes)

        # Apply Stage 1 predictions where confident (or if not forcing stages)
        for i in range(total_samples):
            if stage1_accepted[i] and not self.force_stages:
                predictions[i] = stage1_preds[i]
                stage_stats["stage1_accepted"] += 1

        total_usage += type('Usage', (), {
            'prompt_tokens': stage1_usage.prompt_tokens,
            'completion_tokens': stage1_usage.completion_tokens,
            'total_tokens': stage1_usage.total_tokens,
            'api_calls': stage1_usage.api_calls
        })()
        stage_stats["stage1_cost"] = stage1_usage.total_tokens

        # Save intermediate state for error recovery
        self._last_predictions = predictions.copy()
        self._last_stage_stats = stage_stats.copy()

        # Find samples that need Stage 2/3
        if self.force_stages:
            # When forcing, all samples go through all stages
            deferred_indices = list(range(total_samples))
            stage_stats["stage1_accepted"] = 0
        else:
            deferred_indices = [i for i in range(total_samples) if predictions[i] is None]

        if self.verbose:
            print(f"\nStage 1 Summary:")
            print(f"  Accepted: {stage_stats['stage1_accepted']}/{total_samples}")
            print(f"  Deferred: {len(deferred_indices)}/{total_samples}")
            if self.force_stages:
                print(f"  (Forcing all samples through all stages)")

        # Stage 2: RAG for deferred samples
        if deferred_indices:
            if self.verbose:
                print(f"\nStage 2: RAG with External Examples")

            stage2_preds, stage2_usage, stage2_examples, stage2_accepted = self.stage2.predict_batch(
                codes, labels, deferred_indices
            )

            # Find samples that need Stage 3 (not accepted in Stage 2)
            stage3_indices = []
            for i, idx in enumerate(deferred_indices):
                stage_stats["stage2_used"] += 1
                if self.use_stage3 and self.stage3 is not None:
                    # Check if Stage 2 accepted this prediction
                    if not stage2_accepted[i]:
                        # Need to go to Stage 3
                        stage3_indices.append(idx)
                    else:
                        # Stage 2 accepted - use prediction
                        predictions[idx] = stage2_preds[i]
                else:
                    # No Stage 3 - use Stage 2 as final
                    predictions[idx] = stage2_preds[i]

            # Print Stage 2 acceptance stats
            accepted_count = sum(stage2_accepted)
            if self.verbose:
                print(f"Stage 2: Accepted {accepted_count}/{len(deferred_indices)}, Deferred {len(deferred_indices) - accepted_count} to Stage 3")

            total_usage += type('Usage', (), {
                'prompt_tokens': stage2_usage.prompt_tokens,
                'completion_tokens': stage2_usage.completion_tokens,
                'total_tokens': stage2_usage.total_tokens,
                'api_calls': stage2_usage.api_calls
            })()
            stage_stats["stage2_cost"] = stage2_usage.total_tokens

            # Save intermediate state after Stage 2
            self._last_predictions = predictions.copy()
            self._last_stage_stats = stage_stats.copy()

        # Stage 3: Multi-Agent Collaboration
        # Run Stage 3 only for samples not accepted in Stage 2
        if self.use_stage3 and self.stage3 is not None and stage3_indices:
            if self.verbose:
                print(f"\nStage 3: Multi-Agent Collaboration")

            stage3_preds, stage3_votes, stage3_usage = self.stage3.predict_batch(
                codes, stage3_indices, stage2_examples
            )
            for idx in stage3_indices:
                predictions[idx] = stage3_preds[idx]
                stage_stats["stage3_used"] += 1

            # Handle usage
            if hasattr(stage3_usage, 'prompt_tokens'):
                total_usage.prompt_tokens += stage3_usage.prompt_tokens
                total_usage.completion_tokens += stage3_usage.completion_tokens
                total_usage.total_tokens += stage3_usage.total_tokens
                total_usage.api_calls += stage3_usage.api_calls
            else:
                total_usage.api_calls += len(stage3_indices) * len(self.stage3.agents)

            stage_stats["stage3_cost"] = getattr(stage3_usage, 'total_tokens', 0)
            if self.verbose:
                print(f"Stage 3 processed: {stage_stats['stage3_used']} samples")

        if self.verbose:
            print(f"\n{'='*60}")
            print("ConColl Framework Summary")
            print(f"{'='*60}")
            print(f"Stage 1 (Direct): {stage_stats['stage1_accepted']} samples")
            print(f"Stage 2 (RAG): {stage_stats['stage2_used']} samples")
            print(f"Stage 3 (Multi-Agent): {stage_stats['stage3_used']} samples")
            print(f"\nTotal Tokens: {total_usage.total_tokens:,}")
            print(f"API Calls: {total_usage.api_calls}")
            print(f"{'='*60}\n")

        return predictions, stage_stats


def save_intermediate_result(config: Config, predictions: list, labels: list,
                             stage_stats: dict, completed_stage: int):
    """Save intermediate results when a stage completes or error occurs."""
    import json

    # Compute metrics for completed predictions
    valid_preds = [p for p in predictions if p is not None]
    valid_labels = [labels[i] for i, p in enumerate(predictions) if p is not None]

    if len(valid_preds) == 0:
        return

    # Compute binary metrics
    binary_metrics = compute_binary_metrics(valid_preds, valid_labels)

    result = {
        "method": "concoll",
        "num_samples": len(predictions),
        "completed_stage": completed_stage,
        "valid_samples": len(valid_preds),
        "stage_stats": stage_stats,
        "binary_metrics": {
            "accuracy": float(binary_metrics.accuracy),
            "precision": float(binary_metrics.precision),
            "recall": float(binary_metrics.recall),
            "f1": float(binary_metrics.f1),
            "tp": int(binary_metrics.tp),
            "fp": int(binary_metrics.fp),
            "tn": int(binary_metrics.tn),
            "fn": int(binary_metrics.fn)
        },
        "status": "interrupted" if completed_stage < 3 else "completed"
    }

    os.makedirs(config.results_dir, exist_ok=True)
    result_path = os.path.join(config.results_dir, "concoll_results.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[Auto-saved] Intermediate result (Stage {completed_stage}): {result_path}")


def run_experiment(config: Config, use_test_data: bool = False,
                   confidence_threshold: float = 0.3,
                   stage2_threshold: float = 0.2,
                   force_stages: bool = False,
                   simulate_mode: bool = False,
                   simulate_ratios: dict = None):
    """Run ConColl experiment."""
    print(f"\n{'='*60}")
    print(f"ConColl Framework for Vulnerability Detection")
    print(f"{'='*60}")

    # Load data
    if use_test_data:
        print("Using synthetic test data...")
        test_samples = create_test_samples()
        train_samples = test_samples  # Use same for simplicity
    else:
        print("Loading PrimeVul dataset from local file...")

        # Load test data
        test_loader = LocalPrimeVulLoader(
            max_samples=config.max_samples,
            random_seed=config.random_seed
        )
        test_samples = test_loader.load()

        # Load training data for RAG examples
        train_loader = LocalPrimeVulLoader(
            max_samples=200,  # More examples for RAG
            random_seed=43
        )
        try:
            train_samples = train_loader.load()
        except:
            print("Warning: Could not load separate training data, using test data for RAG examples")
            train_samples = test_samples

    # Prepare data - use BOTH vulnerable and fixed code
    codes = []
    labels = []
    for s in test_samples:
        codes.append(s.vulnerable_code)
        labels.append(1)
        codes.append(s.fixed_code)
        labels.append(0)

    print(f"Loaded {len(codes)} test samples ({len(set(labels))} unique labels)")
    print(f"Loaded {len(train_samples)} training samples for RAG")

    # Validate config
    config.validate()

    # Create client
    print(f"\nUsing API provider: {config.api_provider}")
    print(f"Model: {config.model}")
    print(f"Provider: {config.api_provider}")
    print(f"Supports logprobs: {'Yes (OpenAI/GPT)' if config.api_provider == 'openai' else 'No (GLM/MiniMax)'}")

    # Create unified client
    # When ready to switch to GPT, simply change .env:
    #   API_PROVIDER=openai
    #   OPENAI_API_KEY=your_gpt_key
    #   OPENAI_BASE_URL=https://api.openai.com/v1 (or compatible)
    #   MODEL=gpt-4
    client = create_client_from_config(config)

    # Create ConColl framework
    if simulate_mode:
        print(f"\nSimulate Mode: ENABLED")
        print(f"  Ratios: {simulate_ratios}")
        print(f"  This simulates GPT-4 confidence distribution for GLM testing\n")

    framework = ConCollFramework(
        client=client,
        confidence_threshold=confidence_threshold,
        stage2_threshold=stage2_threshold,
        rag_examples=min(50, len(train_samples)),
        verbose=True,
        force_stages=force_stages,
        use_stage3=True,  # Enable Stage 3 by default for full pipeline
        simulate_mode=simulate_mode,
        simulate_ratios=simulate_ratios
    )

    # Setup stages with training examples
    framework.setup_stages(train_samples)

    # Initialize for error recovery
    predictions = None
    stage_stats = None

    # Run predictions with error handling
    try:
        predictions, stage_stats = framework.predict_batch(codes, labels)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("[INFO] Saving intermediate results...")

        # Get current state from framework
        # predictions and stage_stats may be partially filled
        # Use framework's internal state if available
        if hasattr(framework, '_last_predictions'):
            predictions = framework._last_predictions
        if hasattr(framework, '_last_stage_stats'):
            stage_stats = framework._last_stage_stats

        if predictions is None:
            print("[WARNING] No predictions available to save")
        else:
            # Try to compute metrics with available predictions
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            valid_preds = [predictions[i] for i in valid_indices]
            valid_labels = [labels[i] for i in valid_indices]

            if len(valid_preds) > 0:
                save_intermediate_result(config, predictions, labels,
                                       stage_stats if stage_stats else
                                       {"stage1_accepted": 0, "stage2_used": 0, "stage3_used": 0,
                                        "stage1_cost": 0, "stage2_cost": 0, "stage3_cost": 0},
                                       3)  # Assume Stage 3 was running
                binary_metrics = compute_binary_metrics(valid_preds, valid_labels)
                print(f"[INFO] Saved results for {len(valid_preds)}/{len(predictions)} samples")
                print(f"[INFO] Accuracy so far: {binary_metrics.accuracy:.4f}")
            else:
                print("[WARNING] No valid predictions to save")

        # Re-raise to halt execution
        raise

    # Compute metrics
    binary_metrics = compute_binary_metrics(predictions, labels)

    # Create cost metrics
    cost_metrics = type('CostMetrics', (), {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'total_api_calls': 0,
        'estimated_cost_usd': 0.0,
        'to_dict': lambda self: {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'total_api_calls': self.total_api_calls,
            'estimated_cost_usd': self.estimated_cost_usd
        }
    })()

    # Note: Token usage tracking needs to be accumulated across stages
    # This is a simplified version

    # Create result
    result = {
        "method": "concoll",
        "num_samples": len(codes),
        "stage_stats": stage_stats,
        "binary_metrics": {
            "accuracy": float(binary_metrics.accuracy),
            "precision": float(binary_metrics.precision),
            "recall": float(binary_metrics.recall),
            "f1": float(binary_metrics.f1),
            "tp": int(binary_metrics.tp),
            "fp": int(binary_metrics.fp),
            "tn": int(binary_metrics.tn),
            "fn": int(binary_metrics.fn)
        }
    }

    # Save result
    os.makedirs(config.results_dir, exist_ok=True)
    result_path = os.path.join(config.results_dir, "concoll_results.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {binary_metrics.accuracy:.4f}")
    print(f"Precision: {binary_metrics.precision:.4f}")
    print(f"Recall:    {binary_metrics.recall:.4f}")
    print(f"F1 Score:  {binary_metrics.f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={binary_metrics.tp} | FN={binary_metrics.fn}")
    print(f"  FP={binary_metrics.fp} | TN={binary_metrics.tn}")
    print(f"\nStage Statistics:")
    print(f"  Stage 1 (Direct): {stage_stats['stage1_accepted']} samples")
    print(f"  Stage 2 (RAG): {stage_stats['stage2_used']} samples")
    print(f"\nResults saved to: {result_path}")
    print(f"{'='*60}\n")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ConColl: Sequential Multi-Stage Vulnerability Detection"
    )
    parser.add_argument("--test-data", action="store_true", help="Use synthetic test data")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Stage 1 confidence threshold (default: 0.3)")
    parser.add_argument("--stage2-threshold", type=float, default=0.2,
                        help="Stage 2 confidence threshold (default: 0.2)")
    parser.add_argument("--force-stages", action="store_true",
                        help="Force all samples through all three stages")
    parser.add_argument("--simulate", action="store_true",
                        help="Enable simulate mode (use fixed ratios instead of real logprobs)")
    parser.add_argument("--stage1-ratio", type=float, default=0.7,
                        help="Ratio of samples for Stage 1 in simulate mode (default: 0.7)")
    parser.add_argument("--stage2-ratio", type=float, default=0.25,
                        help="Ratio of samples for Stage 2 in simulate mode (default: 0.25)")
    parser.add_argument("--stage3-ratio", type=float, default=0.05,
                        help="Ratio of samples for Stage 3 in simulate mode (default: 0.05)")
    args = parser.parse_args()

    # Create config first to get default values
    default_config = Config()

    config = Config.from_args(
        max_samples=args.max_samples,
        results_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold if args.confidence_threshold != 0.3 else default_config.confidence_threshold
    )

    # Build simulate ratios if simulate mode is enabled
    simulate_ratios = None
    if args.simulate:
        total = args.stage1_ratio + args.stage2_ratio + args.stage3_ratio
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Ratios sum to {total}, normalizing to 1.0")
        simulate_ratios = {
            "stage1": args.stage1_ratio / total,
            "stage2": args.stage2_ratio / total,
            "stage3": args.stage3_ratio / total
        }

    run_experiment(
        config,
        args.test_data,
        config.confidence_threshold,
        args.stage2_threshold,
        args.force_stages,
        args.simulate,
        simulate_ratios
    )


if __name__ == "__main__":
    main()
