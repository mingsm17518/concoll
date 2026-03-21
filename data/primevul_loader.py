"""PrimeVul dataset loader for vulnerability detection."""

import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import random

try:
    from datasets import load_dataset, Dataset
except ImportError:
    raise ImportError("Please install datasets: pip install datasets")


@dataclass
class VulnerabilitySample:
    """A single vulnerability sample from PrimeVul."""
    idx: int
    func_before: str  # Vulnerable code
    func_after: str   # Fixed code
    cwe: str          # CWE identifier
    label: int        # 1 = vulnerable, 0 = safe (after fix)
    commit_id: str
    repo: str

    @property
    def vulnerable_code(self) -> str:
        """Get the vulnerable code."""
        return self.func_before

    @property
    def fixed_code(self) -> str:
        """Get the fixed code."""
        return self.func_after


class PrimeVulLoader:
    """Loader for PrimeVul dataset."""

    # PrimeVul on HuggingFace
    # Reference: https://huggingface.co/datasets/ASSERT-KTH/PrimeVul

    def __init__(
        self,
        dataset_name: str = "ASSERT-KTH/PrimeVul",
        split: str = "test_paired",
        max_samples: Optional[int] = None,
        random_seed: int = 42,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the PrimeVul loader.

        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Dataset split to load (train_paired, test_paired, valid_paired)
            max_samples: Maximum number of samples to load (None = all)
            random_seed: Random seed for sampling
            cache_dir: Directory to cache downloaded datasets
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.cache_dir = cache_dir
        self._dataset: Optional[Dataset] = None
        self._samples: List[VulnerabilitySample] = []

    def load(self) -> List[VulnerabilitySample]:
        """
        Load the PrimeVul dataset.

        Returns:
            List of VulnerabilitySample objects
        """
        print(f"Loading PrimeVul dataset: {self.dataset_name} ({self.split} split)...")

        # Set HuggingFace endpoint to direct (avoid mirror rate limits)
        import os
        os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
        print("Using HuggingFace direct endpoint...")

        try:
            # Try loading from HuggingFace
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir
            )
            print(f"Dataset loaded successfully. Size: {len(dataset)}")
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("\nTrying alternative dataset names...")
            # Try common alternatives
            alternatives = [
                "nguyenvulebinh/primevul",
                "primevul/primevul",
            ]
            dataset = None
            for alt_name in alternatives:
                try:
                    dataset = load_dataset(alt_name, split=self.split, cache_dir=self.cache_dir)
                    print(f"Successfully loaded from: {alt_name}")
                    break
                except Exception:
                    continue

            if dataset is None:
                raise RuntimeError(
                    f"Could not load PrimeVul dataset. "
                    f"Please ensure you have access to the dataset on HuggingFace.\n"
                    f"You may need to: huggingface-cli login"
                )

        self._dataset = dataset

        # Convert to VulnerabilitySample objects
        self._samples = self._convert_dataset(dataset)

        # Apply max_samples limit if specified
        if self.max_samples and len(self._samples) > self.max_samples:
            random.seed(self.random_seed)
            self._samples = random.sample(self._samples, self.max_samples)
            print(f"Sampled {self.max_samples} examples (random seed={self.random_seed})")

        print(f"Total samples loaded: {len(self._samples)}")
        return self._samples

    def _convert_dataset(self, dataset: Dataset) -> List[VulnerabilitySample]:
        """
        Convert HuggingFace dataset to VulnerabilitySample objects.

        PrimeVul dataset structure may vary, this handles common formats.
        """
        samples = []

        for idx, item in enumerate(dataset):
            # Handle different dataset formats
            # Common field names in PrimeVul: func_before, func_after, cwe, etc.
            func_before = item.get("func_before", item.get("vulnerable_func", ""))
            func_after = item.get("func_after", item.get("fixed_func", ""))
            cwe = item.get("cwe", item.get("CWE", "Unknown"))
            commit_id = item.get("commit_id", item.get("commit", ""))
            repo = item.get("repo", item.get("project", "Unknown"))

            # Skip if no code available
            if not func_before or not func_after:
                continue

            sample = VulnerabilitySample(
                idx=idx,
                func_before=func_before,
                func_after=func_after,
                cwe=str(cwe),
                label=1,  # func_before is vulnerable
                commit_id=str(commit_id),
                repo=str(repo)
            )
            samples.append(sample)

        return samples

    def get_samples(self) -> List[VulnerabilitySample]:
        """Get loaded samples."""
        if not self._samples:
            return self.load()
        return self._samples

    def create_pairs(self) -> Tuple[List[str], List[int]]:
        """
        Create code-label pairs for binary classification.

        Returns:
            Tuple of (codes, labels) where:
            - codes: list of code snippets
            - labels: 1 for vulnerable, 0 for safe
        """
        samples = self.get_samples()
        codes = []
        labels = []

        for sample in samples:
            # Add vulnerable code
            codes.append(sample.vulnerable_code)
            labels.append(1)

            # Add fixed code as safe
            codes.append(sample.fixed_code)
            labels.append(0)

        return codes, labels

    def get_cwe_distribution(self) -> Dict[str, int]:
        """Get distribution of CWE types in the dataset."""
        samples = self.get_samples()
        cwe_counts = {}
        for sample in samples:
            cwe = sample.cwe
            cwe_counts[cwe] = cwe_counts.get(cwe, 0) + 1
        return dict(sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True))


def create_test_samples() -> List[VulnerabilitySample]:
    """
    Create small test samples for testing the framework without dataset.

    Returns:
        List of synthetic VulnerabilitySample objects
    """
    return [
        VulnerabilitySample(
            idx=0,
            func_before="""void func(char *input) {
    char buffer[10];
    strcpy(buffer, input);
}""",
            func_after="""void func(char *input) {
    char buffer[10];
    strncpy(buffer, input, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\\0';
}""",
            cwe="CWE-120",
            label=1,
            commit_id="test1",
            repo="test"
        ),
        VulnerabilitySample(
            idx=1,
            func_before="""int divide(int a, int b) {
    return a / b;
}""",
            func_after="""int divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}""",
            cwe="CWE-369",
            label=1,
            commit_id="test2",
            repo="test"
        ),
        VulnerabilitySample(
            idx=2,
            func_before="""void process(char *user) {
    char cmd[100];
    sprintf(cmd, "echo %s", user);
    system(cmd);
}""",
            func_after="""void process(char *user) {
    // Validate input before using in command
    if (!is_valid_input(user)) return;
    char cmd[100];
    snprintf(cmd, sizeof(cmd), "echo %s", user);
    system(cmd);
}""",
            cwe="CWE-78",
            label=1,
            commit_id="test3",
            repo="test"
        ),
    ]
