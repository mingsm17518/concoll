"""Local PrimeVul dataset loader for pre-downloaded JSONL files."""

import json
import os
from typing import List, Optional
from dataclasses import dataclass

from .primevul_loader import VulnerabilitySample


class LocalPrimeVulLoader:
    """Loader for local PrimeVul JSONL files."""

    def __init__(
        self,
        data_path: str = "PrimeVul_v0.1/primevul_test_paired.jsonl",
        max_samples: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize the local PrimeVul loader.

        Args:
            data_path: Path to the JSONL file
            max_samples: Maximum number of pairs to load (None = all)
            random_seed: Random seed for sampling
        """
        self.data_path = data_path
        self.max_samples = max_samples
        self.random_seed = random_seed
        self._samples: List[VulnerabilitySample] = []

    def load(self) -> List[VulnerabilitySample]:
        """
        Load the PrimeVul dataset from local JSONL file.

        The paired format has:
        - target=1: vulnerable code (func_before)
        - target=0: fixed code (func_after)

        Returns:
            List of VulnerabilitySample objects
        """
        print(f"Loading PrimeVul from: {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        samples = []
        vulnerable_code = None
        vulnerable_meta = None

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                data = json.loads(line)

                if data['target'] == 1:
                    # Vulnerable code
                    vulnerable_code = data['func']
                    vulnerable_meta = {
                        'idx': data['idx'],
                        'commit_id': data['commit_id'],
                        'project': data['project'],
                        'cwe': data.get('cwe', ['Unknown'])[0] if data.get('cwe') else 'Unknown',
                        'cve': data.get('cve', ''),
                    }
                elif data['target'] == 0 and vulnerable_code is not None:
                    # Fixed code - pair with previous vulnerable code
                    sample = VulnerabilitySample(
                        idx=vulnerable_meta['idx'],
                        func_before=vulnerable_code,
                        func_after=data['func'],
                        cwe=vulnerable_meta['cwe'],
                        label=1,  # func_before is vulnerable
                        commit_id=vulnerable_meta['commit_id'],
                        repo=vulnerable_meta['project']
                    )
                    samples.append(sample)
                    vulnerable_code = None
                    vulnerable_meta = None

        print(f"Loaded {len(samples)} paired samples")

        # Apply max_samples limit if specified
        if self.max_samples and len(samples) > self.max_samples:
            import random
            random.seed(self.random_seed)
            samples = random.sample(samples, self.max_samples)
            print(f"Sampled {self.max_samples} examples (random seed={self.random_seed})")

        self._samples = samples
        return samples

    def get_samples(self) -> List[VulnerabilitySample]:
        """Get loaded samples."""
        if not self._samples:
            return self.load()
        return self._samples
