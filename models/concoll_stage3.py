"""ConColl Stage 3: Multi-Agent Collaboration."""

from typing import List, Tuple, Dict
from dataclasses import dataclass

try:
    from ..llm.anthropic_client import AnthropicClient
    from ..llm.gpt4o_client import GPT4oClient
    from ..llm.gpt4o_client import TokenUsage
except ImportError:
    from llm.anthropic_client import AnthropicClient
    from llm.gpt4o_client import GPT4oClient
    from llm.gpt4o_client import TokenUsage


class MultiAgentCollaboration:
    """
    ConColl Stage 3: Multi-Agent Collaboration.

    Uses multiple agents with different roles to collaboratively
    analyze the code and reach a consensus.
    """

    # Agent roles and prompts
    AGENTS = {
        "security_analyst": {
            "name": "Security Analyst",
            "system": """You are a Security Analyst specializing in vulnerability detection.
Your role is to identify potential security issues in code from a defensive perspective.
Focus on: buffer overflows, injection attacks, authentication issues, authorization flaws.""",
            "prompt": """As a Security Analyst, analyze this code for vulnerabilities:
```c
{code}
```
Respond with 'Yes' if vulnerable, 'No' if safe."""
        },
        "code_reviewer": {
            "name": "Code Reviewer",
            "system": """You are a Code Reviewer focused on code quality and best practices.
Your role is to identify potential issues from a software engineering perspective.
Focus on: null pointer dereferences, memory leaks, race conditions, error handling.""",
            "prompt": """As a Code Reviewer, analyze this code for issues:
```c
{code}
```
Respond with 'Yes' if vulnerable, 'No' if safe."""
        },
        "attacker": {
            "name": "Attacker",
            "system": """You are simulating an Attacker mindset to find exploitable vulnerabilities.
Your role is to think like a malicious actor trying to exploit the code.
Focus on: input validation bypasses, privilege escalation, unauthorized access.""",
            "prompt": """As an Attacker looking for exploits, analyze this code:
```c
{code}
```
Respond with 'Yes' if exploitable, 'No' if safe."""
        }
    }

    def __init__(
        self,
        client,
        agents: List[str] = None,
        voting_strategy: str = "majority",
        verbose: bool = True
    ):
        """
        Initialize Multi-Agent Collaboration.

        Args:
            client: LLM client
            agents: List of agent names to use (default: all three)
            voting_strategy: How to combine agent votes ("majority", "unanimous", "any")
            verbose: Whether to print progress
        """
        self.client = client
        self.agents = agents or ["security_analyst", "code_reviewer", "attacker"]
        self.voting_strategy = voting_strategy
        self.verbose = verbose
        self.name = "concoll_stage3"

    def predict(self, code: str) -> Tuple[int, Dict, object]:
        """
        Make prediction through multi-agent collaboration.

        Args:
            code: Source code to analyze

        Returns:
            Tuple of (prediction, agent_votes, total_usage)
        """
        agent_votes = {}
        total_usage = TokenUsage()

        # Get prediction from each agent
        for agent_key in self.agents:
            agent = self.AGENTS[agent_key]

            messages = [
                {"role": "system", "content": agent["system"]},
                {"role": "user", "content": agent["prompt"].format(code=code)}
            ]

            response, usage = self.client.chat_completion(messages)
            # Handle both simple usage and nested (usage, logprobs_info) tuple
            if isinstance(usage, tuple):
                actual_usage = usage[0] if hasattr(usage[0], 'prompt_tokens') else usage
            else:
                actual_usage = usage
            total_usage += actual_usage

            # Parse response
            response_lower = response.strip().lower()
            vote = 1 if 'yes' in response_lower[:10] else 0
            agent_votes[agent_key] = vote

            if self.verbose:
                print(f"    {agent['name']}: {'Vulnerable' if vote == 1 else 'Safe'}")

        # Combine votes using voting strategy
        final_vote = self._combine_votes(agent_votes)

        return final_vote, agent_votes, total_usage

    def _combine_votes(self, votes: Dict[str, int]) -> int:
        """
        Combine agent votes into final prediction.

        Args:
            votes: Dictionary of agent_name -> vote (0 or 1)

        Returns:
            Final prediction (0 or 1)
        """
        vote_values = list(votes.values())

        if self.voting_strategy == "majority":
            return 1 if sum(vote_values) > len(vote_values) / 2 else 0
        elif self.voting_strategy == "unanimous":
            return 1 if all(v == 1 for v in vote_values) else 0
        elif self.voting_strategy == "any":
            return 1 if any(v == 1 for v in vote_values) else 0
        else:
            # Default to majority
            return 1 if sum(vote_values) > len(vote_values) / 2 else 0

    def predict_batch(
        self,
        codes: List[str],
        indices: List[int] = None
    ) -> Tuple[List[int], Dict[str, List[int]], object]:
        """
        Make predictions for a batch of codes.

        Args:
            codes: List of source code snippets
            indices: Indices of codes to process (None = all)

        Returns:
            Tuple of (predictions, all_agent_votes, total_usage)
        """
        if indices is None:
            indices = range(len(codes))

        predictions = [None] * len(codes)
        all_agent_votes = {agent: [None] * len(codes) for agent in self.agents}
        total_usage = TokenUsage()

        for i, idx in enumerate(indices):
            if self.verbose and (i + 1) % 2 == 0:
                print(f"  Stage 3: Processed {i + 1}/{len(indices)} samples...")

            if self.verbose:
                print(f"    Sample {idx + 1}:")

            code = codes[idx]
            vote, agent_votes, usage = self.predict(code)

            predictions[idx] = vote
            # Handle both simple usage and nested (usage, logprobs_info) tuple
            if isinstance(usage, tuple):
                actual_usage = usage[0] if hasattr(usage[0], 'prompt_tokens') else usage
            else:
                actual_usage = usage
            total_usage += actual_usage

            for agent, agent_vote in agent_votes.items():
                all_agent_votes[agent][idx] = agent_vote

        if self.verbose:
            print(f"  Stage 3: Completed {len(indices)} samples")

        return predictions, all_agent_votes, total_usage

    def get_name(self) -> str:
        """Get method name."""
        return self.name
