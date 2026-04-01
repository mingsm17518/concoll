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
            "system": """You are a Security Analyst and your task is to assess whether the provided code contains any security vulnerabilities.""",
            "prompt": """You are a security analyst and your task is to assess whether the provided code contains any security vulnerabilities. \n\n Here are some examples: \n\n {examples} \n\n Is the following code vulnerable? Respond with only 'Yes' or 'No'. \n\n {code}"""
        },
        "penetration_tester": {
            "name": "Penetration Tester",
            "system": """You are a Penetration Tester and your task is to assess whether the provided code contains any security vulnerabilities.""",
            "prompt": """You are a penetration tester and your task is to assess whether the provided code contains any security vulnerabilities. \n\n Here are some examples: \n\n {examples} \n\n Is the following code vulnerable? Respond with only 'Yes' or 'No'. \n\n {code}"""
        },
        "software_security_engineer": {
            "name": "Software Security Engineer",
            "system": """You are a Software Security Engineer and your task is to assess whether the provided code contains any security vulnerabilities.""",
            "prompt": """You are a software security engineer and your task is to assess whether the provided code contains any security vulnerabilities. \n\n Here are some examples: \n\n {examples} \n\n Is the following code vulnerable? Respond with only 'Yes' or 'No'. \n\n {code}"""
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
        self.agents = agents or ["security_analyst", "penetration_tester", "software_security_engineer"]
        self.voting_strategy = voting_strategy
        self.verbose = verbose
        self.name = "concoll_stage3"

    def predict(self, code: str, examples: List["RAGExample"] = None) -> Tuple[int, Dict, object]:
        """
        Make prediction through multi-agent collaboration.

        Args:
            code: Source code to analyze
            examples: Retrieved examples from Stage 2 (optional)

        Returns:
            Tuple of (prediction, agent_votes, total_usage)
        """
        agent_votes = {}
        total_usage = TokenUsage()

        # Format examples into text
        if examples:
            examples_text = self._format_examples(examples)
        else:
            examples_text = "No examples available."

        # Get prediction from each agent
        for agent_key in self.agents:
            agent = self.AGENTS[agent_key]

            messages = [
                {"role": "system", "content": agent["system"]},
                {"role": "user", "content": agent["prompt"].format(code=code, examples=examples_text)}
            ]

            response, usage = self.client.chat_completion(messages)
            # Handle both simple usage and nested (usage, logprobs_info) tuple
            if isinstance(usage, tuple):
                actual_usage = usage[0] if hasattr(usage[0], 'prompt_tokens') else usage
            else:
                actual_usage = usage
            if actual_usage is not None:
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

    def _format_examples(self, examples: List) -> str:
        """Format RAG examples into prompt text."""
        if not examples:
            return "No examples available."

        formatted = []
        for i, ex in enumerate(examples, 1):
            label_str = "VULNERABLE" if ex.label == 1 else "SAFE"
            cwe_str = ex.cwe if hasattr(ex, 'cwe') and ex.cwe else "N/A"
            formatted.append(f"""Example {i} ({label_str}, {cwe_str}):
{ex.code}""")

        return "\n\n".join(formatted)

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
        indices: List[int] = None,
        examples_list: List[List] = None
    ) -> Tuple[List[int], Dict[str, List[int]], object]:
        """
        Make predictions for a batch of codes.

        Args:
            codes: List of source code snippets
            indices: Indices of codes to process (None = all)
            examples_list: List of examples from Stage 2 for each code

        Returns:
            Tuple of (predictions, all_agent_votes, total_usage)
        """
        if indices is None:
            indices = range(len(codes))

        # Return results corresponding to indices, not full codes list
        predictions = []
        all_agent_votes = {agent: [] for agent in self.agents}
        total_usage = TokenUsage()

        for i, idx in enumerate(indices):
            if self.verbose and (i + 1) % 2 == 0:
                print(f"  Stage 3: Processed {i + 1}/{len(indices)} samples...")

            if self.verbose:
                print(f"    Sample {idx + 1}:")

            code = codes[idx]
            # Get examples for this specific code
            examples = examples_list[idx] if examples_list and idx < len(examples_list) else None
            vote, agent_votes, usage = self.predict(code, examples)

            predictions.append(vote)
            # Handle both simple usage and nested (usage, logprobs_info) tuple
            if isinstance(usage, tuple):
                actual_usage = usage[0] if hasattr(usage[0], 'prompt_tokens') else usage
            else:
                actual_usage = usage
            if actual_usage is not None:
                total_usage += actual_usage

            for agent, agent_vote in agent_votes.items():
                all_agent_votes[agent].append(agent_vote)

        if self.verbose:
            print(f"  Stage 3: Completed {len(indices)} samples")

        return predictions, all_agent_votes, total_usage

    def get_name(self) -> str:
        """Get method name."""
        return self.name
