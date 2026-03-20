from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

from .base import BaseEnv

try:
    import reasoning_gym as rg
except ImportError:
    rg = None


class ReasoningGymEnv(BaseEnv):
    """Wrapper for reasoning-gym procedural datasets."""

    env_type = "reasoning_gym"

    def __init__(
        self,
        name: str,
        gym_config: str | Dict[str, Any] | List[str | Dict[str, Any]],
        num_examples: int = 50,
        seed: int = 42,
    ):
        self.name = name
        self.gym_config = gym_config
        self.num_examples = num_examples
        self.seed = seed
        self._dataset = None
        self._current_idx = 0
        self._current_entry = None

    def _load(self):
        if rg is None:
            raise ImportError("reasoning-gym not installed.")

        if isinstance(self.gym_config, str):
            self._dataset = rg.create_dataset(self.gym_config, size=self.num_examples, seed=self.seed)
        elif isinstance(self.gym_config, dict):
            # Assumes it's a single dataset with config
            name = self.gym_config.get("name")
            config = self.gym_config.get("config", {})
            self._dataset = rg.create_dataset(name, size=self.num_examples, seed=self.seed, **config)
        elif isinstance(self.gym_config, list):
            # Composite dataset
            from reasoning_gym.composite import DatasetSpec

            specs = []
            for item in self.gym_config:
                if isinstance(item, str):
                    specs.append(DatasetSpec(name=item, weight=1.0, config={}))
                else:
                    specs.append(DatasetSpec(
                        name=item["name"],
                        weight=item.get("weight", 1.0),
                        config=item.get("config", {})
                    ))
            self._dataset = rg.create_dataset("composite", datasets=specs, size=self.num_examples, seed=self.seed)
        else:
            raise ValueError(f"Invalid gym_config type: {type(self.gym_config)}")

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        if self._dataset is None:
            self._load()

        self._current_entry = self._dataset[self._current_idx]
        prompt = self._current_entry["question"]
        
        meta = {
            "task": f"Reasoning Gym: {self._current_entry['metadata'].get('source_dataset', self.name)}",
            "entry": self._current_entry,
            "idx": self._current_idx,
            "source": "reasoning_gym",
        }
        return str(prompt), meta

    def step(self, decision) -> Tuple[str, float, bool, Dict[str, Any]]:
        # Single turn environment: answer is checked immediately
        # We assume the decision has the answer in a field or as a string
        response = getattr(decision, "action", decision)
        if hasattr(decision, "raw_text") and decision.raw_text:
            # Try to extract from think/answer tags if the harness doesn't do it
            response = decision.raw_text

        reward = self._dataset.score_answer(answer=str(response), entry=self._current_entry)
        
        # Advance index for next reset if this was a multi-episode run
        self._current_idx = (self._current_idx + 1) % self.num_examples

        observation = f"Correct Answer: {self._current_entry.get('answer')}"
        done = True
        info = {
            "valid_action": True,
            "reward": float(reward),
            "score": float(reward),
            "answer": self._current_entry.get("answer"),
            "response": response,
        }
        return observation, float(reward), done, info
