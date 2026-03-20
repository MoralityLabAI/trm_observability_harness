from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import BaseEnv


class NeedlePathfindingEnv(BaseEnv):
    """
    Wrapper for the Needle-in-a-Haystack pathfinding verifier.
    
    This environment presents the model with a storyworld and asks it to find a 
    specific target ending among many possibilities.
    """

    env_type = "needle_pathfinding"

    def __init__(
        self,
        name: str,
        repo_path: str,
        storyworld_path: str,
        target_ending: str,
        n_endings: int = 12,
        max_steps: int = 5,
    ):
        self.name = name
        self.repo_path = Path(repo_path).resolve()
        self.storyworld_path = Path(storyworld_path).resolve()
        self.target_ending = target_ending
        self.n_endings = n_endings
        self.max_steps = max_steps
        
        self.current_step = 0
        self.attempts: List[Dict[str, Any]] = []
        
        # We need the underlying storyworld env to actually play
        # For now, we'll mock the observation as the target task
        self.task_description = (
            f"You are navigating a complex storyworld: {self.storyworld_path.name}\n"
            f"Your objective is to reach the target ending: '{self.target_ending}'.\n"
            f"There are {self.n_endings} possible endings in this haystack.\n"
            "Analyze the state and choose the path that leads closest to the target."
        )

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        self.current_step = 0
        self.attempts = []
        meta = {
            "task": "Needle Pathfinding",
            "target": self.target_ending,
            "n_endings": self.n_endings,
        }
        return self.task_description, meta

    def step(self, decision: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        # In a real integration, we would play the storyworld here.
        # For the harness, we simulate one 'play' per step or a multi-turn navigation.
        # If the model is 'acting', we record its attempt.
        
        action_text = getattr(decision, "action", str(decision))
        
        # Mocking the evaluation logic from needle_pathfinding_env.py
        # In a real run, we'd run a rollout and see which ending it hits.
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Placeholder reward logic: if it's the last step, we check if it found it.
        # For now, we just return a small reward to keep the loop going.
        reward = 0.0
        if "reach" in action_text.lower() or "target" in action_text.lower():
            reward = 0.5 # Partial credit for intent
            
        info = {
            "valid_action": True,
            "reward": reward,
            "step": self.current_step,
        }
        
        if done:
            info["outcome"] = "Completed pathfinding session."
            
        observation = f"Current Step: {self.current_step}/{self.max_steps}. Action taken: {action_text}"
        
        return observation, float(reward), done, info

    def trace_profile(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        profile = super().trace_profile(meta)
        profile.update(
            {
                "family": "storyworld",
                "max_trace_steps": 5,
                "action_guidance": "Use action for the next pathfinding move or succinct route commitment.",
                "step_labels": ["state_read", "target_alignment", "risk_check", "next_move", "commit"],
            }
        )
        return profile
