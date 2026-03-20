from __future__ import annotations
from typing import Tuple, Dict, Any
from .base import BaseEnv

class DummyPrimeHubEnv(BaseEnv):
    def __init__(self, name: str):
        self.name = name
        self.turn = 0

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        self.turn = 0
        return f'{self.name}: initial observation', {'task': f'Solve task in {self.name}'}

    def step(self, decision) -> Tuple[str, float, bool, Dict[str, Any]]:
        action = getattr(decision, 'action', decision)
        self.turn += 1
        if action == 'retry_with_safe_defaults':
            obs = f'{self.name}: recovered state'
            reward = 0.7
        else:
            obs = f'{self.name}: progressed state {self.turn}'
            reward = 0.4
        done = self.turn >= 3
        info: Dict[str, Any] = {'valid_action': True}
        if done:
            info['episode_summary'] = f'Completed {self.name} in {self.turn} steps.'
        return obs, reward, done, info
