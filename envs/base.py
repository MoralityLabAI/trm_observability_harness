from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseEnv(ABC):
    name: str

    @abstractmethod
    def reset(self) -> Tuple[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        ...
