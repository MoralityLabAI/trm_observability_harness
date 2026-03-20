from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from harness.trace_contract import default_trace_profile

class BaseEnv(ABC):
    name: str

    @abstractmethod
    def reset(self) -> Tuple[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        ...

    def trace_profile(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return default_trace_profile(
            env_name=getattr(self, "name", self.__class__.__name__),
            env_type=getattr(self, "env_type", self.__class__.__name__),
            meta=meta,
        )
