from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Any, Dict, List

@dataclass
class StepRecord:
    env_name: str
    env_type: str
    episode_id: str
    step_id: int
    task: str
    observation: str
    action: str
    action_type: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    short_justification: Optional[str] = None
    confidence: Optional[float] = None
    valid_action: Optional[bool] = None
    reward: Optional[float] = None
    score: Optional[float] = None
    done: bool = False
    failure_type: Optional[str] = None
    retry_attempt: Optional[int] = None
    recovery_action: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    episode_summary: Optional[str] = None
    next_action: Optional[str] = None
    action_class: Optional[str] = None
    success_likelihood: Optional[float] = None
    constraint_risk: Optional[float] = None
    recovery_needed: Optional[bool] = None
    recovery_action_class: Optional[str] = None
    outcome: Optional[str] = None
    raw_env: Optional[Dict[str, Any]] = None
    raw_model: Optional[Dict[str, Any]] = None
    thought: Optional[str] = None
    diary: Optional[str] = None
    trace_contract_version: Optional[str] = None
    trace_mode: Optional[str] = None
    reasoning_trace: List[Dict[str, str]] = field(default_factory=list)
    reasoning_summary: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
