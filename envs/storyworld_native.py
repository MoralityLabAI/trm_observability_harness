from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from .base import BaseEnv


def _add_repo_path(repo_path: str) -> None:
    repo = Path(repo_path).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    venv_site_packages = repo / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
        sys.path.insert(0, str(venv_site_packages))


class NativeStoryWorldEnv(BaseEnv):
    """Native wrapper around GPTStoryworld DiplomacyStoryworldEnv."""

    env_type = "storyworld_native"

    def __init__(self, name: str, repo_path: str, storyworld_path: str, seed: int | None = None, log_path: str | None = None):
        self.name = name
        self.repo_path = repo_path
        self.storyworld_path = storyworld_path
        self.seed = seed
        self.log_path = log_path
        self._env = None
        self.agent_ids = []

    def _load(self):
        _add_repo_path(self.repo_path)
        from storyworld.env.diplomacy_env import DiplomacyStoryworldEnv
        from storyworld.env.storyworld_env import load_storyworld

        storyworld = load_storyworld(self.storyworld_path)
        self._env = DiplomacyStoryworldEnv(storyworld=storyworld, seed=self.seed, log_path=self.log_path)

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        if self._env is None:
            self._load()
        state = self._env.reset(seed=self.seed)
        self.agent_ids = list((state.get("beliefs") or {}).keys())
        meta = {
            "task": "Choose safe multi-agent actions and messages for the storyworld.",
            "state": state,
            "repo_path": self.repo_path,
            "storyworld_path": self.storyworld_path,
            "source": "storyworld_native",
        }
        return str(state), meta

    def _parse_decision(self, decision):
        if hasattr(decision, "raw_text") and decision.raw_text:
            import json

            try:
                payload = json.loads(decision.raw_text)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        if isinstance(decision, str):
            import json

            try:
                payload = json.loads(decision)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return {"action": decision}
        return {"action": getattr(decision, "action", "wait")}

    def step(self, decision):
        payload = self._parse_decision(decision)
        actions = payload.get("actions")
        messages = payload.get("messages")
        if actions is None:
            action = payload.get("action", "wait")
            agent_id = self.agent_ids[0] if self.agent_ids else "agent_0"
            actions = {agent_id: {"type": action, "target": payload.get("target")}}
        state, event, done = self._env.step(actions, messages)
        info = {
            "valid_action": True,
            "failure_type": payload.get("failure_type"),
            "episode_summary": event.get("outcome"),
            "reward": event.get("metrics", {}).get("coalition_mean_stability", 0.0),
            "score": event.get("metrics", {}).get("coalition_mean_stability", 0.0),
            "raw_event": event,
            "state": state,
        }
        return str(state), float(info["reward"]), bool(done), info
