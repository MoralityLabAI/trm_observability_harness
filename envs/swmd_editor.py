from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from .base import BaseEnv


def _add_repo_path(repo_path: str) -> None:
    repo = Path(repo_path).resolve()
    env_dir = repo / "environments" / "sweepweave_prime_env"
    if env_dir.exists() and str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    venv_site_packages = repo / "environments" / "sweepweave_prime_env" / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
        sys.path.insert(0, str(venv_site_packages))


class SwmdEditorEnv(BaseEnv):
    """Native wrapper around the SWMD editor/edit-minify-rehydrate workflow."""

    env_type = "swmd_editor"

    def __init__(self, name: str, repo_path: str, load_kwargs: Dict[str, Any] | None = None):
        self.name = name
        self.repo_path = repo_path
        self.load_kwargs = load_kwargs or {}
        self._env = None

    def _load(self):
        _add_repo_path(self.repo_path)
        import sweepweave

        self._env = sweepweave.load_environment(**self.load_kwargs)

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        if self._env is None:
            self._load()
        state = self._env.reset()
        prompt = state.get("example", {}).get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
        meta = {
            "task": "Generate or edit valid SWMD editor JSON or markdown artifacts.",
            "state": state,
            "repo_path": self.repo_path,
            "source": "swmd_editor",
        }
        return str(prompt), meta

    def step(self, decision):
        response = getattr(decision, "action", decision)
        state, event, done = self._env.step(response, None)
        observation = str(state)
        info = {
            "valid_action": event.get("done", True),
            "failure_type": None,
            "episode_summary": event.get("outcome"),
            "reward": event.get("metrics", {}).get("coalition_mean_stability", 0.0),
            "score": event.get("metrics", {}).get("coalition_mean_stability", 0.0),
            "raw_event": event,
            "state": state,
        }
        return observation, float(info["reward"]), bool(done), info

    def trace_profile(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        profile = super().trace_profile(meta)
        profile.update(
            {
                "family": "swmd_editor",
                "action_guidance": "Use action for the exact edit command, patch, or JSON payload the SWMD editor workflow scores.",
                "step_labels": ["task_parse", "constraint_check", "candidate", "self_check"],
            }
        )
        return profile


SweepweavePrimeEnv = SwmdEditorEnv
