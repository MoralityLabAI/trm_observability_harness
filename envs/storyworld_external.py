from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from .base import BaseEnv


class ExternalStoryWorldEnv(BaseEnv):
    """Adapter for a local external GPTStoryworld-style repo.

    The external command is expected to print a single JSON object to stdout.
    Reset output schema:
      {"observation": "...", "task": "...", "session": {...optional...}}
    Step output schema:
      {
        "observation": "...",
        "reward": 0.0,
        "done": false,
        "valid_action": true,
        "failure_type": null,
        "episode_summary": null
      }
    """

    def __init__(self, name: str, repo_path: str, command_template: List[str], reset_args: List[str] | None = None):
        self.name = name
        self.repo_path = str(Path(repo_path))
        self.command_template = command_template
        self.reset_args = reset_args or []
        self.state: Dict[str, Any] = {}

    def _run(self, extra_args: List[str], payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cmd = [part.format(repo_path=self.repo_path) for part in self.command_template] + list(extra_args)
        proc = subprocess.run(
            cmd,
            cwd=self.repo_path,
            input=(json.dumps(payload) if payload is not None else None),
            text=True,
            capture_output=True,
            check=True,
        )
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f'External StoryWorld adapter did not return JSON. stderr={proc.stderr}') from e

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        out = self._run(self.reset_args)
        self.state = out.get('session', {})
        return out['observation'], {'task': out.get('task', f'Complete task in {self.name}')}

    def step(self, decision):
        action = getattr(decision, 'action', decision)
        payload = {'action': action, 'state': self.state}
        out = self._run(['--step'], payload=payload)
        self.state = out.get('session', self.state)
        info = {
            'valid_action': out.get('valid_action', True),
            'failure_type': out.get('failure_type'),
            'episode_summary': out.get('episode_summary'),
        }
        return out['observation'], float(out.get('reward', 0.0)), bool(out.get('done', False)), info

    def trace_profile(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        profile = super().trace_profile(meta)
        profile.update(
            {
                "family": "storyworld",
                "max_trace_steps": 5,
                "action_guidance": "Use action for the exact storyworld bridge command or payload expected by the adapter.",
                "step_labels": ["state_read", "agent_intent", "risk_check", "next_move", "commit"],
            }
        )
        return profile
