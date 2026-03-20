from __future__ import annotations

from typing import Any, Dict, List


TRACE_CONTRACT_VERSION = "trm_trace_v1"


def default_trace_profile(env_name: str, env_type: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    meta = meta or {}
    source = str(meta.get("source", env_type))
    profile: Dict[str, Any] = {
        "contract_version": TRACE_CONTRACT_VERSION,
        "family": "generic",
        "mode": "stepwise",
        "max_trace_steps": 4,
        "max_step_chars": 220,
        "action_guidance": "Commit one action that the environment can score immediately.",
        "step_labels": [
            "state_read",
            "constraints",
            "options",
            "decision",
        ],
    }

    if env_type == "reasoning_gym" or "reasoning" in source:
        profile.update(
            {
                "family": "reasoning",
                "action_guidance": "Return only the final answer string in action.",
                "step_labels": [
                    "givens",
                    "strategy",
                    "compute",
                    "verify",
                ],
            }
        )
    elif env_type in {"storyworld_native", "storyworld_external", "needle_pathfinding"} or "storyworld" in source:
        profile.update(
            {
                "family": "storyworld",
                "max_trace_steps": 5,
                "action_guidance": "Choose the next action or structured action payload that safely advances the storyworld.",
                "step_labels": [
                    "state_read",
                    "agent_intent",
                    "risk_check",
                    "next_move",
                    "commit",
                ],
            }
        )
    elif env_type in {"swmd_editor", "sweepweave_prime", "primehub_external"} or "prime" in source:
        profile.update(
            {
                "family": "swmd_editor",
                "action_guidance": "Return a verifier-friendly edit action or payload that obeys the SWMD editor contract exactly.",
                "step_labels": [
                    "task_parse",
                    "constraint_check",
                    "candidate",
                    "self_check",
                ],
            }
        )
    return profile


def build_system_prompt(profile: Dict[str, Any]) -> str:
    labels = ", ".join(profile.get("step_labels", []))
    max_steps = int(profile.get("max_trace_steps", 4))
    max_chars = int(profile.get("max_step_chars", 220))
    family = profile.get("family", "generic")
    guidance = profile.get("action_guidance", "Commit one action.")
    contract_version = profile.get("contract_version", TRACE_CONTRACT_VERSION)
    return (
        "You are an environment control agent.\n"
        f"Use the structured trace contract {contract_version}.\n"
        f"Environment family: {family}.\n"
        "Think in bounded visible steps, not a long monologue.\n"
        f"Emit 2 to {max_steps} reasoning_trace items using these labels when appropriate: {labels}.\n"
        f"Each reasoning_trace item must be short, at most about {max_chars} characters.\n"
        f"Action rule: {guidance}\n"
        "Return exactly one JSON object with these keys:\n"
        "- reasoning_trace: array of objects with label and content\n"
        "- reasoning_summary: one short sentence\n"
        "- action: final action string for the environment\n"
        "- short_justification: short visible justification\n"
        "- confidence: float from 0.0 to 1.0\n"
        "- action_type: short action category\n"
        "- action_args: optional object\n"
        "- tool_calls: optional list\n"
        "Do not use markdown fences.\n"
    )


def normalize_reasoning_trace(value: Any, max_steps: int = 6) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if isinstance(value, list):
        for item in value[:max_steps]:
            if isinstance(item, dict):
                label = str(item.get("label", "step")).strip() or "step"
                content = str(item.get("content", "")).strip()
            else:
                label = "step"
                content = str(item).strip()
            if content:
                normalized.append({"label": label, "content": content})
    elif isinstance(value, str) and value.strip():
        for idx, line in enumerate(value.splitlines()[:max_steps]):
            content = line.strip()
            if content:
                normalized.append({"label": f"step_{idx + 1}", "content": content})
    return normalized
