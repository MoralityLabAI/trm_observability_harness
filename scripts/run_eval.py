from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.logger import JsonlLogger
from harness.model_client import DummyModelClient, ExLlamaV2Client, OpenAICompatibleClient
from harness.runner import EpisodeRunner
from envs.registry import build_env


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(spec):
    provider = spec.get("provider", "dummy")
    if provider == "dummy":
        return DummyModelClient(spec.get("model_name", "dummy-model"))
    if provider == "exllamav2":
        return ExLlamaV2Client(
            model_dir=spec["model_dir"],
            max_new_tokens=int(spec.get("max_new_tokens", 160)),
            temperature=float(spec.get("temperature", 0.2)),
            top_p=float(spec.get("top_p", 0.9)),
            top_k=int(spec.get("top_k", 40)),
            max_seq_len=int(spec.get("max_seq_len", 4096)),
            gpu_split=spec.get("gpu_split"),
        )
    if provider == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=spec["base_url"],
            model_name=spec.get("model_name", "qwen3.5-27b"),
            max_new_tokens=int(spec.get("max_new_tokens", 160)),
            temperature=float(spec.get("temperature", 0.2)),
            top_p=float(spec.get("top_p", 0.9)),
        )
    if provider == "dummy-model":
        return DummyModelClient(model_name=spec.get("model_name", "dummy"))
    raise ValueError(f"Unsupported provider: {provider}")


def run_episode(model, export_path: str, env_spec, max_steps: int, episode_index: int):
    logger = JsonlLogger(export_path)
    runner = EpisodeRunner(model, logger)
    env = build_env(env_spec)
    records = runner.run_episode(env, max_steps=max_steps)
    return {
        "env_name": env.name,
        "env_type": getattr(env, "env_type", env.__class__.__name__),
        "episode_index": episode_index,
        "steps": len(records),
        "final_reward": records[-1].reward if records else None,
        "done": bool(records[-1].done) if records else False,
        "failure_types": [r.failure_type for r in records if r.failure_type],
        "episode_id": records[0].episode_id if records else None,
    }


def summarize(export_path: str):
    path = Path(export_path)
    per_env = Counter()
    failures = Counter()
    rewards = defaultdict(float)
    steps = 0
    episodes = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            steps += 1
            per_env[row["env_name"]] += 1
            if row.get("failure_type"):
                failures[row["failure_type"]] += 1
            rewards[row["env_name"]] += float(row.get("reward") or 0.0)
            if row.get("episode_id"):
                episodes.add(row["episode_id"])
    return {
        "steps": steps,
        "episodes": len(episodes),
        "per_env_steps": dict(per_env),
        "failure_types": dict(failures),
        "reward_totals": dict(rewards),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_path = cfg["export_path"]
    episodes = int(cfg.get("max_episodes", 1))
    max_steps = int(cfg.get("max_steps_per_episode", 8))
    env_specs = cfg.get("envs", [])

    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    open(export_path, "a", encoding="utf-8").close()

    model = build_model(cfg["model"])

    summaries = []
    for env_spec in env_specs:
        for episode_index in range(episodes):
            summary = run_episode(model, export_path, env_spec, max_steps, episode_index)
            summaries.append(summary)
            print(
                f"done env={summary['env_name']} type={summary['env_type']} "
                f"ep={summary['episode_index']} steps={summary['steps']} done={summary['done']}"
            )

    report = summarize(export_path)
    report_path = Path(export_path).with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed {len(summaries)} episodes.")
    print(f"Replay data written to {Path(export_path).resolve()}")
    print(f"Summary written to {report_path.resolve()}")


if __name__ == "__main__":
    main()
