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
from harness.model_client import DummyModelClient, ExLlamaV2Client, OpenAIAPIClient, OpenAICompatibleClient
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
    if provider == "openai_api":
        return OpenAIAPIClient(
            model_name=spec.get("model_name", "o3"),
            api_key=spec.get("api_key"),
            api_key_path=spec.get("api_key_path"),
            max_output_tokens=int(spec.get("max_output_tokens", spec.get("max_new_tokens", 512))),
            reasoning_effort=spec.get("reasoning_effort"),
        )
    if provider == "dummy-model":
        return DummyModelClient(model_name=spec.get("model_name", "dummy"))
    raise ValueError(f"Unsupported provider: {provider}")


def run_episode(model, export_path: str, env_spec, max_steps: int, episode_index: int):
    logger = JsonlLogger(export_path)
    runner = EpisodeRunner(model, logger)
    env = build_env(env_spec)
    records = runner.run_episode(env, max_steps=max_steps)
    token_total = 0
    for record in records:
        usage = record.usage or {}
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, int):
                token_total += total
            else:
                prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
                if isinstance(prompt, int):
                    token_total += prompt
                if isinstance(completion, int):
                    token_total += completion
    return {
        "env_name": env.name,
        "env_type": getattr(env, "env_type", env.__class__.__name__),
        "episode_index": episode_index,
        "steps": len(records),
        "final_reward": records[-1].reward if records else None,
        "done": bool(records[-1].done) if records else False,
        "failure_types": [r.failure_type for r in records if r.failure_type],
        "episode_id": records[0].episode_id if records else None,
        "token_total": token_total,
    }


def summarize(export_path: str):
    path = Path(export_path)
    per_env = Counter()
    failures = Counter()
    rewards = defaultdict(float)
    token_usage = defaultdict(int)
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
            usage = row.get("usage") or (row.get("raw_model") or {}).get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("input_tokens"), int):
                    token_usage["input_tokens"] += int(usage["input_tokens"])
                if isinstance(usage.get("prompt_tokens"), int):
                    token_usage["prompt_tokens"] += int(usage["prompt_tokens"])
                if isinstance(usage.get("output_tokens"), int):
                    token_usage["output_tokens"] += int(usage["output_tokens"])
                if isinstance(usage.get("completion_tokens"), int):
                    token_usage["completion_tokens"] += int(usage["completion_tokens"])
                if isinstance(usage.get("total_tokens"), int):
                    token_usage["total_tokens"] += int(usage["total_tokens"])
            if row.get("episode_id"):
                episodes.add(row["episode_id"])
    return {
        "steps": steps,
        "episodes": len(episodes),
        "per_env_steps": dict(per_env),
        "failure_types": dict(failures),
        "reward_totals": dict(rewards),
        "token_usage": dict(token_usage),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--token-budget", type=int, default=None)
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
    total_tokens = 0
    for env_spec in env_specs:
        for episode_index in range(episodes):
            summary = run_episode(model, export_path, env_spec, max_steps, episode_index)
            summaries.append(summary)
            total_tokens += int(summary.get("token_total") or 0)
            print(
                f"done env={summary['env_name']} type={summary['env_type']} "
                f"ep={summary['episode_index']} steps={summary['steps']} done={summary['done']} "
                f"tokens={summary.get('token_total', 0)} total_tokens={total_tokens}"
            )
            if args.token_budget is not None and total_tokens >= args.token_budget:
                print(f"Token budget reached: {total_tokens} >= {args.token_budget}. Stopping early.")
                env_specs = []
                break
        if not env_specs:
            break

    report = summarize(export_path)
    report["run_token_total"] = total_tokens
    report["token_budget"] = args.token_budget
    report_path = Path(export_path).with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed {len(summaries)} episodes.")
    print(f"Replay data written to {Path(export_path).resolve()}")
    print(f"Summary written to {report_path.resolve()}")


if __name__ == "__main__":
    main()
