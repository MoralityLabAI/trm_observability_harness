from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/replays.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"No replay file found at {path}")

    per_env = Counter()
    env_types = Counter()
    actions = Counter()
    failure_types = Counter()
    rewards = defaultdict(float)
    valid = Counter()
    trace_rows = 0
    trace_steps = 0
    token_usage = defaultdict(int)
    steps = 0
    episodes = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            steps += 1
            env = row.get("env_name", "unknown")
            per_env[env] += 1
            env_types[row.get("env_type", "unknown")] += 1
            actions[row.get("action", "unknown")] += 1
            rewards[env] += float(row.get("reward") or 0.0)
            trace = row.get("reasoning_trace") or []
            if trace:
                trace_rows += 1
                trace_steps += len(trace)
            usage = row.get("usage") or (row.get("raw_model") or {}).get("usage")
            if isinstance(usage, dict):
                for key in ("input_tokens", "prompt_tokens"):
                    if isinstance(usage.get(key), int):
                        token_usage[key] += int(usage[key])
                for key in ("output_tokens", "completion_tokens"):
                    if isinstance(usage.get(key), int):
                        token_usage[key] += int(usage[key])
                if isinstance(usage.get("total_tokens"), int):
                    token_usage["total_tokens"] += int(usage["total_tokens"])
            if row.get("valid_action") is True:
                valid["true"] += 1
            elif row.get("valid_action") is False:
                valid["false"] += 1
            if row.get("failure_type"):
                failure_types[row["failure_type"]] += 1
            if row.get("episode_id"):
                episodes.add(row["episode_id"])

    report = {
        "steps": steps,
        "episodes": len(episodes),
        "per_env_rows": dict(per_env),
        "env_types": dict(env_types),
        "top_actions": actions.most_common(20),
        "failure_types": dict(failure_types),
        "reward_totals": dict(rewards),
        "valid_action_counts": dict(valid),
        "trace_rows": trace_rows,
        "avg_trace_steps_per_row": round(trace_steps / trace_rows, 4) if trace_rows else 0.0,
        "token_usage": dict(token_usage),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
