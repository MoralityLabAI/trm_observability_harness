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
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
