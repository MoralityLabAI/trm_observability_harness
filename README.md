# TRM Observability Harness

A lightweight starter repo for running a stronger base model through environment episodes and exporting a structured replay corpus for downstream TRM training.

## What this is

This harness is designed to:
- run model-driven episodes across multiple environments
- log observations, actions, bounded step-wise reasoning traces, short visible justifications, confidence, outcomes, tool calls, and recovery attempts
- export JSONL replay data for later training of TRMs as monitors, critics, recovery modules, and route selectors
- support a local ExLlamaV2 backend for EXL2 / ExLlamaV2-friendly quantized models
- support an OpenAI-compatible server path for a larger remote GPU box or local llama-server
- rapid-fire episodes with configurable concurrency
- accept a local GPTStoryworld checkout path and generic external Prime Hub wrappers

## What this is not

- not unrestricted hidden chain-of-thought capture
- not a complete Dragon Rider / router pipeline
- not a finalized Prime Hub or GPTStoryworld integration; those use adapter shims you can match to the real repo APIs

## Layout

- `harness/` core runner, schemas, logging, model clients
- `envs/` environment adapters and registry
- `configs/` example run configs
- `scripts/` entrypoints and repo fetch helper
- `docs/` schema and extension notes

## ExLlamaV2 notes

The example config assumes:
- a local ExLlamaV2-compatible model directory
- a quantized model format appropriate for ExLlamaV2, such as EXL2
- enough VRAM/RAM for your chosen quantization

This specific workstation has a 4 GB RTX 3050 Laptop GPU, so Qwopus 27B does not fit locally here. Use a 24 GB-class GPU box, a remote OpenAI-compatible server, or a smaller stand-in model for smoke tests.

Update this field in `configs/example_primehub.yaml`:

```yaml
model:
  provider: exllamav2
  model_dir: /models/Qwopus-27B-EXL2
```

The client prompts the model to emit a compact JSON object:
- `reasoning_trace`
- `reasoning_summary`
- `action`
- `short_justification`
- `confidence`
- `action_type`

The trace is bounded and environment-aware:
- reasoning envs: `givens`, `strategy`, `compute`, `verify`
- storyworld envs: `state_read`, `agent_intent`, `risk_check`, `next_move`, `commit`
- Prime Hub envs: `task_parse`, `constraint_check`, `candidate`, `self_check`

This keeps the replay export structured, step-wise, and trainable across different env families.

## Quick start

Clone GPTStoryworld locally if you want StoryWorld runs:

```bash
python scripts/fetch_storyworld_repo.py --repo-url <GPTSTORYWORLD_REPO_URL>
```

Then edit `configs/example_primehub.yaml` to match your local paths and env entrypoints.

Run collection:

```bash
python scripts/run_eval.py --config configs/example_primehub.yaml
python scripts/summarize_replays.py
```

Remote 3090 box:

```bash
python scripts/run_eval.py --config configs/snacksack_remote.yaml
```

To inspect the local runtime before a run:

```bash
python scripts/inspect_runtime.py
```

## Rapid-fire collection

Use the config field:

```yaml
concurrency: 4
```

This spins up multiple episode workers. Start conservative because each worker may load its own model instance depending on your runtime. On a single-GPU ExLlamaV2 machine, practical concurrency may be 1 unless you restructure around a shared inference server. For CPU-bound or external-env-only runs, higher concurrency can help.

## External adapter contract

Both external adapters expect the target command to print a single JSON object to stdout.

Reset:

```json
{"observation": "...", "task": "...", "session": {}}
```

Step:

```json
{
  "observation": "...",
  "reward": 0.0,
  "done": false,
  "valid_action": true,
  "failure_type": null,
  "episode_summary": null,
  "session": {}
}
```

This is deliberate: it gives your brother a thin bridge layer to add on top of the real GPTStoryworld repo and Prime Hub envs without rewriting the harness.

## Suggested next steps

1. Add a `bridge.py` file to GPTStoryworld that conforms to the JSON contract above.
2. Point `primehub_external` commands at the real Prime Env Hub CLI/module entrypoints.
3. If true rapid fire is needed on one model, replace per-worker local model loading with a single shared inference server process.
4. Expand schema fields for failure tags, retries, and reward shaping.
5. Run episodes and inspect `data/replays.jsonl`.
