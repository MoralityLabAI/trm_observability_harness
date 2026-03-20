# Run Summary

## Smoke run

- Config: `C:/tmp/trm_observability_harness_smoke.yaml`
- Model: `DummyModelClient`
- Env: `dummy_primehub_smoke`
- Episodes: 2
- Steps: 6

## Observed metrics

- `valid_action`: 6/6 true
- `failure_types`: none
- `top_actions`: `inspect_and_continue`
- `reward_total`: 2.4

## Notes

- This confirms the JSONL replay path, derived labels, and summary generation.
- It does not validate Qwopus 27B inference on this laptop.
- The machine needs either a larger GPU host or a remote server for the target 27B runtime.

