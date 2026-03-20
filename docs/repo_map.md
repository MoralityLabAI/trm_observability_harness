# Repo Map

## Local repos discovered

- `C:/projects/prime-environments`
  - Source of `sweepweave.load_environment(...)`
  - Best seam for Prime Hub-style synthetic/verifier tasks
- `C:/projects/GPTStoryworld`
  - Source of `storyworld.env.DiplomacyStoryworldEnv`
  - Best seam for native storyworld rollouts and multi-agent control traces
- `C:/projects/Tesseract`
  - Existing router / TRM-style dataset and control-plane artifacts
- `C:/projects/TRM`
  - Supporting docs and control-plane references

## Harness modules

- `harness/model_client.py`
  - `DummyModelClient`
  - `ExLlamaV2Client`
  - `OpenAICompatibleClient`
- `harness/runner.py`
  - Single-episode loop and replay record assembly
- `harness/schemas.py`
  - Step-level replay schema plus derived labels
- `envs/registry.py`
  - Adapter dispatch for dummy, native, and external environments
- `envs/sweepweave_native.py`
  - Native adapter for `prime-environments`
- `envs/storyworld_native.py`
  - Native adapter for `GPTStoryworld`
- `envs/primehub_external.py`
  - Thin bridge for CLI-driven Prime Hub-like environments
- `envs/storyworld_external.py`
  - Thin bridge for CLI-driven storyworld environments

## Missing seams

- A local 24 GB-class GPU host for Qwopus 27B EXL2 / ExLlamaV2
- A stable, documented Prime Hub CLI contract for all target envs
- A canonical storyworld sample path in GPTStoryworld
- A shared inference server for one-model multi-env rapid-fire runs

