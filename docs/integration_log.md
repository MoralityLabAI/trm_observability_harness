# Integration Log

## 2026-03-18

- Extracted the harness starter archive into `C:/projects/trm_observability_harness`
- Confirmed local hardware: RTX 3050 Laptop GPU with 4 GB VRAM, 24.5 GB system RAM
- Conclusion: Qwopus 27B in local ExLlamaV2 is not feasible on this machine
- Added a safe OpenAI-compatible client path for larger GPU boxes or a local server
- Added native adapters for:
  - `prime-environments` Sweepweave
  - GPTStoryworld Diplomacy storyworlds
- Kept replay outputs bounded:
  - short visible justification only
  - action payloads and environment transitions
  - no hidden chain-of-thought capture
- On `snacksack`:
  - confirmed NVIDIA RTX 3090 with 24 GB VRAM
  - downloaded `Qwen3.5-27B.Q4_K_M.gguf` to `~/Qwopus/models`
  - bootstrapped user-local `pip`
  - installed `torch`, `safetensors`, `sentencepiece`, `ninja`, and `exllamav2`
  - started a background install for `llama-cpp-python[server]==0.3.16` from the cu121 wheel index
  - added a readiness watcher that starts `qwopus.service` when the model and runtime are both present

## Current assumptions

- Native GPTStoryworld can load `C:/projects/GPTStoryworld/raw_storyworld.json`
- Native Sweepweave can be imported from `C:/projects/prime-environments/environments/sweepweave_prime_env`
- The model runtime will be either:
  - an EXL2/ExLlamaV2 server on a bigger GPU box, or
  - an OpenAI-compatible endpoint
- The `snacksack` model endpoint will bind on Tailscale-accessible `0.0.0.0:8080`

## Open gaps

- No benchmarked throughput numbers yet because the target runtime is still finishing install
- No Prime Hub CLI contract was exercised yet
