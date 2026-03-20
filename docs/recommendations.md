# TRM Training Priorities

## First models to train

1. `next_action` classifier
   - Highest leverage control-plane label
2. `recovery_needed` and `recovery_action_class`
   - Useful for rollback, retry, and constraint repair
3. `valid_action`
   - Strong verifier / critic signal
4. `constraint_risk`
   - Good for policy gating and refusal prevention
5. `success_likelihood`
   - Useful for route selection and confidence calibration

## Recommended corpus mix

- Sweepweave for structured generation and verifier-rich failures
- GPTStoryworld for multi-agent state transitions and message/action coupling
- Prime Hub-style external tasks for planning, retrieval, and constrained decision-making

## Trace collection policy

- Prefer bounded `reasoning_trace` steps over free-form hidden CoT dumps
- Use environment-family labels so traces are comparable across runs
- Keep `action` clean and verifier-friendly; do not bury the answer inside the trace
- Treat `thought` and `diary` as compatibility fields only

## Keep bounded

- Preserve short visible justifications and one-sentence `reasoning_summary`
- Store action payloads, state, and outcome labels
- Cap trace length per step so the corpus stays trainable
