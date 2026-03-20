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

## Keep bounded

- Preserve short visible justifications only
- Store action payloads, state, and outcome labels
- Avoid raw hidden reasoning dumps

