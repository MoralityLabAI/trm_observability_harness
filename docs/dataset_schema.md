# Replay Dataset Schema

Each JSONL row represents one environment step.

Fields:
- `env_name`: environment identifier
- `env_type`: environment family or adapter type
- `episode_id`: unique episode ID
- `step_id`: integer step index
- `task`: top-level task description
- `observation`: current environment observation
- `action`: action emitted by the model/controller
- `action_type`: optional action category
- `action_args`: optional structured action arguments
- `short_justification`: short visible rationale summary
- `confidence`: float confidence estimate in [0,1]
- `valid_action`: boolean or null
- `reward`: numeric reward if available
- `score`: adapter-specific score or reward proxy
- `done`: episode completion flag
- `failure_type`: compact failure taxonomy label
- `retry_attempt`: retry count if the policy attempted recovery
- `recovery_action`: next repair action or null
- `tool_calls`: structured tool calls or env action payloads
- `episode_summary`: optional final summary on terminal step
- `next_action`: derived label for next-step prediction
- `action_class`: normalized action class
- `success_likelihood`: derived success probability label
- `constraint_risk`: derived risk label
- `recovery_needed`: derived boolean
- `recovery_action_class`: normalized recovery class
- `outcome`: compact terminal or transition outcome
- `raw_env`: raw adapter payload for debugging
- `raw_model`: raw model payload for debugging
- `trace_contract_version`: prompt-and-parse contract version for this row
- `trace_mode`: trace collection mode, currently `stepwise`
- `reasoning_trace`: bounded visible reasoning steps as `[{label, content}]`
- `reasoning_summary`: one-sentence summary of why the chosen action was selected
- `meta`: reset-time environment metadata

Trace policy:
- keep traces bounded and step-wise rather than long monologues
- reasoning tasks should usually use labels like `givens`, `strategy`, `compute`, `verify`
- storyworld tasks should usually use labels like `state_read`, `agent_intent`, `risk_check`, `next_move`, `commit`
- Prime Hub and verifier tasks should usually use labels like `task_parse`, `constraint_check`, `candidate`, `self_check`

Compatibility:
- if a model still emits `<think>...</think>`, the harness may salvage it into `reasoning_trace`
- `thought` and `diary` remain optional compatibility fields, not the primary training target
