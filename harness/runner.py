from __future__ import annotations
import uuid
from typing import List, Dict, Any
from .schemas import StepRecord
from .trace_contract import TRACE_CONTRACT_VERSION

class EpisodeRunner:
    def __init__(self, model_client, logger):
        self.model_client = model_client
        self.logger = logger

    def run_episode(self, env, max_steps: int = 8) -> List[StepRecord]:
        episode_id = str(uuid.uuid4())
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            observation, meta = reset_out
        else:
            observation, meta = reset_out, {}
        task = meta.get('task', 'Unknown task')
        trace_profile = env.trace_profile(meta)
        records: List[StepRecord] = []

        for step_id in range(max_steps):
            resp = self.model_client.act(
                task=task,
                observation=observation,
                context={
                    'env_name': env.name,
                    'env_type': getattr(env, 'env_type', env.__class__.__name__),
                    'meta': meta,
                    'step_id': step_id,
                    'max_steps': max_steps,
                    'trace_profile': trace_profile,
                },
            )
            next_obs, reward, done, info = env.step(resp)
            outcome = info.get('outcome') if isinstance(info, dict) else None
            labels = self._derive_labels(resp.action_type, reward, info)
            record = StepRecord(
                env_name=env.name,
                env_type=getattr(env, 'env_type', env.__class__.__name__),
                episode_id=episode_id,
                step_id=step_id,
                task=task,
                observation=observation,
                action=resp.action,
                action_type=resp.action_type,
                action_args=resp.action_args,
                short_justification=resp.short_justification,
                confidence=resp.confidence,
                valid_action=info.get('valid_action'),
                reward=reward,
                score=info.get('score', reward),
                done=done,
                failure_type=info.get('failure_type'),
                retry_attempt=info.get('retry_attempt'),
                recovery_action=info.get('recovery_action') or ('retry_with_safe_defaults' if 'recovery' in resp.action_type else None),
                tool_calls=resp.tool_calls or [],
                episode_summary=info.get('episode_summary') if done else None,
                next_action=labels.get('next_action'),
                action_class=labels.get('action_class'),
                success_likelihood=labels.get('success_likelihood'),
                constraint_risk=labels.get('constraint_risk'),
                recovery_needed=labels.get('recovery_needed'),
                recovery_action_class=labels.get('recovery_action_class'),
                thought=resp.thought,
                diary=resp.diary,
                trace_contract_version=resp.trace_contract_version or trace_profile.get('contract_version', TRACE_CONTRACT_VERSION),
                trace_mode=resp.trace_mode or trace_profile.get('mode', 'stepwise'),
                reasoning_trace=resp.reasoning_trace,
                reasoning_summary=resp.reasoning_summary,
                outcome=outcome,
                raw_env=info,
                raw_model={
                    'raw_text': resp.raw_text,
                    'thought': resp.thought,
                    'diary': resp.diary,
                    'reasoning_trace': resp.reasoning_trace,
                    'reasoning_summary': resp.reasoning_summary,
                },
                meta=meta,
            )
            self.logger.write(record)
            records.append(record)
            observation = next_obs
            if done:
                break
        return records

    def _derive_labels(self, action_type: str | None, reward: float | None, info: Dict[str, Any]) -> Dict[str, Any]:
        reward_val = float(reward or 0.0)
        valid = bool(info.get('valid_action', True))
        failure = info.get('failure_type')
        recovery_needed = (not valid) or bool(failure)
        success_likelihood = min(1.0, max(0.0, 0.5 + reward_val / 2.0))
        constraint_risk = 0.8 if recovery_needed else 0.2
        action_class = action_type or 'generic'
        recovery_action_class = 'recovery' if recovery_needed else None
        next_action = 'recover' if recovery_needed else 'continue'
        return {
            'next_action': next_action,
            'action_class': action_class,
            'success_likelihood': round(success_likelihood, 4),
            'constraint_risk': round(constraint_risk, 4),
            'recovery_needed': recovery_needed,
            'recovery_action_class': recovery_action_class,
        }
