from __future__ import annotations
from typing import Any, Dict
from .dummy_primehub import DummyPrimeHubEnv
from .storyworld_external import ExternalStoryWorldEnv
from .primehub_external import ExternalPrimeHubEnv
from .swmd_editor import SwmdEditorEnv
from .storyworld_native import NativeStoryWorldEnv
from .reasoning_gym import ReasoningGymEnv
from .needle_pathfinding import NeedlePathfindingEnv


def build_env(spec: Dict[str, Any]):
    env_type = spec['type']
    name = spec.get('name', env_type)
    if env_type == 'dummy_primehub':
        return DummyPrimeHubEnv(name)
    if env_type == 'needle_pathfinding':
        return NeedlePathfindingEnv(
            name=name,
            repo_path=spec['repo_path'],
            storyworld_path=spec['storyworld_path'],
            target_ending=spec['target_ending'],
            n_endings=spec.get('n_endings', 12),
            max_steps=spec.get('max_steps', 5),
        )
    if env_type in {'swmd_editor', 'sweepweave_prime'}:
        return SwmdEditorEnv(
            name=name,
            repo_path=spec['repo_path'],
            load_kwargs=spec.get('load_kwargs', {}),
        )
    if env_type == 'reasoning_gym':
        return ReasoningGymEnv(
            name=name,
            gym_config=spec['gym_config'],
            num_examples=spec.get('num_examples', 50),
            seed=spec.get('seed', 42),
        )
    if env_type == 'storyworld_native':
        return NativeStoryWorldEnv(
            name=name,
            repo_path=spec['repo_path'],
            storyworld_path=spec['storyworld_path'],
            seed=spec.get('seed'),
            log_path=spec.get('log_path'),
        )
    if env_type == 'storyworld_external':
        return ExternalStoryWorldEnv(
            name=name,
            repo_path=spec['repo_path'],
            command_template=spec['command_template'],
            reset_args=spec.get('reset_args', []),
        )
    if env_type == 'primehub_external':
        return ExternalPrimeHubEnv(
            name=name,
            command_template=spec['command_template'],
            reset_args=spec.get('reset_args', []),
        )
    raise ValueError(f'Unsupported env type: {env_type}')
