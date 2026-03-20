from __future__ import annotations
import json
import os
import re
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List

from .trace_contract import TRACE_CONTRACT_VERSION, build_system_prompt, default_trace_profile, normalize_reasoning_trace


@dataclass
class ModelResponse:
    action: str
    short_justification: str
    confidence: float
    action_type: str = 'generic'
    action_args: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: Optional[str] = None
    thought: Optional[str] = None
    diary: Optional[str] = None
    reasoning_trace: List[Dict[str, str]] = field(default_factory=list)
    reasoning_summary: Optional[str] = None
    trace_contract_version: Optional[str] = None
    trace_mode: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class DummyModelClient:
    def __init__(self, model_name: str = 'dummy-model'):
        self.model_name = model_name

    def act(self, task: str, observation: str, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        trace_profile = (context or {}).get('trace_profile') or default_trace_profile('dummy', 'dummy', {})
        text = observation.lower()
        if 'invalid' in text or 'error' in text:
            return ModelResponse(
                action='retry_with_safe_defaults',
                short_justification='Previous step appears faulty; use conservative recovery.',
                confidence=0.62,
                action_type='recovery',
                action_args={'mode': 'safe_defaults'},
                tool_calls=[],
                thought='The observation indicates an error state. Recovery is required.',
                reasoning_trace=[
                    {'label': 'state_read', 'content': 'The environment looks invalid or errorful.'},
                    {'label': 'decision', 'content': 'Use the safest recovery action available.'},
                ],
                reasoning_summary='Recover conservatively from the visible error state.',
                trace_contract_version=trace_profile.get('contract_version', TRACE_CONTRACT_VERSION),
                trace_mode=trace_profile.get('mode', 'stepwise'),
                usage=None,
            )
        return ModelResponse(
            action='inspect_and_continue',
            short_justification='Advance cautiously based on the current observation.',
            confidence=0.71,
            action_type='progress',
            action_args={'mode': 'inspect'},
            tool_calls=[],
            thought='State seems normal. Proceeding with standard inspection.',
            reasoning_trace=[
                {'label': 'state_read', 'content': 'The current state appears stable enough to continue.'},
                {'label': 'decision', 'content': 'Inspect and continue without invoking recovery.'},
            ],
            reasoning_summary='Continue with a low-risk inspection move.',
            trace_contract_version=trace_profile.get('contract_version', TRACE_CONTRACT_VERSION),
            trace_mode=trace_profile.get('mode', 'stepwise'),
            usage=None,
        )


class ExLlamaV2Client:
    """Lightweight ExLlamaV2 wrapper.

    Expects a local model directory containing EXL2/GPTQ-compatible files for ExLlamaV2.
    The model is prompted to return a compact JSON object with:
      - action
      - short_justification
      - confidence
      - action_type
    """

    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int = 160,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        max_seq_len: int = 4096,
        gpu_split: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.gpu_split = gpu_split

        try:
            from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
            from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
        except Exception as e:  # pragma: no cover
            raise ImportError(
                'ExLlamaV2 is not installed. Install it in the runtime environment before using this client.'
            ) from e

        self._ExLlamaV2 = ExLlamaV2
        self._ExLlamaV2Config = ExLlamaV2Config
        self._ExLlamaV2Cache = ExLlamaV2Cache
        self._ExLlamaV2Tokenizer = ExLlamaV2Tokenizer
        self._ExLlamaV2DynamicGenerator = ExLlamaV2DynamicGenerator
        self._ExLlamaV2Sampler = ExLlamaV2Sampler

        self._load()

    def _load(self):
        config = self._ExLlamaV2Config()
        config.model_dir = self.model_dir
        config.prepare()
        config.max_seq_len = self.max_seq_len

        self.model = self._ExLlamaV2(config)
        if self.gpu_split:
            split = [float(x) for x in self.gpu_split.split(',')]
            self.model.load(split)
        else:
            self.model.load()

        self.tokenizer = self._ExLlamaV2Tokenizer(config)
        self.cache = self._ExLlamaV2Cache(self.model, lazy=True)
        self.generator = self._ExLlamaV2DynamicGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )

    def _build_prompt(self, task: str, observation: str, context: Optional[Dict[str, Any]] = None) -> str:
        trace_profile = (context or {}).get('trace_profile') or default_trace_profile('unknown', 'generic', {})
        env_name = (context or {}).get('env_name', 'unknown')
        env_type = (context or {}).get('env_type', 'generic')
        step_id = int((context or {}).get('step_id', 0))
        max_steps = int((context or {}).get('max_steps', 1))
        return (
            f"{build_system_prompt(trace_profile)}\n"
            f"ENV_NAME: {env_name}\n"
            f"ENV_TYPE: {env_type}\n"
            f"STEP: {step_id + 1}/{max_steps}\n\n"
            f"TASK:\n{task}\n\n"
            f"OBSERVATION:\n{observation}\n\n"
            "JSON:"
        )

    def _extract_thought(self, text: str) -> Optional[str]:
        match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_diary(self, text: str) -> Optional[str]:
        match = re.search(r'<diary>(.*?)</diary>', text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_json(self, text: str) -> dict:
        # Strip both think and diary tags for cleaner JSON matching
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<diary>.*?</diary>', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        match = re.search(r'\{.*\}', clean_text if clean_text else text, flags=re.DOTALL)
        if not match:
            return {
                'action': 'inspect_and_continue',
                'short_justification': 'Model did not return valid JSON; using safe fallback.',
                'confidence': 0.2,
                'action_type': 'fallback',
            }
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                'action': 'inspect_and_continue',
                'short_justification': 'JSON parse failed; using safe fallback.',
                'confidence': 0.2,
                'action_type': 'fallback',
                'reasoning_trace': [],
                'reasoning_summary': 'JSON parsing failed.',
            }
        return obj

    def act(self, task: str, observation: str, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        trace_profile = (context or {}).get('trace_profile') or default_trace_profile('unknown', 'generic', {})
        prompt = self._build_prompt(task=task, observation=observation, context=context)

        settings = self._ExLlamaV2Sampler.Settings()
        settings.temperature = self.temperature
        settings.top_p = self.top_p
        settings.top_k = self.top_k
        settings.token_repetition_penalty = 1.05

        raw = self.generator.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            add_bos=True,
            stop_conditions=['\n\n', '</s>'],
            gen_settings=settings,
        )
        parsed = self._extract_json(raw)
        thought = self._extract_thought(raw)
        diary = self._extract_diary(raw)
        return ModelResponse(
            action=str(parsed.get('action', 'inspect_and_continue')),
            short_justification=str(parsed.get('short_justification', 'No justification provided.')),
            confidence=float(parsed.get('confidence', 0.5)),
            action_type=str(parsed.get('action_type', 'generic')),
            action_args=parsed.get('action_args') if isinstance(parsed.get('action_args'), dict) else None,
            tool_calls=parsed.get('tool_calls') if isinstance(parsed.get('tool_calls'), list) else [],
            raw_text=raw,
            thought=thought,
            diary=diary,
            reasoning_trace=_fallback_trace_from_text(parsed, thought, trace_profile),
            reasoning_summary=str(parsed.get('reasoning_summary', '')).strip() or None,
            trace_contract_version=trace_profile.get('contract_version', TRACE_CONTRACT_VERSION),
            trace_mode=trace_profile.get('mode', 'stepwise'),
            usage=None,
        )


class OpenAICompatibleClient:
    """Client for local or remote OpenAI-compatible /v1/chat/completions endpoints."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_new_tokens: int = 160,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _build_prompt(self, task: str, observation: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        trace_profile = (context or {}).get("trace_profile") or default_trace_profile("unknown", "generic", {})
        env_name = (context or {}).get("env_name", "unknown")
        env_type = (context or {}).get("env_type", "generic")
        step_id = int((context or {}).get("step_id", 0))
        max_steps = int((context or {}).get("max_steps", 1))
        system = build_system_prompt(trace_profile)
        user = (
            f"ENV_NAME: {env_name}\n"
            f"ENV_TYPE: {env_type}\n"
            f"STEP: {step_id + 1}/{max_steps}\n\n"
            f"TASK:\n{task}\n\nOBSERVATION:\n{observation}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _extract_thought(self, text: str) -> Optional[str]:
        match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_diary(self, text: str) -> Optional[str]:
        match = re.search(r'<diary>(.*?)</diary>', text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_json(self, text: str) -> dict:
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<diary>.*?</diary>', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()
        match = re.search(r'\{.*\}', clean_text if clean_text else text, flags=re.DOTALL)
        if not match:
            return {
                'action': 'inspect_and_continue',
                'short_justification': 'Model did not return valid JSON; using safe fallback.',
                'confidence': 0.2,
                'action_type': 'fallback',
                'reasoning_trace': [],
                'reasoning_summary': 'No valid JSON object returned.',
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                'action': 'inspect_and_continue',
                'short_justification': 'JSON parse failed; using safe fallback.',
                'confidence': 0.2,
                'action_type': 'fallback',
                'reasoning_trace': [],
                'reasoning_summary': 'JSON parsing failed.',
            }

    def act(self, task: str, observation: str, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        trace_profile = (context or {}).get("trace_profile") or default_trace_profile("unknown", "generic", {})
        payload = {
            "model": self.model_name,
            "messages": self._build_prompt(task, observation, context=context),
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        text = raw["choices"][0]["message"]["content"]
        parsed = self._extract_json(text)
        thought = self._extract_thought(text)
        diary = self._extract_diary(text)
        return ModelResponse(
            action=str(parsed.get("action", "inspect_and_continue")),
            short_justification=str(parsed.get("short_justification", "No justification provided.")),
            confidence=float(parsed.get("confidence", 0.5)),
            action_type=str(parsed.get("action_type", "generic")),
            action_args=parsed.get("action_args") if isinstance(parsed.get("action_args"), dict) else None,
            tool_calls=parsed.get("tool_calls") if isinstance(parsed.get("tool_calls"), list) else [],
            raw_text=text,
            thought=thought,
            diary=diary,
            reasoning_trace=_fallback_trace_from_text(parsed, thought, trace_profile),
            reasoning_summary=str(parsed.get("reasoning_summary", "")).strip() or None,
            trace_contract_version=trace_profile.get("contract_version", TRACE_CONTRACT_VERSION),
            trace_mode=trace_profile.get("mode", "stepwise"),
            usage=raw.get("usage") if isinstance(raw.get("usage"), dict) else None,
        )


def _fallback_trace_from_text(parsed: dict, thought: Optional[str], trace_profile: Dict[str, Any]) -> List[Dict[str, str]]:
    trace = normalize_reasoning_trace(parsed.get("reasoning_trace"), trace_profile.get("max_trace_steps", 4))
    if trace:
        return trace
    if thought:
        return normalize_reasoning_trace(thought, trace_profile.get("max_trace_steps", 4))
    return []


def _extract_api_key(api_key: Optional[str] = None, api_key_path: Optional[str] = None) -> str:
    if api_key:
        return api_key.strip()
    if api_key_path:
        text = Path(api_key_path).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"API key file is empty: {api_key_path}")
        match = re.search(r'OPENAI_API_KEY["\\\']?\s*=\s*["\\\']([^"\\\']+)["\\\']', text)
        if match:
            return match.group(1).strip()
        if text.startswith("sk-"):
            return text
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or provide api_key_path.")


class OpenAIAPIClient:
    """Client for the OpenAI Responses API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_key_path: Optional[str] = None,
        max_output_tokens: int = 512,
        reasoning_effort: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = _extract_api_key(api_key=api_key, api_key_path=api_key_path)
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.base_url = "https://api.openai.com/v1/responses"

    def _build_input(self, task: str, observation: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        trace_profile = (context or {}).get("trace_profile") or default_trace_profile("unknown", "generic", {})
        env_name = (context or {}).get("env_name", "unknown")
        env_type = (context or {}).get("env_type", "generic")
        step_id = int((context or {}).get("step_id", 0))
        max_steps = int((context or {}).get("max_steps", 1))
        system = build_system_prompt(trace_profile)
        user = (
            f"ENV_NAME: {env_name}\n"
            f"ENV_TYPE: {env_type}\n"
            f"STEP: {step_id + 1}/{max_steps}\n\n"
            f"TASK:\n{task}\n\n"
            f"OBSERVATION:\n{observation}"
        )
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
            },
        ]

    def _extract_output_text(self, raw: Dict[str, Any]) -> str:
        output_parts: List[str] = []
        for item in raw.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    output_parts.append(content.get("text", ""))
        return "".join(output_parts).strip()

    def act(self, task: str, observation: str, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        trace_profile = (context or {}).get("trace_profile") or default_trace_profile("unknown", "generic", {})
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": self._build_input(task, observation, context=context),
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        text = self._extract_output_text(raw)
        parsed = OpenAICompatibleClient._extract_json(self, text)
        thought = None
        diary = None
        return ModelResponse(
            action=str(parsed.get("action", "inspect_and_continue")),
            short_justification=str(parsed.get("short_justification", "No justification provided.")),
            confidence=float(parsed.get("confidence", 0.5)),
            action_type=str(parsed.get("action_type", "generic")),
            action_args=parsed.get("action_args") if isinstance(parsed.get("action_args"), dict) else None,
            tool_calls=parsed.get("tool_calls") if isinstance(parsed.get("tool_calls"), list) else [],
            raw_text=text,
            thought=thought,
            diary=diary,
            reasoning_trace=_fallback_trace_from_text(parsed, thought, trace_profile),
            reasoning_summary=str(parsed.get("reasoning_summary", "")).strip() or None,
            trace_contract_version=trace_profile.get("contract_version", TRACE_CONTRACT_VERSION),
            trace_mode=trace_profile.get("mode", "stepwise"),
            usage=raw.get("usage") if isinstance(raw.get("usage"), dict) else None,
        )
