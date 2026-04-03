"""Direct IR baseline: one model generates IR, then compiler finalizes JSON."""

from __future__ import annotations

import json

from ntust_thesis.core.config_models import DirectIRBaselineModelConfig
from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.schemas import (
    EventOutput,
    Prediction,
    PredictionMetadata,
    Sample,
)
from ntust_thesis.models.components.ir_compiler import DeterministicIRCompiler
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.models.llm.vllm_client import VllmChatCompletionsClient
from ntust_thesis.prompts import build_direct_ir_prompt
from ntust_thesis.utils.env import (
    DEFAULT_DOTENV_PATH,
    get_env_float,
    get_env_int,
    get_env_int_list,
    get_required_env,
)


class DirectIRBaselineModel(Model):
    """Single-stage model that outputs IR directly."""

    def __init__(self, config: DirectIRBaselineModelConfig) -> None:
        """Initialize direct IR baseline model from config."""
        self._backend = config.backend
        dotenv_paths = [DEFAULT_DOTENV_PATH]
        self._temperature = get_env_float(
            "llm_temperature",
            default=0.0,
            fallback_paths=dotenv_paths,
        )
        timeout = get_env_int(
            "llm_timeout_seconds",
            default=120,
            fallback_paths=dotenv_paths,
        )
        sleep_seconds = get_env_float(
            "llm_sleep_seconds",
            default=1.0,
            fallback_paths=dotenv_paths,
        )
        max_retries = get_env_int(
            "llm_max_retries",
            default=5,
            fallback_paths=dotenv_paths,
        )
        backoff_initial_seconds = get_env_float(
            "llm_backoff_initial_seconds",
            default=2.0,
            fallback_paths=dotenv_paths,
        )
        backoff_multiplier = get_env_float(
            "llm_backoff_multiplier",
            default=2.0,
            fallback_paths=dotenv_paths,
        )
        backoff_max_seconds = get_env_float(
            "llm_backoff_max_seconds",
            default=32.0,
            fallback_paths=dotenv_paths,
        )
        retry_http_statuses = get_env_int_list(
            "llm_retry_http_statuses",
            default=[429, 500, 502, 503, 504],
            fallback_paths=dotenv_paths,
        )
        model_name = config.llm_name
        if config.backend == "gemini":
            api_key = get_required_env(config.api_key_env, fallback_paths=dotenv_paths)
            self._llm = GeminiClient(
                api_key=api_key,
                model_name=model_name,
                timeout_seconds=timeout,
                enable_sleep=config.enable_sleep,
                sleep_seconds=sleep_seconds,
                enable_retry=config.enable_retry,
                max_retries=max_retries,
                backoff_initial_seconds=backoff_initial_seconds,
                backoff_multiplier=backoff_multiplier,
                backoff_max_seconds=backoff_max_seconds,
                retry_http_statuses=tuple(retry_http_statuses),
            )
        elif config.backend == "vllm":
            api_base = config.api_base or "http://127.0.0.1:8000/v1"
            self._llm = VllmChatCompletionsClient(
                api_base=api_base,
                model_name=model_name,
                timeout_seconds=timeout,
                enable_sleep=config.enable_sleep,
                sleep_seconds=sleep_seconds,
                enable_retry=config.enable_retry,
                max_retries=max_retries,
                backoff_initial_seconds=backoff_initial_seconds,
                backoff_multiplier=backoff_multiplier,
                backoff_max_seconds=backoff_max_seconds,
                retry_http_statuses=tuple(retry_http_statuses),
            )
        else:
            msg = f"Unsupported backend: {config.backend}"
            raise ValueError(msg)
        self._compiler = DeterministicIRCompiler(ir_grammar=config.ir_grammar)

    def name(self) -> str:
        """Return model key."""
        return "direct_ir_baseline"

    def predict(self, sample: Sample) -> Prediction:
        """Generate IR directly and compile to final JSON output."""
        system_prompt, user_prompt = build_direct_ir_prompt(
            sentence=sample.raw_sentence,
            event_type=sample.metadata.event_type,
            candidate_roles=sample.metadata.candidate_roles,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        ir_text = self._llm.generate(
            system_prompt,
            user_prompt,
            self._temperature,
            allow_empty=True,
        )

        error_message: str | None = None
        compiled: EventOutput | None = None
        try:
            compiled = self._compiler.compile(
                ir_text=ir_text,
                event_type=sample.metadata.event_type or "unknown.event",
            )
        except Exception as exc:
            error_message = str(exc)

        if compiled is None:
            raw_output = ir_text
            parsed_output = None
        else:
            raw_output = json.dumps(compiled.model_dump(), ensure_ascii=False)
            parsed_output = compiled

        return Prediction(
            sample_id=sample.sample_id,
            raw_output=raw_output,
            parsed_output=parsed_output,
            metadata=PredictionMetadata(
                model=self.name(),
                backend=self._backend,
                ir_text=ir_text,
                compile_error=error_message,
                model_input={
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
            ),
        )


def register() -> None:
    """Register built-in direct IR baseline model."""
    MODEL_REGISTRY.register("direct_ir_baseline", _build_direct_ir_baseline_model)


def _build_direct_ir_baseline_model(config: object) -> DirectIRBaselineModel:
    """Build direct IR baseline model from boundary input."""
    typed_config = DirectIRBaselineModelConfig.model_validate(config)
    return DirectIRBaselineModel(config=typed_config)
