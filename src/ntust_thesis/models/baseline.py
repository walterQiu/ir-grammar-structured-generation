"""Baseline direct generation model."""

from __future__ import annotations

import json

from ntust_thesis.core.config_models import BaselineModelConfig
from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.schemas import Prediction, PredictionMetadata, Sample
from ntust_thesis.models.components.event_output_parser import (
    parse_event_output_from_arguments_json,
)
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.models.llm.vllm_client import VllmChatCompletionsClient
from ntust_thesis.prompts import build_baseline_event_extraction_prompt
from ntust_thesis.utils.env import (
    DEFAULT_DOTENV_PATH,
    get_env_float,
    get_env_int,
    get_env_int_list,
    get_required_env,
)


class BaselineModel(Model):
    """Baseline model using pluggable LLM backends."""

    def __init__(self, config: BaselineModelConfig) -> None:
        """Initialize baseline model from config."""
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
            api_key_env = config.api_key_env
            api_key = get_required_env(api_key_env, fallback_paths=dotenv_paths)
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

    def name(self) -> str:
        """Return model key."""
        return "baseline"

    def predict(self, sample: Sample) -> Prediction:
        """Generate direct JSON output and parse into typed event output."""
        prompt = build_baseline_event_extraction_prompt(
            sentence=sample.raw_sentence,
            event_type=sample.metadata.event_type,
            candidate_roles=sample.metadata.candidate_roles,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        raw_model_text = self._llm.generate(prompt, temperature=self._temperature)
        parsed_output = parse_event_output_from_arguments_json(
            raw_output=raw_model_text,
            event_type=sample.metadata.event_type or "unknown.event",
        )

        if parsed_output is None:
            raw_output = raw_model_text
        else:
            raw_output = json.dumps(parsed_output.model_dump(), ensure_ascii=False)

        return Prediction(
            sample_id=sample.sample_id,
            raw_output=raw_output,
            parsed_output=parsed_output,
            metadata=PredictionMetadata(
                model=self.name(),
                backend=self._backend,
                model_input=prompt,
            ),
        )


def register() -> None:
    """Register built-in baseline model."""
    MODEL_REGISTRY.register("baseline", _build_baseline_model)


def _build_baseline_model(config: object) -> BaselineModel:
    """Build baseline model from boundary input."""
    typed_config = BaselineModelConfig.model_validate(config)
    return BaselineModel(config=typed_config)
