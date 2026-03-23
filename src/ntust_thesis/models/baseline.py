"""Baseline direct generation model."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ntust_thesis.core.config_models import BaselineModelConfig
from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.types import (
    EventOutput,
    Prediction,
    PredictionMetadata,
    Sample,
)
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.prompts import build_baseline_event_extraction_prompt
from ntust_thesis.utils.env import get_required_env
from ntust_thesis.utils.json_parser import parse_json_object


class BaselineModel(Model):
    """Baseline model using Gemini direct generation."""

    def __init__(self, config: BaselineModelConfig) -> None:
        """Initialize baseline model from config."""
        self._backend = config.backend
        self._temperature = config.temperature
        api_key_env = config.api_key_env
        dotenv_path = Path(config.dotenv_path)
        api_key = get_required_env(api_key_env, fallback_paths=[dotenv_path])
        model_name = config.llm_name
        timeout = config.timeout_seconds
        self._llm = GeminiClient(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout,
        )

    def name(self) -> str:
        """Return model key."""
        return "baseline"

    def predict(self, sample: Sample) -> Prediction:
        """Generate text output then parse it as JSON."""
        raw_output = self._generate_gemini(
            sentence=sample.input_text,
            event_type=sample.metadata.event_type,
            legal_roles=sample.metadata.legal_roles,
        )
        parsed_output = _parse_event_output(raw_output)
        return Prediction(
            sample_id=sample.sample_id,
            raw_output=raw_output,
            parsed_output=parsed_output,
            metadata=PredictionMetadata(
                model=self.name(),
            ),
        )

    def _generate_gemini(
        self,
        sentence: str,
        event_type: str | None,
        legal_roles: list[str] | None,
    ) -> str:
        """Generate strict JSON with Gemini."""
        prompt = build_baseline_event_extraction_prompt(
            sentence=sentence,
            event_type=event_type,
            legal_roles=legal_roles,
        )
        return self._llm.generate(prompt, temperature=self._temperature)


def _parse_event_output(raw_output: str) -> EventOutput | None:
    """Parse raw model text into typed event output."""
    parsed = parse_json_object(raw_output)
    if parsed is None:
        return None
    try:
        return EventOutput.model_validate(parsed)
    except ValidationError:
        return None


def register() -> None:
    """Register built-in baseline model."""
    MODEL_REGISTRY.register("baseline", _build_baseline_model)


def _build_baseline_model(config: object) -> BaselineModel:
    """Build baseline model from boundary input."""
    typed_config = BaselineModelConfig.model_validate(config)
    return BaselineModel(config=typed_config)
