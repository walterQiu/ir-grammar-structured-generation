"""Baseline direct generation model."""

from __future__ import annotations

import json
from pathlib import Path

from ntust_thesis.core.config_models import BaselineModelConfig
from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.schemas import Prediction, PredictionMetadata, Sample
from ntust_thesis.models.components.event_output_parser import (
    parse_event_output_from_arguments_json,
)
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.prompts import build_baseline_event_extraction_prompt
from ntust_thesis.utils.env import get_required_env


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
        """Generate direct JSON output and parse into typed event output."""
        prompt = build_baseline_event_extraction_prompt(
            sentence=sample.raw_sentence,
            event_type=sample.metadata.event_type,
            legal_roles=sample.metadata.legal_roles,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        raw_model_text = self._llm.generate(prompt, temperature=self._temperature)
        source_sentence = sample.metadata.sentence_text or sample.raw_sentence
        parsed_output = parse_event_output_from_arguments_json(
            raw_output=raw_model_text,
            source_sentence=source_sentence,
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
