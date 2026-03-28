"""IR pipeline model with extraction -> IR -> deterministic compiler."""

from __future__ import annotations

import json
from pathlib import Path

from ntust_thesis.core.config_models import IRPipelineModelConfig, StageModelConfig
from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.schemas import (
    EventOutput,
    Prediction,
    PredictionMetadata,
    Sample,
)
from ntust_thesis.models.components.extractor import (
    Extractor,
    GeminiExtractor,
)
from ntust_thesis.models.components.ir_compiler import DeterministicIRCompiler
from ntust_thesis.models.components.ir_generator import (
    GeminiIRGenerator,
    IRGenerator,
)
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.prompts import build_ir_extraction_prompt, build_ir_generation_prompt
from ntust_thesis.utils.env import get_required_env


class IRPipelineModel(Model):
    """IR pipeline with pluggable extraction and IR generation backends."""

    def __init__(self, config: IRPipelineModelConfig) -> None:
        """Initialize IR pipeline with model config."""
        extraction_cfg = config.extraction_model
        ir_cfg = config.ir_model
        self._extraction_backend = extraction_cfg.backend
        self._ir_backend = ir_cfg.backend
        self._extractor = self._build_extractor(extraction_cfg)
        self._ir_generator = self._build_ir_generator(ir_cfg)
        self._compiler = DeterministicIRCompiler()

    def name(self) -> str:
        """Return model key."""
        return "ir_pipeline"

    def predict(self, sample: Sample) -> Prediction:
        """Run extraction -> IR -> compile and return final prediction."""
        extraction_prompt = build_ir_extraction_prompt(
            sentence=sample.raw_sentence,
            event_type=sample.metadata.event_type,
            legal_roles=sample.metadata.legal_roles,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        extraction_text = self._extractor.extract(
            sentence=sample.raw_sentence,
            event_type=sample.metadata.event_type,
            legal_roles=sample.metadata.legal_roles,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        ir_prompt = build_ir_generation_prompt(
            extraction_text=extraction_text,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        ir_text = self._ir_generator.generate(
            extraction_text=extraction_text,
            role_multiplicities=sample.metadata.role_multiplicities,
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
                extraction_backend=self._extraction_backend,
                ir_backend=self._ir_backend,
                extraction_text=extraction_text,
                ir_text=ir_text,
                compile_error=error_message,
                model_input={
                    "extraction_prompt": extraction_prompt,
                    "ir_prompt": ir_prompt,
                },
            ),
        )

    def _build_extractor(self, cfg: StageModelConfig) -> Extractor:
        """Create extraction stage from extraction model config."""
        backend = cfg.backend
        if backend == "gemini":
            temperature = cfg.temperature
            llm = self._build_gemini_client(cfg)
            return GeminiExtractor(llm=llm, temperature=temperature)
        msg = f"Unsupported extraction backend: {backend}"
        raise ValueError(msg)

    def _build_ir_generator(self, cfg: StageModelConfig) -> IRGenerator:
        """Create IR generation stage from ir model config."""
        backend = cfg.backend
        if backend == "gemini":
            temperature = cfg.temperature
            llm = self._build_gemini_client(cfg)
            return GeminiIRGenerator(llm=llm, temperature=temperature)
        msg = f"Unsupported IR backend: {backend}"
        raise ValueError(msg)

    @staticmethod
    def _build_gemini_client(cfg: StageModelConfig) -> GeminiClient:
        """Create Gemini client from stage-specific config."""
        api_key_env = cfg.api_key_env
        dotenv_path = Path(cfg.dotenv_path)
        api_key = get_required_env(api_key_env, fallback_paths=[dotenv_path])
        model_name = cfg.llm_name
        timeout = cfg.timeout_seconds
        return GeminiClient(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout,
        )


def register() -> None:
    """Register built-in IR pipeline model."""
    MODEL_REGISTRY.register("ir_pipeline", _build_ir_pipeline_model)


def _build_ir_pipeline_model(config: object) -> IRPipelineModel:
    """Build IR pipeline model from boundary input."""
    typed_config = IRPipelineModelConfig.model_validate(config)
    return IRPipelineModel(config=typed_config)
