"""IR pipeline model with extraction -> IR -> deterministic compiler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.types import Prediction, Sample
from ntust_thesis.models.components.extractor import (
    Extractor,
    GeminiExtractor,
    MockExtractor,
)
from ntust_thesis.models.components.ir_compiler import DeterministicIRCompiler
from ntust_thesis.models.components.ir_generator import (
    GeminiIRGenerator,
    IRGenerator,
    MockIRGenerator,
)
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.utils.env import get_required_env


class IRPipelineModel(Model):
    """IR pipeline with pluggable extraction and IR generation backends."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize IR pipeline with model config."""
        cfg = config or {}
        extraction_cfg = cfg.get("extraction_model", {})
        ir_cfg = cfg.get("ir_model", {})
        if not isinstance(extraction_cfg, dict):
            msg = "model.extraction_model must be a mapping."
            raise TypeError(msg)
        if not isinstance(ir_cfg, dict):
            msg = "model.ir_model must be a mapping."
            raise TypeError(msg)

        self._extraction_backend = str(extraction_cfg.get("backend", "mock"))
        self._ir_backend = str(ir_cfg.get("backend", "mock"))
        self._extractor = self._build_extractor(extraction_cfg)
        self._ir_generator = self._build_ir_generator(ir_cfg)
        self._compiler = DeterministicIRCompiler()

    def name(self) -> str:
        """Return model key."""
        return "ir_pipeline"

    def predict(self, sample: Sample) -> Prediction:
        """Run extraction -> IR -> compile and return final prediction."""
        extraction_text = self._extractor.extract(sample.input_text)
        ir_text = self._ir_generator.generate(extraction_text)

        error_message: str | None = None
        compiled: dict[str, Any] | None = None
        try:
            compiled = self._compiler.compile(ir_text, sample.schema)
        except Exception as exc:
            error_message = str(exc)

        if compiled is None:
            raw_output = ir_text
            parsed_output = None
        else:
            raw_output = json.dumps(compiled, ensure_ascii=False)
            parsed_output = compiled

        return Prediction(
            sample_id=sample.sample_id,
            raw_output=raw_output,
            parsed_output=parsed_output,
            metadata={
                "model": self.name(),
                "extraction_backend": self._extraction_backend,
                "ir_backend": self._ir_backend,
                "extraction_text": extraction_text,
                "ir_text": ir_text,
                "compile_error": error_message,
            },
        )

    def _build_extractor(self, cfg: dict[str, Any]) -> Extractor:
        """Create extraction stage from extraction model config."""
        backend = str(cfg.get("backend", "mock"))
        if backend == "mock":
            return MockExtractor()
        if backend == "gemini":
            temperature = float(cfg.get("temperature", 0.0))
            llm = self._build_gemini_client(cfg)
            return GeminiExtractor(llm=llm, temperature=temperature)
        msg = f"Unsupported extraction backend: {backend}"
        raise ValueError(msg)

    def _build_ir_generator(self, cfg: dict[str, Any]) -> IRGenerator:
        """Create IR generation stage from ir model config."""
        backend = str(cfg.get("backend", "mock"))
        if backend == "mock":
            return MockIRGenerator()
        if backend == "gemini":
            temperature = float(cfg.get("temperature", 0.0))
            llm = self._build_gemini_client(cfg)
            return GeminiIRGenerator(llm=llm, temperature=temperature)
        msg = f"Unsupported IR backend: {backend}"
        raise ValueError(msg)

    @staticmethod
    def _build_gemini_client(cfg: dict[str, Any]) -> GeminiClient:
        """Create Gemini client from stage-specific config."""
        api_key_env = str(cfg.get("api_key_env", "GEMINI_API_KEY"))
        dotenv_path = Path(str(cfg.get("dotenv_path", "dotenv/.env")))
        api_key = get_required_env(api_key_env, fallback_paths=[dotenv_path])
        model_name = str(cfg.get("llm_name", "gemini-2.5-flash-lite"))
        timeout = int(cfg.get("timeout_seconds", 60))
        return GeminiClient(
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout,
        )


def register() -> None:
    """Register built-in IR pipeline model."""
    MODEL_REGISTRY.register("ir_pipeline", IRPipelineModel)
