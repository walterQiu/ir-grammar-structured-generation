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
from ntust_thesis.utils.env import get_env_with_default, get_required_env


class IRPipelineModel(Model):
    """IR pipeline with pluggable extraction and IR generation backends."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize IR pipeline with model config."""
        cfg = config or {}
        self._backend = str(cfg.get("backend", "mock"))
        self._temperature = float(cfg.get("temperature", 0.0))
        self._extractor, self._ir_generator = self._build_stages(cfg)
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
                "backend": self._backend,
                "extraction_text": extraction_text,
                "ir_text": ir_text,
                "compile_error": error_message,
            },
        )

    def _build_stages(self, cfg: dict[str, Any]) -> tuple[Extractor, IRGenerator]:
        """Create extractor and IR generator according to backend config."""
        if self._backend == "mock":
            return MockExtractor(), MockIRGenerator()

        if self._backend != "gemini":
            msg = f"Unsupported IR backend: {self._backend}"
            raise ValueError(msg)

        api_key_env = str(cfg.get("api_key_env", "GEMINI_API_KEY"))
        model_env = str(cfg.get("llm_name_env", "GEMINI_MODEL"))
        dotenv_path = Path(str(cfg.get("dotenv_path", "dotenv/.env")))
        api_key = get_required_env(api_key_env, fallback_paths=[dotenv_path])
        model_name = str(
            cfg.get("llm_name")
            or get_env_with_default(
                model_env,
                default="gemini-2.5-flash-lite",
                fallback_paths=[dotenv_path],
            )
        )
        timeout = int(cfg.get("timeout_seconds", 60))
        llm = GeminiClient(
            api_key=api_key, model_name=model_name, timeout_seconds=timeout
        )
        return (
            GeminiExtractor(llm=llm, temperature=self._temperature),
            GeminiIRGenerator(llm=llm, temperature=self._temperature),
        )


def register() -> None:
    """Register built-in IR pipeline model."""
    MODEL_REGISTRY.register("ir_pipeline", IRPipelineModel)
