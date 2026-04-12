"""IR pipeline model with extraction -> IR -> deterministic compiler."""

from __future__ import annotations

import json

from ntust_thesis.core.config_models import StageModelConfig, TwoStageModelConfig
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
from ntust_thesis.models.llm.vllm_client import VllmChatCompletionsClient
from ntust_thesis.prompts import (
    build_two_stage_extraction_prompt,
    build_two_stage_ir_prompt,
)
from ntust_thesis.utils.env import (
    DEFAULT_DOTENV_PATH,
    get_env_float,
    get_env_int,
    get_env_int_list,
    get_required_env,
)


class TwoStageModel(Model):
    """Two-stage IR pipeline with extraction and IR generation backends."""

    def __init__(self, config: TwoStageModelConfig) -> None:
        """Initialize two-stage IR pipeline with model config."""
        dotenv_paths = [DEFAULT_DOTENV_PATH]
        self._llm_temperature = get_env_float(
            "llm_temperature",
            default=0.0,
            fallback_paths=dotenv_paths,
        )
        self._llm_timeout_seconds = get_env_int(
            "llm_timeout_seconds",
            default=120,
            fallback_paths=dotenv_paths,
        )
        self._llm_sleep_seconds = get_env_float(
            "llm_sleep_seconds",
            default=1.0,
            fallback_paths=dotenv_paths,
        )
        self._llm_max_retries = get_env_int(
            "llm_max_retries",
            default=5,
            fallback_paths=dotenv_paths,
        )
        self._llm_backoff_initial_seconds = get_env_float(
            "llm_backoff_initial_seconds",
            default=2.0,
            fallback_paths=dotenv_paths,
        )
        self._llm_backoff_multiplier = get_env_float(
            "llm_backoff_multiplier",
            default=2.0,
            fallback_paths=dotenv_paths,
        )
        self._llm_backoff_max_seconds = get_env_float(
            "llm_backoff_max_seconds",
            default=32.0,
            fallback_paths=dotenv_paths,
        )
        self._llm_retry_http_statuses = tuple(
            get_env_int_list(
                "llm_retry_http_statuses",
                default=[429, 500, 502, 503, 504],
                fallback_paths=dotenv_paths,
            )
        )

        extraction_cfg = config.extraction_model
        ir_cfg = config.ir_model
        self._ir_grammar = config.ir_grammar
        self._extraction_backend = extraction_cfg.backend
        self._ir_backend = ir_cfg.backend
        self._extractor = self._build_extractor(extraction_cfg)
        self._ir_generator = self._build_ir_generator(ir_cfg)
        self._compiler = DeterministicIRCompiler(ir_grammar=self._ir_grammar)

    def name(self) -> str:
        """Return model key."""
        return "two_stage"

    def predict(self, sample: Sample) -> Prediction:
        """Run extraction -> IR -> compile and return final prediction."""
        extraction_system_prompt, extraction_user_prompt = (
            build_two_stage_extraction_prompt(
                sentence=sample.raw_sentence,
                event_type=sample.metadata.event_type,
                role_multiplicities=sample.metadata.role_multiplicities,
            )
        )
        extraction_text = self._extractor.extract(
            sentence=sample.raw_sentence,
            event_type=sample.metadata.event_type,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        ir_system_prompt, ir_user_prompt = build_two_stage_ir_prompt(
            extraction_text=extraction_text,
            event_type=sample.metadata.event_type,
            role_multiplicities=sample.metadata.role_multiplicities,
            ir_grammar=self._ir_grammar,
        )
        ir_text = self._ir_generator.generate(
            extraction_text=extraction_text,
            event_type=sample.metadata.event_type,
            role_multiplicities=sample.metadata.role_multiplicities,
        )

        error_message: str | None = None
        compiled: EventOutput | None = None
        try:
            compiled = self._compiler.compile(
                ir_text=ir_text,
                event_type=sample.metadata.event_type,
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
                    "extraction_system_prompt": extraction_system_prompt,
                    "extraction_user_prompt": extraction_user_prompt,
                    "ir_system_prompt": ir_system_prompt,
                    "ir_user_prompt": ir_user_prompt,
                },
            ),
        )

    def _build_extractor(self, cfg: StageModelConfig) -> Extractor:
        """Create extraction stage from extraction model config."""
        backend = cfg.backend
        if backend in {"gemini", "vllm"}:
            temperature = self._llm_temperature
            llm = self._build_llm_client(cfg)
            return GeminiExtractor(llm=llm, temperature=temperature)
        msg = f"Unsupported extraction backend: {backend}"
        raise ValueError(msg)

    def _build_ir_generator(self, cfg: StageModelConfig) -> IRGenerator:
        """Create IR generation stage from ir model config."""
        backend = cfg.backend
        if backend in {"gemini", "vllm"}:
            temperature = self._llm_temperature
            llm = self._build_llm_client(cfg)
            return GeminiIRGenerator(
                llm=llm,
                temperature=temperature,
                ir_grammar=self._ir_grammar,
            )
        msg = f"Unsupported IR backend: {backend}"
        raise ValueError(msg)

    def _build_llm_client(
        self, cfg: StageModelConfig
    ) -> GeminiClient | VllmChatCompletionsClient:
        """Create backend-specific LLM client from stage config."""
        model_name = cfg.llm_name
        common_kwargs = {
            "timeout_seconds": self._llm_timeout_seconds,
            "enable_sleep": cfg.enable_sleep,
            "sleep_seconds": self._llm_sleep_seconds,
            "enable_retry": cfg.enable_retry,
            "max_retries": self._llm_max_retries,
            "backoff_initial_seconds": self._llm_backoff_initial_seconds,
            "backoff_multiplier": self._llm_backoff_multiplier,
            "backoff_max_seconds": self._llm_backoff_max_seconds,
            "retry_http_statuses": self._llm_retry_http_statuses,
        }
        if cfg.backend == "gemini":
            api_key_env = cfg.api_key_env
            api_key = get_required_env(
                api_key_env,
                fallback_paths=[DEFAULT_DOTENV_PATH],
            )
            return GeminiClient(
                api_key=api_key,
                model_name=model_name,
                **common_kwargs,
            )
        if cfg.backend == "vllm":
            api_base = cfg.api_base or "http://127.0.0.1:8000/v1"
            return VllmChatCompletionsClient(
                api_base=api_base,
                model_name=model_name,
                **common_kwargs,
            )
        msg = f"Unsupported backend: {cfg.backend}"
        raise ValueError(msg)


def register() -> None:
    """Register built-in two-stage pipeline model."""
    MODEL_REGISTRY.register("two_stage", _build_two_stage_model)


def _build_two_stage_model(config: object) -> TwoStageModel:
    """Build two-stage pipeline model from boundary input."""
    typed_config = TwoStageModelConfig.model_validate(config)
    return TwoStageModel(config=typed_config)
