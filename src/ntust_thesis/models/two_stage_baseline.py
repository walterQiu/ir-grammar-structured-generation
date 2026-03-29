"""Two-stage baseline: extraction text -> final JSON."""

from __future__ import annotations

import json

from ntust_thesis.core.config_models import (
    StageModelConfig,
    TwoStageBaselineModelConfig,
)
from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.schemas import Prediction, PredictionMetadata, Sample
from ntust_thesis.models.components.event_output_parser import (
    parse_event_output_from_arguments_json,
)
from ntust_thesis.models.components.extractor import Extractor, GeminiExtractor
from ntust_thesis.models.components.schema_generator import (
    LLMSchemaGenerator,
    SchemaGenerator,
)
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.models.llm.vllm_client import VllmChatCompletionsClient
from ntust_thesis.prompts import (
    build_ir_extraction_prompt,
    build_schema_generation_prompt,
)
from ntust_thesis.utils.env import (
    DEFAULT_DOTENV_PATH,
    get_env_float,
    get_env_int,
    get_env_int_list,
    get_required_env,
)


class TwoStageBaselineModel(Model):
    """Two-stage baseline with no deterministic IR compiler."""

    def __init__(self, config: TwoStageBaselineModelConfig) -> None:
        """Initialize model from config."""
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
        schema_cfg = config.schema_model
        self._extraction_backend = extraction_cfg.backend
        self._schema_backend = schema_cfg.backend
        self._extractor = self._build_extractor(extraction_cfg)
        self._schema_generator = self._build_schema_generator(schema_cfg)

    def name(self) -> str:
        """Return model key."""
        return "two_stage_baseline"

    def predict(self, sample: Sample) -> Prediction:
        """Run extraction -> schema generation and parse final JSON output."""
        extraction_prompt = build_ir_extraction_prompt(
            sentence=sample.raw_sentence,
            event_type=sample.metadata.event_type,
            candidate_roles=sample.metadata.candidate_roles,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        extraction_text = self._extractor.extract(
            sentence=sample.raw_sentence,
            event_type=sample.metadata.event_type,
            candidate_roles=sample.metadata.candidate_roles,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        schema_prompt = build_schema_generation_prompt(
            extraction_text=extraction_text,
            role_multiplicities=sample.metadata.role_multiplicities,
        )
        schema_text = self._schema_generator.generate(
            extraction_text=extraction_text,
            role_multiplicities=sample.metadata.role_multiplicities,
        )

        parsed_output = parse_event_output_from_arguments_json(
            raw_output=schema_text,
            event_type=sample.metadata.event_type or "unknown.event",
        )

        if parsed_output is None:
            raw_output = schema_text
        else:
            raw_output = json.dumps(parsed_output.model_dump(), ensure_ascii=False)

        return Prediction(
            sample_id=sample.sample_id,
            raw_output=raw_output,
            parsed_output=parsed_output,
            metadata=PredictionMetadata(
                model=self.name(),
                extraction_backend=self._extraction_backend,
                extraction_text=extraction_text,
                model_input={
                    "extraction_prompt": extraction_prompt,
                    "schema_prompt": schema_prompt,
                },
                # schema_backend=self._schema_backend,
                # schema_text=schema_text,
            ),
        )

    def _build_extractor(self, cfg: StageModelConfig) -> Extractor:
        """Create extraction stage."""
        llm = self._build_llm_client(cfg)
        return GeminiExtractor(llm=llm, temperature=self._llm_temperature)

    def _build_schema_generator(self, cfg: StageModelConfig) -> SchemaGenerator:
        """Create schema-generation stage."""
        llm = self._build_llm_client(cfg)
        return LLMSchemaGenerator(llm=llm, temperature=self._llm_temperature)

    def _build_llm_client(
        self, cfg: StageModelConfig
    ) -> GeminiClient | VllmChatCompletionsClient:
        """Create backend-specific LLM client from stage config."""
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
            api_key = get_required_env(
                cfg.api_key_env,
                fallback_paths=[DEFAULT_DOTENV_PATH],
            )
            return GeminiClient(
                api_key=api_key,
                model_name=cfg.llm_name,
                **common_kwargs,
            )
        if cfg.backend == "vllm":
            api_base = cfg.api_base or "http://127.0.0.1:8000/v1"
            return VllmChatCompletionsClient(
                api_base=api_base,
                model_name=cfg.llm_name,
                **common_kwargs,
            )
        msg = f"Unsupported backend: {cfg.backend}"
        raise ValueError(msg)


def register() -> None:
    """Register built-in two-stage baseline model."""
    MODEL_REGISTRY.register("two_stage_baseline", _build_two_stage_baseline_model)


def _build_two_stage_baseline_model(config: object) -> TwoStageBaselineModel:
    """Build two-stage baseline model from boundary input."""
    typed_config = TwoStageBaselineModelConfig.model_validate(config)
    return TwoStageBaselineModel(config=typed_config)
