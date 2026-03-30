"""Pydantic models for experiment configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["rams"]
    data_dir: str = "datasets/RAMS/data"
    split: Literal["train", "dev", "test"] = "train"
    max_samples: int | None = None


class StageModelConfig(BaseModel):
    """Stage-level LLM backend configuration."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["gemini", "vllm"] = "gemini"
    llm_name: str = "gemini-2.5-flash-lite"
    api_key_env: str = "GEMINI_API_KEY"
    api_base: str | None = None
    enable_sleep: bool = False
    enable_retry: bool = True


class BaselineModelConfig(BaseModel):
    """Baseline model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["baseline"]
    backend: Literal["gemini", "vllm"] = "gemini"
    llm_name: str = "gemini-2.5-flash-lite"
    api_key_env: str = "GEMINI_API_KEY"
    api_base: str | None = None
    enable_sleep: bool = False
    enable_retry: bool = True


class DirectIRBaselineModelConfig(BaseModel):
    """Direct IR baseline model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["direct_ir_baseline"]
    backend: Literal["gemini", "vllm"] = "gemini"
    llm_name: str = "gemini-2.5-flash-lite"
    api_key_env: str = "GEMINI_API_KEY"
    api_base: str | None = None
    enable_sleep: bool = False
    enable_retry: bool = True


class IRPipelineModelConfig(BaseModel):
    """IR pipeline model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["ir_pipeline"]
    extraction_model: StageModelConfig = Field(default_factory=StageModelConfig)
    ir_model: StageModelConfig = Field(default_factory=StageModelConfig)


class TwoStageBaselineModelConfig(BaseModel):
    """Two-stage baseline: extraction text -> final JSON."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["two_stage_baseline"]
    extraction_model: StageModelConfig = Field(default_factory=StageModelConfig)
    schema_model: StageModelConfig = Field(default_factory=StageModelConfig)


ModelConfig = Annotated[
    BaselineModelConfig
    | DirectIRBaselineModelConfig
    | IRPipelineModelConfig
    | TwoStageBaselineModelConfig,
    Field(discriminator="name"),
]


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default_factory=lambda: [
            "is_valid_json",
            "schema_exact_match",
            "exact_match",
            "arg_i_f1",
            "arg_c_f1",
            "content_similarity_sbert",
            "bemeae",
            "json_structure_f1",
        ]
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "unnamed_experiment"
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
