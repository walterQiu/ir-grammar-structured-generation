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


class OneStageModelConfig(BaseModel):
    """One-stage model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["one_stage"]
    stage_model: StageModelConfig = Field(default_factory=StageModelConfig)
    ir_grammar: str
    apply_icl: bool = True


class TwoStageModelConfig(BaseModel):
    """Two-stage model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["two_stage"]
    extraction_model: StageModelConfig = Field(default_factory=StageModelConfig)
    ir_model: StageModelConfig = Field(default_factory=StageModelConfig)
    ir_grammar: str
    apply_icl: bool = True


ModelConfig = Annotated[
    OneStageModelConfig | TwoStageModelConfig,
    Field(discriminator="name"),
]


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default_factory=lambda: [
            "is_valid_ir",
            "arg_i_f1",
            "arg_c_f1",
            "content_similarity_sbert",
            "bemeae",
            "ecar",
        ]
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "unnamed_experiment"
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
