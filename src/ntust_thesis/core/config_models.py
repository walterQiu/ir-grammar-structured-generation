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

    backend: Literal["mock", "gemini"] = "mock"
    llm_name: str = "gemini-2.5-flash-lite"
    api_key_env: str = "GEMINI_API_KEY"
    dotenv_path: str = "dotenv/.env"
    temperature: float = 0.0
    timeout_seconds: int = 60


class BaselineModelConfig(BaseModel):
    """Baseline model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["baseline"]
    backend: Literal["mock", "gemini"] = "mock"
    llm_name: str = "gemini-2.5-flash-lite"
    api_key_env: str = "GEMINI_API_KEY"
    dotenv_path: str = "dotenv/.env"
    temperature: float = 0.0
    timeout_seconds: int = 60


class IRPipelineModelConfig(BaseModel):
    """IR pipeline model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["ir_pipeline"]
    extraction_model: StageModelConfig = Field(default_factory=StageModelConfig)
    ir_model: StageModelConfig = Field(default_factory=StageModelConfig)


ModelConfig = Annotated[
    BaselineModelConfig | IRPipelineModelConfig,
    Field(discriminator="name"),
]


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    validators: list[str] = Field(default_factory=lambda: ["strict"])
    metrics: list[str] = Field(default_factory=lambda: ["strict_rates"])


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "unnamed_experiment"
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
