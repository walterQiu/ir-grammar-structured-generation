"""Prompt helpers."""

from ntust_thesis.prompts.model_prompts import (
    build_one_stage_code4struct_ir_prompt,
    build_one_stage_dot_notation_ir_prompt,
    build_one_stage_ir_prompt,
    build_one_stage_json_prompt,
    build_two_stage_extraction_prompt,
    build_two_stage_ir_prompt,
    build_two_stage_json_prompt,
)

__all__ = [
    "build_one_stage_code4struct_ir_prompt",
    "build_one_stage_dot_notation_ir_prompt",
    "build_one_stage_ir_prompt",
    "build_one_stage_json_prompt",
    "build_two_stage_extraction_prompt",
    "build_two_stage_ir_prompt",
    "build_two_stage_json_prompt",
]
