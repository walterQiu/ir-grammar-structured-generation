# Strict Structured State Generation Benchmark

## Overview
This project aims to evaluate and improve **strict structured state generation** using LLMs.

Unlike traditional tasks (e.g., IE/EAE) that allow partial correctness, this project requires:
- Exact structural correctness
- Schema adherence
- Deterministic validation

The goal is to test whether LLM outputs can be reliably used as **machine-consumable state**.

---

## Core Idea
Given:
- Input text / instruction
- Predefined schema

The model must generate a structured output (JSON) that:
1. Matches the schema exactly
2. Matches the gold annotation exactly (strict match)

---

## Supported Tasks / Datasets
We will support multiple datasets under a unified interface:

### 1. EAE (Event Argument Extraction)
- Input: text
- Output: event type + arguments
- Dataset examples: RAMS

### 2. Tool Calling
- Input: instruction
- Output: function name + arguments

### 3. DST (Dialogue State Tracking)
- Input: dialogue context
- Output: slot-value pairs

---

## Baseline

### Direct Generation
- Input → LLM → JSON output
- No intermediate representation

File: `models/baseline.py`

---

## IR Pipeline (Proposed Method)

### Step 1: Extraction LM
- Output: unstructured or semi-structured text

### Step 2: IR LM
- Convert to intermediate representation (IR)
- Format: incremental-assignment

Example:
```
event.type = Attack
event.arguments.agent = rebels
event.arguments.location = Baghdad
```

### Step 3: Deterministic Compiler
- Convert IR → JSON
- Ensure schema correctness

File: `models/ir_pipeline.py`

---

## Evaluation

### 1. JSON Validity
- Check if output is valid JSON

### 2. Schema Validation
- Validate using jsonschema or custom checker

### 3. Exact Match (Main Metric)
- Output must exactly match gold JSON
- Order of keys does NOT matter

---

## Validator Implementation

File: `evaluation/validator.py`

Functions:

```python
def is_valid_json(output_str) -> bool:
    pass

def validate_schema(json_obj, schema) -> bool:
    pass

def exact_match(pred, gold) -> bool:
    pass
```

---

## Metrics

File: `evaluation/metrics.py`

```python
def compute_metrics(predictions, golds):
    return {
        "json_valid_rate": ...,
        "schema_valid_rate": ...,
        "exact_match_rate": ...
    }
```

---

## Running Experiments

File: `run.py`

Steps:
1. Load dataset
2. Load schema
3. Run model (baseline or IR)
4. Validate outputs
5. Compute metrics

---

## Key Requirement

- ANY deviation from schema or gold = incorrect
- No partial credit
- Strict evaluation only

---

## Future Extensions (Optional)

- Add constraint checking (ontology rules)
- Add multi-step workflow simulation
- Add error analysis (type of failure)

---

## Summary

This project focuses on:
- Strict structured generation
- Cross-task evaluation (EAE, tool, DST)
- Comparing baseline vs IR pipeline
