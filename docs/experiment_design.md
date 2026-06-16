# Experiment Design Overview: IR Grammar Comparison

## Research Goal

This thesis studies how different intermediate representation (IR) grammars affect structured event-argument generation.

The main controlled setting is a two-stage pipeline:

1. Semantic extraction
2. IR generation
3. Deterministic IR compilation into the canonical prediction format

The goal is to separate semantic understanding from structural generation and compare IR grammars under controlled conditions.

---

## Main Results

### M1: IR Grammar Comparison

**Purpose:** Compare how different IR grammars affect generation quality under the same two-stage pipeline, and identify the best-performing IR grammar.

**Setting:**

- Pipeline: two-stage
- Extraction model: `gemini-3.1-pro-preview` fixed
- IR models:
  - `gemini-3.1-pro-preview`
  - `gemini-2.5-flash`
  - `Mistral-7B-Instruct-v0.3`
- IR grammars:
  - JSON
  - Incremental-Assignment
  - CODE4STRUCT

---

## Ablation Studies

### A1: Pipeline Comparison

**Purpose:** Test whether the two-stage pipeline outperforms the one-stage pipeline.

**Comparisons:**

1. One-stage JSON vs. two-stage JSON
2. One-stage best IR vs. two-stage best IR

**Setting:**

- Models: Gemini 3.1 Pro, Gemini 2.5 Flash, Mistral
- For the two-stage A1 setting, the extraction model and IR model are the same model.

### A2: JSON vs. Best IR

**Purpose:** Test whether the best-performing IR grammar from M1 outperforms JSON under the same pipeline type.

**Comparisons:**

1. One-stage JSON vs. one-stage best IR
2. Two-stage JSON vs. two-stage best IR

**Setting:**

- Models: Gemini 3.1 Pro, Gemini 2.5 Flash, Mistral

### A3: ICL vs. Non-ICL

**Purpose:** Test whether in-context learning (ICL) examples improve IR generation.

**Setting:**

- Pipeline: two-stage
- IR grammars:
  - JSON
  - Incremental-Assignment
  - CODE4STRUCT
- Models:
  - Gemini 3.1 Pro
  - Gemini 2.5 Flash
  - Mistral
- Comparison: 0-shot vs. 4-shot

---

## ICL Example Design

The ICL examples are built from four selected samples in the RAMS train split. The `nw_...` strings below are the RAMS `doc_key` values of the source samples.

The same four semantic cases are reused across the three IR grammars. For one-stage prompts, the original sentence is replaced with a simple synthetic sentence such as `role is span`; this is intended to teach the IR construction format without teaching span extraction from natural language.

| Example | Scenario | RAMS train `doc_key` |
|---|---|---|
| 1 | Every gold role has at most one span. | `nw_RC10bdc622f21c5c3aeb343898b9ed2d31fefd2e8e5db923badcfdc9fb` |
| 2 | The ontology allows some roles to have multiple spans, but all gold roles in the sample have only one span. | `nw_RC4d768901bfdeb4296e42dda04d3d8ddc0872462814c789188ae8e591` |
| 3 | The ontology allows some roles to have multiple spans, and the gold annotation also contains multiple spans for a role. | `nw_RC8bfda5484f6491dad438110794bc990f02525d6d9a4751a203f32acb` |
| 4 | The sample contains no valid role-span pair for the allowed roles. | `nw_RC2adbdc1cf8523629763aacf509f3ba13ed44eda47ccf1404c710691e` |
