# RAMS Dataset Usage Spec

## Scope
This document only describes how RAMS data is used in this project:
1. Which RAMS fields are read
2. How they are transformed into project sample fields
3. What input fields are provided to models
4. What output fields are stored from model predictions

## RAMS Source Fields Used
From each RAMS JSONL row, we use:
1. `doc_key`
2. `sentences`
3. `evt_triggers`
4. `gold_evt_links`

### Field Usage Details
1. `doc_key`
   - Used as project `sample_id`.
2. `sentences`
   - Flattened into one token sequence, then joined as plain text.
3. `evt_triggers`
   - Current implementation uses only the first trigger (`evt_triggers[0]`).
   - Trigger span is used to mark trigger text with `**...**`.
   - Trigger event type is used as `event_type`.
4. `gold_evt_links`
   - Used to build gold arguments.
   - Each argument keeps only:
     - `role` (normalized, with RAMS `evtXXXargYY` prefix removed)
     - `text` (surface span text from tokens)

## Ontology File Used
1. Path: `datasets/RAMS/scorer/event_role_multiplicities.txt`
2. Used to map `event_type -> {role: multiplicity}`
3. Provides:
   - `legal_roles`
   - `role_multiplicities`

## Project Sample Fields (After RAMS Conversion)
Each RAMS row is converted into:

```json
{
  "sample_id": "<doc_key>",
  "raw_sentence": "<sentence with trigger marked by ** **>",
  "gold": {
    "event_type": "<event_type>",
    "arguments": [
      {"role": "<role>", "text": "<surface text>"}
    ]
  },
  "metadata": {
    "sentence_text": "<flattened sentence>",
    "marked_sentence": "<trigger-marked sentence>",
    "event_type": "<event_type>",
    "legal_roles": ["<role1>", "<role2>"],
    "role_multiplicities": {"<role1>": 1, "<role2>": 2}
  }
}
```

## Model Input Fields
Models are given dataset-derived fields:
1. `raw_sentence` (trigger-marked sentence)
2. `metadata.event_type`
3. `metadata.legal_roles`
4. `metadata.role_multiplicities`

## Model Prediction Fields Stored
For each sample, prediction artifacts store:
1. `sample_id`
2. `raw_output` (model raw text or compiled JSON string)
3. `parsed_output`
   - Canonical structure:
   - `{"event_type": "...", "arguments": [{"role": "...", "text": "..."}]}`
4. `prediction_metadata`
   - Includes model/backend and prompt-related metadata

## Design Decision: No Span Offsets
This project does not keep token start/end offsets in model outputs or stored prediction artifacts.
Argument representation is text-based (`role` + `text`) only.
