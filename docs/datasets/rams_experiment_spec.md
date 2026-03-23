# RAMS Experiment Specification (Draft v1)

## Goal
Define a strict and reproducible RAMS-based Event Argument Extraction setup for this thesis.

## Model Input Contract
For each RAMS sample, the model should receive:
1. `sentence` (full document text flattened from RAMS `sentences`)
2. `evt_trigger` highlighted in text
   - Trigger span is wrapped with double asterisks like markdown bold
   - Prompt instruction example: `The trigger word(s) of the event is marked with **trigger word**.`
3. `event_type`
   - Taken from RAMS `evt_triggers` (e.g., `life.die.deathcausedbyviolentevents`)
4. `legal role candidates (ontology)`
   - Role candidates are constrained by `event_type`
   - Source ontology: `datasets/RAMS/scorer/event_role_multiplicities.txt`

## Extraction Model Responsibility
Given input above, the extraction model must:
1. For each provided valid role, determine whether it is present in the sentence.
2. If present, predict its token span.
3. If not present, output `None` for that role.

### Example Output Shape (Extraction Stage)
```json
{
  "evt090arg01killer": [85, 88],
  "evt090arg02victim": [90, 91],
  "evt090arg03instrument": None
}
```

## Ontology File Placement Decision
Recommended policy:
Keep the original RAMS ontology file in its current upstream location:
   - `datasets/RAMS/scorer/event_role_multiplicities.txt`

## Notes for Next Implementation Step
1. Build a parser for `event_role_multiplicities.txt` into `{event_type -> allowed_roles}` map.
2. Add trigger-highlighting utility using RAMS trigger span.
3. Update prompts for extraction stage to include ontology constraints.
4. Decide role-output policy:
   - Output all legal roles with `None` when missing, or
   - Output only observed roles.
