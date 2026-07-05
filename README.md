# NTUST Thesis Experiments

Paper: [Intermediate Representation Design for Structured Generation: An Empirical Study of IR Grammars](https://example.com/paper-placeholder)

## Environment Setup

Run all commands from the repository root.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Project Dependencies

```bash
uv sync --dev
```

### 3. Install the pre-commit Hook

```bash
uv run pre-commit install
```

### 4. Configure Environment Variables

```bash
cp dotenv/.env.example dotenv/.env
```

Fill in the required values in `dotenv/.env`. `HF_TOKEN` is optional:

### 5. Add the RAMS Dataset

Download RAMS from the [official dataset page](https://nlp.jhu.edu/rams/), then
place the complete `RAMS/` directory under `datasets/`:

```text
datasets/RAMS/
|-- data/
|   |-- train.jsonlines
|   |-- dev.jsonlines
|   `-- test.jsonlines
`-- scorer/
    `-- event_role_multiplicities.txt
```

### 6. Run a Development Experiment

```bash
UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run python run.py \
  --config experimental_configs/development_used.yaml
```

Run artifacts are written under `outputs/runs/`.

## Inference

For full experiment runs:

1. Start the local Mistral server with vLLM.
   - The vLLM server must expose an OpenAI-compatible `/v1/chat/completions` endpoint.
   - Make sure the `api_base` values in the Mistral experiment YAML files point to the running server.
2. Select experiments in `experimental_configs/batch_run_list.yaml`.
   - The current list is configured to run every experiment reported in the thesis once.
   - Active YAML paths will be executed.
   - Comment out entries that should not be run.
3. Run the selected experiments:

```bash
./tools/run_selected_experiments.sh
```


# File Tree: NTUST-Thesis

```
├── 📁 datasets  # Local datasets
│   └── 📁 RAMS  # RAMS dataset root
│       ├── 📁 data  # RAMS split files
│       │   ├── 📄 dev.jsonlines
│       │   ├── 📄 test.jsonlines
│       │   └── 📄 train.jsonlines
│       └── 📁 scorer  # RAMS ontology and scorer resources
│           └── 📄 event_role_multiplicities.txt  # Event-role multiplicity ontology
├── 📁 docs  # Experiment notes and thesis-facing documentation
│   └── 📝 experiment_design.md  # Current experiment plan
├── 📁 dotenv  # Local environment variable templates
│   ├── ⚙️ .env
│   └── ⚙️ .env.example
├── 📁 experimental_configs  # YAML configs for single and batch experiments
│   ├── ⚙️ A1_A2_one-stage_code4struct_gemini25.yaml
│   ├── ...
│   └── ⚙️ small-M1_json_mistral.yaml
├── 📁 legacy  # Deprecated files kept for reference
│   └── ...
├── 📁 outputs  # Generated outputs
│   ├── 📁 analysis  # Analysis artifacts from helper tools
│   ├── 📁 cache  # Runtime caches
│   │   └── 📁 extraction_notes  # Two-stage extraction-note cache
│   └── 📁 runs  # Experiment run outputs
│       ├── 📁 20260423_192420_M1_A1_A2_A3_two-stage_json_gemini3
│       │   ├── ⚙️ config_snapshot.yaml  # Resolved config used for the run
│       │   ├── ⚙️ failed_samples.json  # Unhandled pipeline-level failures
│       │   ├── ⚙️ metrics.json  # Aggregated evaluation results
│       │   └── 📄 predictions.jsonl  # Per-sample prediction records
│       ├── ...
│       └── 📁 20260426_214313_A3_non-icl_two-stage_code4struct_mistral
│           ├── ⚙️ config_snapshot.yaml
│           ├── ⚙️ failed_samples.json
│           ├── ⚙️ metrics.json
│           └── 📄 predictions.jsonl
├── 📁 src  # Python source package
│   └── 📁 ntust_thesis  # Main project package
│       ├── 📁 cli  # Command-line entrypoints
│       │   ├── 🐍 __init__.py
│       │   └── 🐍 run_experiment.py
│       ├── 📁 core  # Config, schemas, registry, and pipeline orchestration
│       │   ├── 🐍 __init__.py
│       │   ├── 🐍 bootstrap.py
│       │   ├── 🐍 config.py
│       │   ├── 🐍 config_models.py
│       │   ├── 🐍 interfaces.py
│       │   ├── 🐍 pipeline.py
│       │   ├── 🐍 registry.py
│       │   ├── 🐍 role_path.py
│       │   └── 🐍 schemas.py
│       ├── 📁 datasets  # Dataset loaders and dataset-specific schemas
│       │   ├── 🐍 __init__.py
│       │   ├── 🐍 rams.py
│       │   └── 🐍 rams_models.py
│       ├── 📁 evaluation  # Metrics, difficulty grouping, and output formatting
│       │   ├── 📁 metrics  # Evaluation metric implementations
│       │   │   ├── 🐍 __init__.py
│       │   │   ├── 🐍 arg.py
│       │   │   ├── 🐍 bemeae.py
│       │   │   ├── 🐍 common.py
│       │   │   ├── 🐍 content_similarity_sbert.py
│       │   │   ├── 🐍 ecar.py
│       │   │   └── 🐍 is_valid_ir.py
│       │   ├── 🐍 __init__.py
│       │   ├── 🐍 difficulty.py
│       │   └── 🐍 formatter.py
│       ├── 📁 ir  # IR parsers, validators, and grammar registry
│       │   ├── 🐍 __init__.py
│       │   ├── 🐍 code4struct_ir.py
│       │   ├── 🐍 common.py
│       │   ├── 🐍 incremental_assignment_ir.py
│       │   ├── 🐍 json_ir.py
│       │   └── 🐍 registry.py
│       ├── 📁 models  # One-stage and two-stage model pipelines
│       │   ├── 📁 components  # Shared extraction, IR generation, compiler, and cache components
│       │   │   ├── 🐍 __init__.py
│       │   │   ├── 🐍 extraction_cache.py
│       │   │   ├── 🐍 extractor.py
│       │   │   ├── 🐍 ir_compiler.py
│       │   │   └── 🐍 ir_generator.py
│       │   ├── 📁 llm  # LLM backend clients
│       │   │   ├── 🐍 __init__.py
│       │   │   ├── 🐍 gemini_client.py
│       │   │   └── 🐍 vllm_client.py
│       │   ├── 🐍 __init__.py
│       │   ├── 🐍 one_stage.py
│       │   └── 🐍 two_stage.py
│       ├── 📁 prompts  # Prompt builders and ICL examples
│       │   ├── 🐍 __init__.py
│       │   ├── 🐍 icl_examples.py
│       │   └── 🐍 model_prompts.py
│       ├── 📁 utils  # Small shared helpers
│       │   ├── 🐍 __init__.py
│       │   ├── 🐍 artifacts.py
│       │   └── 🐍 env.py
│       └── 🐍 __init__.py
├── 📁 tools  # Manual debugging and analysis utilities
│   ├── 🐍 __init__.py
│   ├── 🐍 bemeae_similarity_check.py
│   ├── 🐍 compare_experiment_predictions.py
│   ├── 📄 run_selected_experiments.sh
│   ├── 🐍 sbert_similarity_check.py
│   └── 🐍 test_llm_sensitive_words.py
├── ⚙️ .gitignore
├── ⚙️ .pre-commit-config.yaml
├── 📄 LICENSE
├── 📝 README.md
├── ⚙️ pyproject.toml
├── 🐍 run.py  # Local experiment runner entrypoint
└── 📄 uv.lock
```

---

# Tools Usage

Most tools keep their inputs as constants at the top of the file. Edit those
values before running the tool.

## Run Selected Experiments

Runs every active config listed in `experimental_configs/batch_run_list.yaml`.

```bash
./tools/run_selected_experiments.sh
```

## Check Raw SBERT Similarity

Compares two text spans with SBERT cosine similarity. Edit `SENTENCE_A` and
`SENTENCE_B` in `tools/sbert_similarity_check.py` before running.

```bash
UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run python tools/sbert_similarity_check.py
```

## Check BEMEAE Similarity

Compares either two spans or two role-to-spans dictionaries using BEMEAE
normalization and Hungarian matching. Edit `CHECK_MODE`, `SENTENCE_A`,
`SENTENCE_B`, `GROUP_A`, and `GROUP_B` in
`tools/bemeae_similarity_check.py`.

```bash
UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run python tools/bemeae_similarity_check.py
```

## Compare Two Experiment Runs

Finds samples whose `parsed_output` differs between two run directories. Edit
`EXPERIMENT_DIR_A`, `EXPERIMENT_DIR_B`, `OUTPUT_PATH`, and `COMPARE_MODE` in
`tools/compare_experiment_predictions.py`.

```bash
PYTHONPATH=src uv run python tools/compare_experiment_predictions.py
```

## Probe Raw LLM Safety Responses

Sends a test prompt to the configured Gemini/vLLM targets and prints the raw API
responses. Edit `SYSTEM_PROMPT`, `USER_PROMPT`, and `TARGET_MODELS` in
`tools/test_llm_sensitive_words.py`.

```bash
PYTHONPATH=src uv run python tools/test_llm_sensitive_words.py
```

# Citation

Citation information will be added after the thesis is published.

```bibtex
@misc{ir-grammar-study-placeholder,
  title = {Intermediate Representation Design for Structured Generation: An Empirical Study of IR Grammars},
  author = {TBD},
  year = {TBD},
  url = {https://example.com/paper-placeholder}
}
```
