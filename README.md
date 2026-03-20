## Sigmoid MLM – Multi-Level LLM Orchestration

This repository contains the research code behind Sigmoid’s experiments on **cost‑efficient large language model orchestration**. The project explores how to coordinate multiple open‑weights models in cascades and debates so that:

- simple questions are answered by **small, cheap models**, and
- only genuinely hard questions are escalated to **larger, more expensive models**.

The system is built around three main evaluation workflows, all logged with MLflow and benchmarked on the **Arena Hard** dataset:

- **Baseline** – a single strong model used in isolation
- **Simple Flow** – an **iterative cascade** of models (small → large) with automatic early‑exit
- **Complex Workflow** – a **hybrid cascade + multi‑agent debate** architecture, where workers collaborate and a validator triggers escalation only when needed

These workflows are used to study the trade‑off between **answer quality**, **token cost**, and **structural design** of LLM systems, as described in the accompanying articles.

---

## High‑Level Architecture

The code in `src/` is organized into a few key layers:

- **Agents** (`src/agents/`)
	- `pydantic_agent.WorkingAgent` – wraps a concrete OpenRouter model via `pydantic_ai`, responsible for generating answers and, in complex flows, critiquing peers.
	- `pydantic_agent.ValidatorAgent` – a judge model that scores answers, parses structured JSON feedback, and decides whether the quality threshold is met.
- **Configuration** (`src/config/`)
	- `cascade_models.yaml` – defines the **cascade topology** (levels, worker models, endpoints, and pricing) for both simple and complex workflows.
	- `make_config.py` – loads `.env`, expands `${VAR}` placeholders in YAML, and returns a fully‑materialized config dict.
	- `settings.py` – central MLflow and default system‑prompt configuration.
- **Data Layer** (`src/datasets/`)
	- `loader.py` – ingests the **Arena Hard** benchmark from Hugging Face (`lmarena-ai/arena-hard-auto`) into an MLflow GenAI dataset.
- **Evaluation & Analysis** (`src/evaluation/`)
	- `arena_judger.py` – implements an **Arena‑style validator** used to compare Simple/Complex workflows against a GPT‑4.1 baseline.
	- `run_benchmark.py` – pulls MLflow runs, computes arena scores, and estimates token‑level cost vs. a GPT‑4.1 reference.
- **Tracking** (`src/tracking/`)
	- `mlflow_logger.py`, `token_counter.py` – helper utilities for MLflow traces, spans, and cost/usage accounting.
- **Entry Scripts** (`src/scripts/`)
	- `01_run_baseline.py` – single‑model baseline evaluation.
	- `02_run_simple.py` – **iterative cascade** evaluation.
	- `03_run_complex.py` – **hybrid cascade + multi‑agent debate** workflow.

All flows rely on MLflow for experiment versioning, trace logging, and dataset management.

---

## Workflows

### 1. Baseline: Single‑Model Evaluation

Script: `src/scripts/01_run_baseline.py`

Goal: measure **quality and cost** of a single strong model (e.g. an Azure OpenAI deployment) on the Arena Hard dataset.

Core behavior:

- Loads an MLflow GenAI dataset (see `mlflow.genai.datasets.get_dataset`).
- Connects to an Azure OpenAI chat model via `pydantic_ai`.
- For each question:
	- starts a dedicated MLflow run named by `question_id`,
	- generates an answer,
	- logs metadata (category, deployment name, answer length, success flag),
	- attaches a trace with `mlflow.trace` for downstream analysis.

This baseline is later used as the **reference system** when comparing cascades and debates.

---

### 2. Simple Flow: Iterative Cascade

Script: `src/scripts/02_run_simple.py`

This workflow implements the **“smaller, smarter, cheaper”** idea: start with a cheap worker model and escalate only if the answer is poor. The high‑level loop is:

1. Load a set of worker models from `cascade_models.yaml` under the `simple_flow` key (small → medium → large).
2. Load an evaluation dataset from MLflow (`DATASET_ID` and `NUM_QUESTIONS` are read from `.env`).
3. For each question:
	 - Call the first worker model.
	 - Use a `ValidatorAgent` to judge the answer and assign a continuous score.
	 - If the score ≥ `ACCEPTABLE_SCORE` (quality threshold), stop and return the answer.
	 - Otherwise, build a **refinement prompt** that includes the question, previous answer, and feedback, and pass it to the next model in the cascade.
4. Log every call, verdict, and cost to MLflow.

Cost tracking is integrated in two ways:

- The OpenRouter responses are intercepted via a patched `openai.resources.chat.completions` client, which extracts token usage and reported cost.
- `calculate_openrouter_cost` uses the pricing metadata from `cascade_models.yaml` to estimate input and output token cost per model.

This setup allows you to reproduce the **Simple Flow** experiments from the article series: measuring how often cheap models are sufficient and how much cost can be saved compared to a strong baseline.

---

### 3. Complex Workflow: Debate‑Triggered Cascade

Script: `src/scripts/03_run_complex.py`

The complex workflow combines two ideas:

- a **multi‑level cascade** (small → mid‑tier → large models), and
- an internal **multi‑agent debate** at each level.

Key concepts:

- **Working Agents** – multiple `WorkingAgent` instances per cascade level with different prompts (strict, very strict, medium strict). They propose candidate answers and can critique each other.
- **Validator / Judge** – a `ValidatorAgent` running a dedicated judge model, which:
	- evaluates all worker answers,
	- orchestrates a “voting” or scoring process, and
	- returns a JSON structure containing the best answer and per‑worker scores.
- **Cascade Levels** – defined in `cascade_models.yaml` under `cascade_complex_run` as `cascade_lvl_1`…`cascade_lvl_5`, each with its own group of workers and pricing.

The workflow roughly follows:

1. **Dataset selection** – fetch a benchmark dataset from MLflow via `search_datasets` (e.g. `arena_hard_v2_0`).
2. **Level 1 debate** – spawn worker agents from `cascade_lvl_1`, run them on the question, and send their answers to the judge.
3. **Validator decision** – if the best answer’s score ≥ `ACCEPTABLE_SCORE`, accept and stop; otherwise, escalate.
4. **Contextual escalation** – build a richer prompt for the next level that includes:
	 - the original question,
	 - worker answers and scores from the previous level,
	 - the judge’s explanations.
5. **Repeat** – continue through up to `MAX_CASCADE_LEVEL` (default 5). For unresolved queries, the final level’s best answer is returned, even if it did not pass the threshold.

This architecture supports the “debate‑triggered cascade” experiments discussed in the articles, including the **v1/v2/v3 complex workflows**, where you vary:

- the number of levels,
- the size and type of models per level,
- and the strictness of prompts and judges.

---

## Datasets and Benchmarks

### Arena Hard ingestion

Module: `src/datasets/loader.py`

This utility converts the public **Arena Hard Auto** dataset into an MLflow GenAI dataset:

- Downloads `lmarena-ai/arena-hard-auto` from Hugging Face, focusing on the `question.jsonl` split for version 2.0.
- Normalizes each record into an MLflow‑compatible format with `inputs`, `tags`, and optional `expectations` (reference answers, multi‑turn metadata, etc.).
- Stores the result under a dedicated MLflow experiment (default `DATASET_Arena_Hard_V2`) and dataset name (default `arena_hard_v2_0`).

You can run this ingestion once and reuse the dataset across all workflows.

### Benchmarking vs. GPT‑4.1

Module: `src/evaluation/run_benchmark.py`

This script reads MLflow runs for a given experiment (e.g. Simple Flow, Complex Workflow, or the GPT‑4.1 baseline) and:

- aligns question‑level runs between **System A** (e.g. your cascade) and **System B** (GPT‑4.1 baseline),
- asks an Arena‑style judge (`ArenaValidatorAgent`) to compare the two answers,
- computes **arena scores** (A≫B, A>B, A=B, B>A, B≫A) and maps them to numeric values,
- reconstructs token usage and cost using both GPT‑4.1 list prices and the per‑model prices from `cascade_models.yaml`.

This is how the figures in the articles (win rates per level, cost curves, funnel plots, answer length distributions) are produced.

---

## MLflow & Docker Setup

The repository includes a ready‑to‑run MLflow tracking server backed by Postgres.

- `docker-compose.yml` – defines:
	- a Postgres 15 service for the MLflow backend store, and
	- an MLflow server container based on `mlflow/mlflow:v3.9.0` with basic auth support.
- `mlflow/Dockerfile` – extends the MLflow image with `psycopg2-binary` and `mlflow[auth]`.

To start the tracking stack locally:

```bash
cd mlm
docker compose up -d
```

The server will be available at `http://localhost:5000` and is used by all scripts as the default `MLFLOW_TRACKING_URI`.

---

## Installation & Configuration

### 1. Python environment

The project targets **Python 3.13+** (see `src/pyproject.toml`). You can create a virtual environment via your preferred tool, for example:

```bash
cd mlm/src
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file at the project root (`mlm/.env`) with at least:

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000

# OpenRouter for cascade/debate workflows
OPENROUTER_API_KEY=...

# Dataset settings (Simple Flow)
DATASET_ID=...          # MLflow GenAI dataset id
NUM_QUESTIONS=0         # 0 = use all questions

# Optional: Azure baseline (01_run_baseline)
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_API_VERSION=2024-08-01-preview
AZURE_DEPLOYMENT_NAME=GPT4o
```

The configuration loader (`config/make_config.py`) will also expand `${VAR}` placeholders found in `cascade_models.yaml`, so your `.env` is the single source of truth for keys and endpoints.

---

## Running the Experiments

Below is a suggested order if you want to reproduce the core ideas from the articles.

### Step 0 – Ingest the benchmark dataset

From `mlm/src`:

```bash
python -m datasets.loader
```

This creates an MLflow experiment (default `DATASET_Arena_Hard_V2`) and a dataset (default `arena_hard_v2_0`). You can then reference this dataset either by `dataset_id` (for the baseline and simple flow) or by name/experiment (in the complex workflow).

### Step 1 – Baseline single‑model evaluation

```bash
python -m scripts.01_run_baseline
```

This runs a fixed Azure OpenAI deployment on a slice of the dataset and logs each question as an MLflow run. The resulting experiment acts as the **GPT‑4.1‑like baseline** in later comparisons.

### Step 2 – Simple Flow (iterative cascade)

```bash
python -m scripts.02_run_simple
```

Make sure `DATASET_ID` in your `.env` points to the Arena Hard MLflow dataset id. The script will:

- use the `simple_flow` section in `cascade_models.yaml` to configure the worker models,
- apply the iterative cascade logic described above,
- and log detailed traces, verdicts, and cost metrics to MLflow.

### Step 3 – Complex Workflow (debate‑triggered cascade)

```bash
python -m scripts.03_run_complex
```

This run uses the `cascade_complex_run` section from `cascade_models.yaml` to:

- set up 3–5 cascade levels with multiple workers per level,
- run internal debates and votes at each stage,
- and only send the hardest questions to the strongest models.

You can tweak:

- `NUM_MAX_QUESTIONS` – how many prompts from the dataset to process,
- `MAX_CASCADE_LEVEL` – number of active levels (e.g. 3‑level vs 5‑level cascade),
- `ACCEPTABLE_SCORE` – how strict the judge must be.

### Step 4 – Aggregate results and compare vs baseline

After running one or more workflows, use the benchmarking tools under `src/evaluation/` to:

- align your system’s runs with the baseline,
- compute arena scores (win/loss/tie),
- and estimate cost per question relative to GPT‑4.1 pricing.

This is where you can recreate plots similar to those in the Medium articles: cost vs quality, funnel per cascade level, and answer length distributions.

---

## Extending the Project

This codebase is designed to be **experiment‑friendly**:

- Swap in new models by editing `cascade_models.yaml` (including their prices and endpoints).
- Adjust prompts (worker and judge) in `scripts/03_run_complex.py` and `config/prompts.py`.
- Change the evaluator (e.g. judge model or scoring scheme) via `ValidatorAgent` or `ArenaValidatorAgent`.
- Plug in new datasets by adapting `datasets/loader.py` or using additional MLflow GenAI datasets.

By tuning these pieces, you can explore your own **reasoning budgets**, cascades, and debate patterns, and study how architecture and orchestration impact both cost and quality.

---

## Credits & Acknowledgements

This project is developed by **Sigmoid** and supported by the **Innovate Moldova Programme**, financed by Sweden and the United Kingdom.

