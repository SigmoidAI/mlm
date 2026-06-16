# Spinning Flows

Both spinning scripts extend the base cascade scripts with a **per-level retry loop** ("spin").
At each level the judge's feedback is fed back into the prompt so the model can self-correct
before the pipeline escalates to a stronger (more expensive) model.

---

## How spinning works

```
for each model level (cheapest → most expensive):
    spin 1  →  generate answer  →  judge scores it
    spin 2  →  refine with judge feedback  →  judge scores again
    spin 3  →  refine again  →  judge scores again
    ...
    stop spinning if:
        score >= ACCEPTABLE_SCORE  ✅  done, skip remaining levels
        improvement < IMPROVEMENT_THRESHOLD (spin 2+)  ⬆  escalate now
        MAX_LOOPS_PER_LEVEL reached  ⬆  escalate
    carry last judge feedback forward to the next level
```

### Tunable constants

| Variable | Default | Override via env |
|---|---|---|
| `MAX_LOOPS_PER_LEVEL` | 3 (simple) / 2 (complex) | `MAX_LOOPS_PER_LEVEL=N` |
| `IMPROVEMENT_THRESHOLD` | 0.05 | `IMPROVEMENT_THRESHOLD=0.03` |
| `ACCEPTABLE_SCORE` | 0.92 (simple) / 0.95 (complex) | hardcoded, edit in script |

### Judge feedback

The judge returns `score` (0–1), `verdict` (Valid/Invalid), and `feedback` (specific fix instructions).
The `feedback` field is injected into the refinement prompt for every subsequent spin and for every
level escalation. The judge's internal `reasoning` field is not forwarded to worker models.

If the judge returns an empty `feedback` string the fallback is `"Score {score:.2f}"` — not ideal,
but it only happens when the judge omits the field entirely.

---

## 04_run_simple_spin.py — Simple cascade with spinning

**Based on:** `02_run_simple.py`

**Model levels** (from `simple_flow` in `cascade_models.yaml`):

| Level | Model |
|---|---|
| 1 | `mistralai/mistral-small-3.2-24b-instruct` |
| 2 | `deepseek/deepseek-v3.2` |
| 3 | `deepseek/deepseek-chat-v3.1` |

**Judge:** `judge_model_1` (`openai/gpt-oss-120b:exacto`)

**MLflow experiment:** `SimpleFlow_Spin_V2_v{N}` (auto-incremented)

**Run:**
```bash
cd src
NUM_QUESTIONS=5 uv run python scripts/04_run_simple_spin.py

# With custom spin settings
NUM_QUESTIONS=10 MAX_LOOPS_PER_LEVEL=2 IMPROVEMENT_THRESHOLD=0.03 uv run python scripts/04_run_simple_spin.py
```

**What gets logged to MLflow per question:**
- `lvl{L}_spin{S}_score` — judge score at level L, spin S
- `lvl{L}_spin{S}_improvement` — score delta vs previous spin
- `lvl{L}_spin{S}_passed` — 1 if accepted, 0 otherwise
- `lvl{L}_spin{S}_model_cost` / `judge_cost` — USD cost per call
- params: `acceptable_score`, `max_loops_per_level`, `improvement_threshold`, model names

---

## 05_run_complex_spin.py — Complex cascade with spinning

**Based on:** `03_run_complex.py`

One "spin" at the complex level is a full multi-agent cycle:
initial answers → debate (critiques) → self-refinement loop → validator ensemble vote.
The validator's per-worker feedback is forwarded into the next spin via the ensemble prompt.

**Cascade levels** (from `cascade_complex_run` in `cascade_models.yaml`):

| Level | Workers |
|---|---|
| 1 | Mistral Small 3.2 24B + Llama 3.1 8B |
| 2 | DeepSeek V3.2 + Mistral Small 3.2 24B + Gemma 3 27B |
| 3 | Llama 3.1 70B + Qwen 2.5 32B + DeepSeek V3.2 |
| 4 | DeepSeek Chat V3.1 + Llama 3.1 70B + Qwen 2.5 72B + Hermes 4 70B |
| 5 | MiniMax M2.5 + GPT OSS 120B + Qwen3 Coder Next + DeepSeek Chat V3.1 |

**Judge:** `ValidatorAgent` (ensemble validator from complex workflow)

**MLflow experiment:** `complex_workflow_spin_{NUM_MAX_QUESTIONS}_v{N}` (auto-incremented)

**Run:**
```bash
cd src
NUM_MAX_QUESTIONS=5 uv run python -m src.scripts.05_run_complex_spin

# With custom spin settings
MAX_LOOPS_PER_LEVEL=3 IMPROVEMENT_THRESHOLD=0.03 uv run python -m src.scripts.05_run_complex_spin
```

**Key difference from simple flow:** each spin is a full ensemble cycle (not a single model call),
so `MAX_LOOPS_PER_LEVEL=2` already means 2× the full multi-agent pipeline per cascade level.

---

## Differences between the two flows

| | Simple (04) | Complex (05) |
|---|---|---|
| Workers per level | 1 model | 2–5 models (ensemble) |
| Spin = one | single model call | full initial→debate→refine→vote cycle |
| Default max spins | 3 | 2 |
| Default acceptable score | 0.92 | 0.95 |
| Judge feedback source | `feedback` field | per-worker `reason` from ensemble validator |
| Cost per spin | low | high (multiple models × debate rounds) |
