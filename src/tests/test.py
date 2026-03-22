"""
test_cascade_lib.py — Manual smoke-test for cascade_lib

Press Run (▶) in your IDE — all suites execute automatically.

To skip expensive suites while iterating, set the flags at the top of
the RUN CONFIGURATION section below (search for "# --- RUN CONFIGURATION").
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment — load .env before anything else so cascade_lib picks up the keys
# ---------------------------------------------------------------------------
# src/tests/ → parent = src/ → parent = repo root
_root     = Path(__file__).resolve().parent.parent.parent  # repo root (mlm project root)
_env_path = _root / "mlm_lib/.env"
load_dotenv(dotenv_path=_env_path)

_required = ["OPENROUTER_API_KEY", "MLFLOW_TRACKING_URI"]
_missing  = [k for k in _required if not os.getenv(k)]
if _missing:
    print(f"[ERROR] Missing required environment variables: {', '.join(_missing)}")
    print(f"        Looked for .env at: {_env_path}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Path setup — repo root and src/ on sys.path so all local imports resolve
# ---------------------------------------------------------------------------
for _p in [str(_root), str(_root / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# cascade_lib import
# ---------------------------------------------------------------------------
from mlm_lib import (
    CascadeRunner,
    CascadeRunnerConfig,
    SimpleConfig,
    ComplexConfig,
    JudgeConfig,
    MLflowConfig,
    RunType,
    RunResult,
    TemperatureStrategy,
    DEFAULT_SIMPLE_SYSTEM_PROMPT,
    DEFAULT_COMPLEX_SYSTEM_PROMPT,
)

# ===========================================================================
# Custom questions
# ===========================================================================
# Add, remove, or edit freely.  Each dict must have:
#   question_id  — unique string identifier
#   question     — the prompt sent to the models
#   category     — "hard_prompt" | "creative_writing" | anything else

SIMPLE_QUESTIONS = [
    {
        "question_id": "simple_factual_1",
        "question": "What is the capital of France?",
        "category": "factual",
    },
    {
        "question_id": "simple_factual_2",
        "question": "What is the difference between TCP and UDP protocols?",
        "category": "hard_prompt",
    },
    {
        "question_id": "simple_code_1",
        "question": (
            "Write a Python function that takes a list of integers and returns "
            "a new list with duplicates removed, preserving the original order."
        ),
        "category": "hard_prompt",
    },
    {
        "question_id": "simple_reasoning_1",
        "question": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
            "the ball. How much does the ball cost? Show your reasoning step by step."
        ),
        "category": "hard_prompt",
    },
    {
        "question_id": "simple_creative_1",
        "question": (
            "Write a short two-paragraph story about a lighthouse keeper who "
            "discovers an unusual message in a bottle."
        ),
        "category": "creative_writing",
    },
]

COMPLEX_QUESTIONS = [
    {
        "question_id": "complex_code_1",
        "question": (
            "Design and implement a Python class for a thread-safe LRU cache "
            "with a configurable maximum size. Include get, put, and delete methods, "
            "and write at least three unit tests."
        ),
        "category": "hard_prompt",
    },
    {
        "question_id": "complex_analysis_1",
        "question": (
            "Is there an early stop-out method to control for the multiple testing "
            "problem in hypothesis tests for a dataset with initial probabilities of passing? "
            "Describe the most appropriate approaches and when to use each."
        ),
        "category": "hard_prompt",
    },
    {
        "question_id": "complex_creative_1",
        "question": (
            "Write the opening scene (300–400 words) of a science-fiction short story "
            "set on a generation ship, where the protagonist discovers the ship's logs "
            "have been falsified for decades."
        ),
        "category": "creative_writing",
    },
]

# Questions used for the batch test (mix of both types)
BATCH_QUESTIONS = [
    {
        "question_id": "batch_q1",
        "question": "Explain the CAP theorem in distributed systems.",
        "category": "hard_prompt",
    },
    {
        "question_id": "batch_q2",
        "question": "What are the main differences between REST and GraphQL APIs?",
        "category": "hard_prompt",
    },
    {
        "question_id": "batch_q3",
        "question": "Write a haiku about recursion in programming.",
        "category": "creative_writing",
    },
]


# ===========================================================================
# Helpers
# ===========================================================================

SEPARATOR = "=" * 65

def _sep(title: str = "") -> None:
    if title:
        pad = max(0, 65 - len(title) - 4)
        print(f"\n{'=' * 2} {title} {'=' * pad}")
    else:
        print(SEPARATOR)

def _print_result(result: RunResult) -> None:
    print(f"\n  summary  : {result.summary()}")
    print(f"  success  : {result.success}")
    print(f"  cost     : ${result.total_cost:.6f}")
    if result.run_type == RunType.SIMPLE:
        print(f"  iterations     : {result.iterations}")
        print(f"  winning model  : {result.winning_model_name}")
        for rec in result.iteration_history:
            status = "✓" if rec.passed else "✗"
            print(f"    [{status}] iter {rec.iteration} | {rec.model_name} | score={rec.score:.4f} | {rec.verdict}")
    else:
        print(f"  winning level  : {result.winning_cascade_level}")
        print(f"  best score     : {result.best_confidence_score:.4f}")
        print(f"  winning model  : {result.winning_model_name}")
        for lvl in result.cascade_level_history:
            status = "✓" if lvl.passed else "✗"
            print(f"    [{status}] level {lvl.level} | score={lvl.best_score:.4f} | cost=${lvl.level_cost:.6f}")
    print(f"\n  answer preview:\n  {result.answer[:300].strip()}{'...' if len(result.answer) > 300 else ''}")

def _save_results(tag: str, results: list[RunResult]) -> None:
    out_dir  = Path(__file__).parent / "test_outputs"
    out_dir.mkdir(exist_ok=True)
    filename = out_dir / f"{tag}_{int(time.time())}.jsonl"
    with open(filename, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps({
                "run_type":             r.run_type.value,
                "question_id":          r.question_id,
                "question":             r.question,
                "answer":               r.answer,
                "success":              r.success,
                "total_cost":           r.total_cost,
                "iterations":           r.iterations,
                "winning_model":        r.winning_model,
                "winning_model_name":   r.winning_model_name,
                "winning_level":        r.winning_cascade_level,
                "best_score":           r.best_confidence_score,
            }, ensure_ascii=False) + "\n")
    print(f"\n  Results saved → {filename}")


# ===========================================================================
# Test suites
# ===========================================================================

def test_simple_defaults() -> list[RunResult]:
    """
    Simple flow with all defaults.
    Reads API key and MLflow URI from environment.
    """
    _sep("SUITE: simple — default config")

    runner = CascadeRunner(
        "simple",
        experiment_name="test_simple_defaults",
    )
    print(f"  runner   : {runner}")
    print(f"  questions: {len(SIMPLE_QUESTIONS)}")

    results = []
    for q in SIMPLE_QUESTIONS:
        _sep(q["question_id"])
        print(f"  Q: {q['question'][:100]}")
        result = runner.run(
            question    = q["question"],
            question_id = q["question_id"],
            category    = q["category"],
        )
        _print_result(result)
        results.append(result)

    _save_results("simple_defaults", results)
    return results


def test_simple_custom_prompts() -> list[RunResult]:
    """
    Simple flow with a strict custom system prompt and a custom
    refinement template.
    """
    _sep("SUITE: simple — custom prompts")

    STRICT_SYSTEM = (
        "You are a highly precise assistant. "
        "Answer the question asked — nothing more, nothing less. "
        "Be factually accurate and concise."
    )

    STRICT_REFINEMENT = (
        "Your previous answer did not meet the quality bar.\n\n"
        "ORIGINAL QUESTION:\n{question}\n\n"
        "YOUR ANSWER (attempt {attempt}):\n{prev_answer}\n\n"
        "JUDGE FEEDBACK:\n{issue}\n\n"
        "Rewrite your answer addressing every point in the feedback. "
        "Be thorough, accurate, and do not repeat previous mistakes."
    )

    runner = (
        CascadeRunner("simple", mlflow_enabled=False)
        .set_system_prompt(STRICT_SYSTEM)
        .set_refinement_template(STRICT_REFINEMENT)
        .set_acceptable_score(0.85)
    )
    print(f"  runner : {runner}")

    results = []
    for q in SIMPLE_QUESTIONS[:2]:          # run only first 2 to keep costs low
        _sep(q["question_id"])
        print(f"  Q: {q['question'][:100]}")
        result = runner.run(
            question    = q["question"],
            question_id = q["question_id"],
            category    = q["category"],
        )
        _print_result(result)
        results.append(result)

    _save_results("simple_custom_prompts", results)
    return results


def test_simple_exhaust_all() -> list[RunResult]:
    """
    Simple flow with exhaust_all_models=True — benchmarks every model
    in the cascade and returns the best-scoring answer.
    """
    _sep("SUITE: simple — exhaust all models (benchmark mode)")

    runner = (
        CascadeRunner("simple", mlflow_enabled=False)
        .exhaust_all_models(True)
        .set_acceptable_score(0.80)
    )
    print(f"  runner : {runner}")

    results = []
    q = SIMPLE_QUESTIONS[1]                 # just one question for speed
    _sep(q["question_id"])
    print(f"  Q: {q['question'][:100]}")
    result = runner.run(
        question    = q["question"],
        question_id = f"{q['question_id']}_exhaust",
        category    = q["category"],
    )
    _print_result(result)
    results.append(result)

    _save_results("simple_exhaust", results)
    return results


def test_complex_defaults() -> list[RunResult]:
    """
    Complex flow with all defaults.
    """
    _sep("SUITE: complex — default config")

    runner = CascadeRunner(
        "complex",
        experiment_name="test_complex_defaults",
        max_cascade_levels=2,               # keep to 2 levels to limit cost
    )
    print(f"  runner   : {runner}")
    print(f"  questions: {len(COMPLEX_QUESTIONS)}")

    results = []
    for q in COMPLEX_QUESTIONS:
        _sep(q["question_id"])
        print(f"  Q: {q['question'][:100]}")
        result = runner.run(
            question    = q["question"],
            question_id = q["question_id"],
            category    = q["category"],
        )
        _print_result(result)
        results.append(result)

    _save_results("complex_defaults", results)
    return results


def test_complex_custom_prompts() -> list[RunResult]:
    """
    Complex flow with custom system prompt, judge prompt, and
    next-level escalation prefix.
    """
    _sep("SUITE: complex — custom prompts")

    CUSTOM_SYSTEM = (
        "You are an expert engineer and analyst. "
        "Provide accurate, well-structured, and complete answers. "
        "When reviewing peer responses, be constructively critical "
        "and only accept improvements that increase precision."
    )

    CUSTOM_JUDGE = (
        "Below are answers from multiple AI worker agents to the same question.\n"
        "Evaluate each answer for: correctness, completeness, clarity, and code quality "
        "(if applicable).\n"
        "Score each 0.0000–1.0000 and select the best.\n\n"
        "Return ONLY valid JSON in this exact shape:\n"
        "```json\n"
        "\"evaluation\": {\n"
        "  \"question\": <question>,\n"
        "  \"best_answer\": {\n"
        "    \"best_worker_model_id\": <id>,\n"
        "    \"best_confidence_score\": <float>,\n"
        "    \"best_reason\": <reason>\n"
        "  },\n"
        "  \"individual_reviews\": {\n"
        "    \"<worker_id>\": {\"confidence_score\": <float>, \"reason\": <reason>}\n"
        "  }\n"
        "}\n"
        "```"
    )

    CUSTOM_NEXT_LEVEL = (
        "The previous cascade level did not produce a satisfactory answer. "
        "Study the workers' attempts and their judge reviews carefully, then "
        "produce a significantly improved answer:"
    )

    runner = (
        CascadeRunner("complex", mlflow_enabled=False)
        .set_system_prompt(CUSTOM_SYSTEM)
        .set_judge_prompt(CUSTOM_JUDGE)
        .set_next_level_prefix(CUSTOM_NEXT_LEVEL)
        .set_max_levels(2)
        .set_acceptable_score(0.90)
        .set_temperature_strategy(
            "category_aware",
            hard_prompt_range=(0.2, 0.5),
            creative_range=(0.7, 1.1),
        )
    )
    print(f"  runner : {runner}")

    results = []
    q = COMPLEX_QUESTIONS[0]                # one question to limit cost
    _sep(q["question_id"])
    print(f"  Q: {q['question'][:100]}")
    result = runner.run(
        question    = q["question"],
        question_id = f"{q['question_id']}_custom",
        category    = q["category"],
    )
    _print_result(result)
    results.append(result)

    _save_results("complex_custom_prompts", results)
    return results


def test_batch_simple() -> list[RunResult]:
    """
    Batch run using simple flow over BATCH_QUESTIONS.
    """
    _sep("SUITE: batch — simple flow")

    runner = CascadeRunner(
        "simple",
        experiment_name="test_batch_simple",
        acceptable_score=0.88,
    )
    print(f"  runner   : {runner}")
    print(f"  questions: {len(BATCH_QUESTIONS)}")

    results = runner.run_batch(
        records       = BATCH_QUESTIONS,
        max_questions = 0,                  # 0 = process all
    )

    for r in results:
        _sep(r.question_id)
        _print_result(r)

    _save_results("batch_simple", results)
    return results


def test_full_config_object() -> list[RunResult]:
    """
    Demonstrates building a CascadeRunnerConfig object explicitly
    and passing it to CascadeRunner.
    """
    _sep("SUITE: full config object — simple flow")

    cfg = CascadeRunnerConfig(
        run_type          = RunType.SIMPLE,
        openrouter_api_key= os.getenv("OPENROUTER_API_KEY"),   # explicit, not from env
        models_yaml_path  = None,                               # auto-detect

        judge = JudgeConfig(
            judge_model_key  = "judge_model_1",
            acceptable_score = 0.90,
        ),

        simple = SimpleConfig(
            model_config_key = "simple_flow",
            system_prompt    = (
                "You are a concise, accurate assistant. "
                "Prioritise correctness over length."
            ),
            exhaust_all_models = False,
        ),

        mlflow = MLflowConfig(
            tracking_uri         = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/"),
            experiment_base_name = "test_full_config",
            auto_version         = True,
            enabled              = True,
            autolog_pydantic     = True,
        ),
    )

    runner = CascadeRunner(config=cfg)
    print(f"  runner : {runner}")

    results = []
    q = SIMPLE_QUESTIONS[2]                 # code question
    _sep(q["question_id"])
    print(f"  Q: {q['question'][:100]}")
    result = runner.run(
        question    = q["question"],
        question_id = f"{q['question_id']}_cfg_obj",
        category    = q["category"],
    )
    _print_result(result)
    results.append(result)

    _save_results("full_config_object", results)
    return results


# ===========================================================================
# RUN CONFIGURATION — toggle suites on/off here, then press ▶
# ===========================================================================

RUN_SIMPLE_DEFAULTS      = True   # simple flow, all defaults
RUN_SIMPLE_CUSTOM_PROMPTS= True   # simple flow, custom system + refinement prompts
RUN_SIMPLE_EXHAUST_ALL   = True   # simple flow, benchmark mode (tries every model)
RUN_COMPLEX_DEFAULTS     = True   # complex flow, all defaults
RUN_COMPLEX_CUSTOM_PROMPTS= True  # complex flow, custom system/judge/escalation prompts
RUN_BATCH_SIMPLE         = True   # batch run over BATCH_QUESTIONS
RUN_FULL_CONFIG_OBJECT   = True   # explicit CascadeRunnerConfig construction


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    _sep()
    print("  cascade_lib smoke-test")
    print(f"  MLFLOW_URI     : {os.getenv('MLFLOW_TRACKING_URI')}")
    print(f"  OPENROUTER_KEY : {'set ✓' if os.getenv('OPENROUTER_API_KEY') else 'MISSING ✗'}")
    _sep()

    # Build the ordered list of suites to run based on the flags above
    suite_plan: list[tuple[str, callable]] = []
    if RUN_SIMPLE_DEFAULTS:       suite_plan.append(("simple_defaults",       test_simple_defaults))
    if RUN_SIMPLE_CUSTOM_PROMPTS: suite_plan.append(("simple_custom_prompts", test_simple_custom_prompts))
    if RUN_SIMPLE_EXHAUST_ALL:    suite_plan.append(("simple_exhaust_all",    test_simple_exhaust_all))
    if RUN_COMPLEX_DEFAULTS:      suite_plan.append(("complex_defaults",      test_complex_defaults))
    if RUN_COMPLEX_CUSTOM_PROMPTS:suite_plan.append(("complex_custom_prompts",test_complex_custom_prompts))
    if RUN_BATCH_SIMPLE:          suite_plan.append(("batch_simple",          test_batch_simple))
    if RUN_FULL_CONFIG_OBJECT:    suite_plan.append(("full_config_object",    test_full_config_object))

    if not suite_plan:
        print("  All suites are disabled. Set at least one RUN_* flag to True.")
        return

    print(f"  Running {len(suite_plan)} suite(s):")
    for name, _ in suite_plan:
        print(f"    • {name}")
    _sep()

    all_res: list[RunResult] = []
    t_start = time.time()

    for name, fn in suite_plan:
        try:
            results = fn()
            all_res.extend(results)
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] Suite '{name}' failed: {exc}")
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Grand summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    _sep("GRAND SUMMARY")
    if all_res:
        n          = len(all_res)
        succeeded  = sum(1 for r in all_res if r.success)
        total_cost = sum(r.total_cost for r in all_res)
        print(f"  questions run : {n}")
        print(f"  succeeded     : {succeeded}/{n} ({100 * succeeded / n:.1f}%)")
        print(f"  total cost    : ${total_cost:.6f}")
        print(f"  elapsed       : {elapsed:.1f}s")
        print()
        for r in all_res:
            tick = "✓" if r.success else "✗"
            print(f"  [{tick}] {r.summary()}")
    else:
        print("  No results collected.")
    _sep()


if __name__ == "__main__":
    main()