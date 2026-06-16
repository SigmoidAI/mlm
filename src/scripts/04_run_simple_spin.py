"""
04_run_simple_spin.py - Simple Cascade with Per-Level Spinning

Same as 02_run_simple.py but at each model level the judge's feedback is
fed back into the prompt and the model is retried (spun) until:
  - the score reaches ACCEPTABLE_SCORE  (done)
  - the score improves by less than IMPROVEMENT_THRESHOLD vs previous spin (escalate)
  - MAX_LOOPS_PER_LEVEL retries are exhausted (escalate)

When escalating the full judge feedback is always carried forward to the next level.
"""

import asyncio
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import yaml
import mlflow
import mlflow.pydantic_ai
from mlflow.entities import SpanType
from mlflow.genai.datasets import get_dataset
import openai.resources.chat.completions
from collections import defaultdict

from src.agents.pydantic_agent import WorkingAgent, ValidatorAgent

warnings.filterwarnings("ignore", category=ResourceWarning)
try:
    import openai._base_client as _oai_client
    for _wrapper in ['SyncHttpxClientWrapper', 'AsyncHttpxClientWrapper']:
        if hasattr(_oai_client, _wrapper):
            setattr(getattr(_oai_client, _wrapper), '__del__', lambda self: None)
except (ImportError, AttributeError):
    pass


# ── CONSTANTS ────────────────────────────────────────────────────────────────

MLFLOW_URI: str       = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/")
DATASET_ID: str       = os.getenv("DATASET_ID")
NUM_QUESTIONS: int    = int(os.getenv("NUM_QUESTIONS", "0"))
MODEL_CONFIG_KEY: str = "simple_flow"
JUDGE_MODEL_KEY: str  = "judge_model_1"

if not DATASET_ID:
    raise RuntimeError("DATASET_ID must be set in .env")

ACCEPTABLE_SCORE: float     = 0.92
MAX_LOOPS_PER_LEVEL: int    = int(os.getenv("MAX_LOOPS_PER_LEVEL", "3"))
IMPROVEMENT_THRESHOLD: float = float(os.getenv("IMPROVEMENT_THRESHOLD", "0.05"))

SYSTEM_PROMPT: str = "You are a helpful AI assistant. Provide detailed, accurate answers."

REFINEMENT_PROMPT_TEMPLATE: str = """The previous answer did not fully meet the requirements.

QUESTION:
{question}

PREVIOUS ANSWER (Attempt {attempt}):
{prev_answer}

FEEDBACK ON PREVIOUS ANSWER:
{issue}

Please revise your answer to address the feedback above. Ensure your response is thorough, accurate, and covers all aspects of the question. Be clear, detailed, and avoid repeating previous mistakes."""


# ── MLFLOW ───────────────────────────────────────────────────────────────────

mlflow.set_tracking_uri(MLFLOW_URI)
try:
    mlflow.pydantic_ai.autolog()
    print("PydanticAI autologging enabled")
except Exception as e:
    print(f"Autologging unavailable: {e}")


# ── COST HOOK ────────────────────────────────────────────────────────────────

USAGE_TRACKER = defaultdict(int)
USAGE_TRACKER['last_cost'] = 0.0
USAGE_TRACKER['last_input'] = 0
USAGE_TRACKER['last_output'] = 0

_orig_create_sync  = openai.resources.chat.completions.Completions.create
_orig_create_async = openai.resources.chat.completions.AsyncCompletions.create


def _extract_cost(response):
    if hasattr(response, 'usage') and response.usage:
        try:
            usage = response.usage.model_dump()
            USAGE_TRACKER['last_input']  = response.usage.prompt_tokens
            USAGE_TRACKER['last_output'] = response.usage.completion_tokens
            cost = usage.get('cost')
            USAGE_TRACKER['last_cost'] = float(cost) if cost is not None else 0.0
            if cost:
                print(f"\n[HOOK] Cost: ${float(cost):.6f}")
        except Exception as e:
            print(f"\n[HOOK] Error: {e}")


def _spy_sync(*args, **kwargs):
    r = _orig_create_sync(*args, **kwargs)
    _extract_cost(r)
    return r


async def _spy_async(*args, **kwargs):
    r = await _orig_create_async(*args, **kwargs)
    _extract_cost(r)
    return r


openai.resources.chat.completions.Completions.create = _spy_sync
openai.resources.chat.completions.AsyncCompletions.create = _spy_async
print("✅ Cost Hook Installed")


# ── DATASET ──────────────────────────────────────────────────────────────────

dataset = get_dataset(dataset_id=DATASET_ID)
RECORDS = dataset.to_dict()
print(f"Loaded {len(RECORDS['records'])} records from dataset ID {DATASET_ID}")


# ── HELPERS ──────────────────────────────────────────────────────────────────

def create_versioned_experiment(base_name: str) -> tuple[str, str, int]:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    versions = []
    for exp in client.search_experiments(filter_string="name LIKE '%'"):
        if exp.name.startswith(f"{base_name}_v"):
            try:
                versions.append(int(exp.name.split("_v")[-1]))
            except ValueError:
                pass
    next_version = max(versions, default=0) + 1
    name = f"{base_name}_v{next_version}"
    exp_id = mlflow.create_experiment(name)
    print(f"Created experiment: {name}")
    return exp_id, name, next_version


def _load_yaml_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "config", "cascade_models.yaml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_models() -> Dict[str, Any]:
    cfg = _load_yaml_config()
    if MODEL_CONFIG_KEY not in cfg:
        raise ValueError(f"'{MODEL_CONFIG_KEY}' not found in cascade_models.yaml")
    return cfg[MODEL_CONFIG_KEY]


def load_judge_config() -> Dict[str, Any]:
    cfg = _load_yaml_config()
    judges = cfg.get("judge_models", {})
    if JUDGE_MODEL_KEY not in judges:
        raise ValueError(f"'{JUDGE_MODEL_KEY}' not found in judge_models")
    return judges[JUDGE_MODEL_KEY]


def make_worker(model_config: Dict[str, Any], model_key: str) -> WorkingAgent:
    return WorkingAgent(
        model_id=model_config['model_name'],
        role_name=model_key,
        system_instruction=SYSTEM_PROMPT,
        config=model_config,
        cascade_tier=model_config.get('tier', 'primary'),
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )


def make_judge() -> ValidatorAgent:
    cfg = load_judge_config()
    print(f"Judge: {cfg.get('short_model_name', cfg['model_name'])}")
    return ValidatorAgent(
        model_name=cfg['model_name'],
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        threshold=ACCEPTABLE_SCORE
    )


def _capture_cost(config: Dict[str, Any], prompt: str, answer: str) -> Dict[str, float]:
    if USAGE_TRACKER['last_cost'] > 0:
        total = USAGE_TRACKER['last_cost']
    else:
        pricing = config.get('pricing', {'input': 0.0, 'output': 0.0})
        in_t  = USAGE_TRACKER['last_input']  or (len(prompt) // 4 + 500)
        out_t = USAGE_TRACKER['last_output'] or (len(answer) // 4)
        total = (in_t / 1_000_000) * pricing['input'] + (out_t / 1_000_000) * pricing['output']
    USAGE_TRACKER['last_cost']   = 0.0
    USAGE_TRACKER['last_input']  = 0
    USAGE_TRACKER['last_output'] = 0
    return {"total_cost": total}


def _run_judge(judge: ValidatorAgent, question: str, answer: str) -> tuple[float, str, str]:
    """Returns (score, verdict, feedback)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio; nest_asyncio.apply()
        result = loop.run_until_complete(judge.evaluate_single(question=question, answer=answer))
    except RuntimeError:
        result = asyncio.run(judge.evaluate_single(question=question, answer=answer))
    if not result:
        return 0.0, "Unknown", ""
    return result.get("score", 0.0), result.get("verdict", "Unknown"), result.get("feedback", "")


def build_refinement_prompt(question: str, prev_answer: str, feedback: str, attempt: int) -> str:
    return REFINEMENT_PROMPT_TEMPLATE.format(
        question=question,
        attempt=attempt + 1,
        prev_answer=prev_answer,
        issue=feedback,
    )


# ── CASCADE WITH SPINNING ─────────────────────────────────────────────────────

def run_cascade(question: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each model level (small → large):
      - Spin up to MAX_LOOPS_PER_LEVEL times feeding judge feedback back into the prompt.
      - Keep spinning as long as the score keeps improving by >= IMPROVEMENT_THRESHOLD.
      - Escalate to the next model when improvement stalls or spins are exhausted.
      - Stop immediately when score >= ACCEPTABLE_SCORE.
    """
    model_names   = list(models.keys())
    judge         = make_judge()
    prompt        = question
    history       = []
    all_answers   = []
    global_attempt = 0

    for level_idx, model_key in enumerate(model_names):
        config     = models[model_key]
        prev_score = -1.0
        last_feedback = ""

        for spin in range(MAX_LOOPS_PER_LEVEL):
            global_attempt += 1
            print(f"   [Level {level_idx+1}/{len(model_names)}] {model_key} — Spin {spin+1}/{MAX_LOOPS_PER_LEVEL}")

            # ── Generate answer ──────────────────────────────────────────────
            agent  = make_worker(config, model_key)
            result = agent.run_sync(prompt)
            answer = result.content
            model_cost = _capture_cost(config, prompt, answer)
            print(f"   → model cost: ${model_cost['total_cost']:.6f}")

            # ── Judge ────────────────────────────────────────────────────────
            score, verdict, feedback = _run_judge(judge, question, answer)
            judge_cost = _capture_cost(load_judge_config(), question + answer, feedback)
            print(f"   → judge cost: ${judge_cost['total_cost']:.6f}")

            improvement  = score - prev_score
            is_good      = score >= ACCEPTABLE_SCORE and verdict == "Valid"
            last_feedback = feedback

            history.append({
                "level":      level_idx + 1,
                "spin":       spin + 1,
                "model":      model_key,
                "model_name": config['model_name'],
                "answer":     answer,
                "score":      score,
                "verdict":    verdict,
                "feedback":   feedback,
                "improvement": round(improvement, 4),
                "passed":     is_good,
                "model_cost": model_cost["total_cost"],
                "judge_cost": judge_cost["total_cost"],
            })
            all_answers.append({
                "answer":     answer,
                "model":      model_key,
                "model_name": config['model_name'],
                "score":      score,
                "verdict":    verdict,
                "level":      level_idx + 1,
                "spin":       spin + 1,
            })

            mlflow.log_metrics({
                f"lvl{level_idx+1}_spin{spin+1}_score":       score,
                f"lvl{level_idx+1}_spin{spin+1}_improvement": round(improvement, 4),
                f"lvl{level_idx+1}_spin{spin+1}_passed":      1 if is_good else 0,
                f"lvl{level_idx+1}_spin{spin+1}_model_cost":  model_cost["total_cost"],
                f"lvl{level_idx+1}_spin{spin+1}_judge_cost":  judge_cost["total_cost"],
            })

            print(f"   → score={score:.4f}  improvement={improvement:+.4f}  verdict={verdict}")

            # ── Accept ───────────────────────────────────────────────────────
            if is_good:
                print(f"   ✅ Accepted at Level {level_idx+1}, Spin {spin+1}")
                total_cost = sum(h["model_cost"] + h["judge_cost"] for h in history)
                return {
                    "answer":     answer,
                    "level":      level_idx + 1,
                    "spin":       spin + 1,
                    "model":      model_key,
                    "model_name": config['model_name'],
                    "success":    True,
                    "history":    history,
                    "total_cost": total_cost,
                    "model_cost": sum(h["model_cost"] for h in history),
                    "judge_cost": sum(h["judge_cost"] for h in history),
                }

            # ── No improvement → escalate ────────────────────────────────────
            if spin > 0 and improvement < IMPROVEMENT_THRESHOLD:
                print(f"   ⬆  improvement {improvement:+.4f} < {IMPROVEMENT_THRESHOLD} — escalating")
                break

            prev_score = score
            # Feed the full judge feedback back into the prompt for the next spin
            prompt = build_refinement_prompt(question, answer, feedback or f"Score {score:.2f}", global_attempt - 1)

        # Carry the last feedback forward to the next model level
        prompt = build_refinement_prompt(question, history[-1]["answer"], last_feedback or f"Score {history[-1]['score']:.2f}", global_attempt - 1)

    # All levels exhausted — return the best answer found
    total_cost  = sum(h["model_cost"] + h["judge_cost"] for h in history)
    valid       = [a for a in all_answers if a["verdict"] == "Valid"]
    best        = max(valid or all_answers, key=lambda x: x["score"])

    return {
        "answer":     best["answer"],
        "level":      best["level"],
        "spin":       best["spin"],
        "model":      best["model"],
        "model_name": best["model_name"],
        "success":    False,
        "history":    history,
        "total_cost": total_cost,
        "model_cost": sum(h["model_cost"] for h in history),
        "judge_cost": sum(h["judge_cost"] for h in history),
    }


# ── EVALUATION ────────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    models       = load_models()
    total_records = len(RECORDS["records"])
    num_questions = NUM_QUESTIONS if NUM_QUESTIONS > 0 else total_records

    exp_id, exp_name, version = create_versioned_experiment("SimpleFlow_Spin_V2")
    mlflow.set_experiment(exp_name)

    print(f"\n{'='*60}")
    print(f"Experiment:         {exp_name}")
    print(f"Models:             {', '.join(models.keys())}")
    print(f"Judge:              {JUDGE_MODEL_KEY}")
    print(f"Acceptable score:   {ACCEPTABLE_SCORE}")
    print(f"Max loops/level:    {MAX_LOOPS_PER_LEVEL}")
    print(f"Improvement thresh: {IMPROVEMENT_THRESHOLD}")
    print(f"{'='*60}\n")

    success_count  = 0
    total_cost_all = 0.0

    for idx, record in enumerate(RECORDS["records"][:num_questions]):
        question    = record["inputs"]["question"]
        question_id = record["inputs"]["question_id"]
        category    = record["inputs"].get("category", "unknown")

        print(f"\n[{idx+1}/{num_questions}] {question_id} ({category})")

        with mlflow.start_run(run_name=question_id):
            mlflow.log_params({
                "question_id":        question_id,
                "category":           category,
                "config_key":         MODEL_CONFIG_KEY,
                "judge_model":        JUDGE_MODEL_KEY,
                "acceptable_score":   ACCEPTABLE_SCORE,
                "max_loops_per_level": MAX_LOOPS_PER_LEVEL,
                "improvement_threshold": IMPROVEMENT_THRESHOLD,
                "version":            version,
            })

            try:
                result = run_cascade(question, models)
                total_cost_all += result["total_cost"]

                with mlflow.start_span(name="best_response", span_type=SpanType.CHAIN) as span:
                    span.set_inputs({"question_id": question_id, "question": question})
                    span.set_outputs({"answer": result["answer"], "model": result["model_name"]})
                    span.set_attributes({"level": result["level"], "spin": result["spin"], "success": result["success"]})

                mlflow.log_metrics({
                    "final_level":   result["level"],
                    "final_spin":    result["spin"],
                    "success":       1 if result["success"] else 0,
                    "answer_length": len(result["answer"]),
                    "total_cost":    result["total_cost"],
                    "model_cost":    result["model_cost"],
                    "judge_cost":    result["judge_cost"],
                })
                mlflow.log_dict(result["history"], "history.json")

                if result["success"]:
                    success_count += 1
                    print(f"   ✅ Done at Level {result['level']}, Spin {result['spin']}")
                else:
                    print(f"   ❌ All levels exhausted — best answer returned")

            except Exception as e:
                print(f"   Error: {e}")
                mlflow.log_metrics({"success": 0})

    rate = success_count / num_questions * 100 if num_questions else 0
    print(f"\n{'='*60}")
    print(f"Results:   {success_count}/{num_questions} ({rate:.1f}%)")
    print(f"Total cost: ${total_cost_all:.4f}")
    print(f"Avg cost:   ${total_cost_all / num_questions:.4f}")
    print(f"MLflow:     {MLFLOW_URI}/#/experiments/{exp_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_evaluation()
