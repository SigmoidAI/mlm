"""
02_run_simple.py - Iterative Cascade Evaluation

Flow:
1. Load cascade models (small -> large) from YAML config
2. For each question, try Model 1 first
3. Check answer quality (length, clarity)
4. If good -> done. If bad -> refine prompt and try next model
5. Track everything in MLflow
"""

# Standard Library Imports
import asyncio
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# Environment Setup (must happen before other imports)
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Add src to path for direct script execution
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add parent of src (mlm root) for proper package resolution
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Third-Party Imports
import yaml
import mlflow
import mlflow.pydantic_ai
from mlflow.entities import SpanType
from mlflow.genai.datasets import EvaluationDataset, search_datasets
import json
results_jsonl_path = os.path.join(os.path.dirname(__file__), "..", "resources", "results.jsonl")

# Local Imports
from src.agents.pydantic_agent import WorkingAgent, ValidatorAgent

# Suppress Warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
import openai.resources.chat.completions
from collections import defaultdict



try:
    import openai._base_client as _oai_client
    for _wrapper in ['SyncHttpxClientWrapper', 'AsyncHttpxClientWrapper']:
        if hasattr(_oai_client, _wrapper):
            setattr(getattr(_oai_client, _wrapper), '__del__', lambda self: None)
except (ImportError, AttributeError):
    pass  # Older openai versions don't need this fix


# CONSTANTS

# MLflow Configuration (from .env)
MLFLOW_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/")


# Dataset Configuration
DATASET_ID: str = os.getenv("DATASET_ID")
if not DATASET_ID:
    raise RuntimeError("DATASET_ID must be set in your .env file.")
NUM_QUESTIONS: int = int(os.getenv("NUM_QUESTIONS", "0"))  # 0 = use all questions

# Model Configuration
MODEL_CONFIG_KEY: str = "simple_flow"
JUDGE_MODEL_KEY: str = "judge_model_1"  # Options: judge_model_1, judge_model_2, judge_model_3

# Evaluation Thresholds
ACCEPTABLE_SCORE: float = 0.9

# Prompt Templates
SYSTEM_PROMPT: str = "You are a helpful AI assistant. Provide detailed, accurate answers."

REFINEMENT_PROMPT_TEMPLATE: str = """The previous answer did not fully meet the requirements.

QUESTION:
{question}

PREVIOUS ANSWER (Attempt {attempt}):
{prev_answer}

FEEDBACK ON PREVIOUS ANSWER:
{issue}

Please revise your answer to address the feedback above. Ensure your response is thorough, accurate, and covers all aspects of the question. Be clear, detailed, and avoid repeating previous mistakes."""


# MLFLOW INITIALIZATION

mlflow.set_tracking_uri(MLFLOW_URI)

try:
    mlflow.pydantic_ai.autolog()
    print("PydanticAI autologging enabled")
except Exception as e:
    print(f"Autologging unavailable: {e}")


def create_or_get_experiment(experiment_name: str) -> str:
    """Create experiment if it doesn't exist, otherwise return existing."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    return experiment_id


def calculate_openrouter_cost(model_config: Dict[str, Any], input_tokens: int, output_tokens: int) -> Dict[str, float]:
    """Calculate cost from model config pricing.

    Args:
        model_config: Model configuration dict with 'pricing' field
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Dict with 'input_cost', 'output_cost', 'total_cost'
    """
    pricing = model_config.get('pricing', {'input': 0.0, 'output': 0.0})
    input_cost = (input_tokens / 1_000_000) * pricing['input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }


# Global tracker
USAGE_TRACKER = defaultdict(int)
USAGE_TRACKER['last_cost'] = 0.0
USAGE_TRACKER['last_input'] = 0
USAGE_TRACKER['last_output'] = 0
_orig_create_sync = openai.resources.chat.completions.Completions.create
_orig_create_async = openai.resources.chat.completions.AsyncCompletions.create


def _extract_cost_from_response(response):
    """Helper to pull cost from response object"""
    if hasattr(response, 'usage') and response.usage:
        in_tokens = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens

        # OpenRouter 'cost' field is often hidden in 'model_extra' by the OpenAI SDK
        # We convert to dict to find it safely
        try:
            usage_dict = response.usage.model_dump()
            exact_cost = usage_dict.get('cost')

            # Update Global Tracker
            USAGE_TRACKER['last_input'] = in_tokens
            USAGE_TRACKER['last_output'] = out_tokens

            if exact_cost is not None:
                USAGE_TRACKER['last_cost'] = float(exact_cost)
                print(f"\n[HOOK] 💰 OpenRouter reported exact cost: ${float(exact_cost):.6f}")
            else:
                USAGE_TRACKER['last_cost'] = 0.0
                print(f"\n[HOOK] ⚠️ Cost not found. Tokens: {in_tokens} in / {out_tokens} out")
        except Exception as e:
            print(f"\n[HOOK] Error parsing usage: {e}")


# 2. Define Sync Spy
def _spy_create_sync(*args, **kwargs):
    response = _orig_create_sync(*args, **kwargs)
    _extract_cost_from_response(response)
    return response


# 3. Define Async Spy (CRITICAL for PydanticAI)
async def _spy_create_async(*args, **kwargs):
    response = await _orig_create_async(*args, **kwargs)
    _extract_cost_from_response(response)
    return response


# 4. Apply Patches
openai.resources.chat.completions.Completions.create = _spy_create_sync
openai.resources.chat.completions.AsyncCompletions.create = _spy_create_async
print("✅ Universal Cost Hook Installed (Sync + Async)!")
# --- END OF UNIVERSAL COST HOOK ---


# DATASET LOADING


# Load dataset at module level using DATASET_ID and get_dataset
from mlflow.genai.datasets import get_dataset
dataset = get_dataset(dataset_id=DATASET_ID)
RECORDS = dataset.to_dict()
print(f"Loaded {len(RECORDS['records'])} records from dataset ID {DATASET_ID}")


# EXPERIMENT MANAGEMENT

def create_versioned_experiment(base_name: str) -> tuple[str, str, int]:
    """Create experiment with auto-incrementing version.
    
    Args:
        base_name: Base name for the experiment (version suffix will be added).
        
    Returns:
        Tuple of (experiment_id, full_experiment_name, version_number).
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    all_experiments = client.search_experiments(filter_string="name LIKE '%'")
    
    versions = []
    for exp in all_experiments:
        if exp.name.startswith(f"{base_name}_v"):
            try:
                v = int(exp.name.split("_v")[-1])
                versions.append(v)
            except ValueError:
                continue
    
    next_version = max(versions, default=0) + 1
    name = f"{base_name}_v{next_version}"
    
    exp_id = mlflow.create_experiment(name)
    print(f"Created experiment: {name}")
    return exp_id, name, next_version


# MODEL CONFIGURATION

def load_models(config_key: str = MODEL_CONFIG_KEY) -> Dict[str, Any]:
    """Load models from cascade_models.yaml by config key.
    
    Args:
        config_key: Key to look up in the YAML config file.
        
    Returns:
        Dictionary of model configurations.
        
    Raises:
        ValueError: If config_key is not found in the YAML file.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "cascade_models.yaml"
    )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config_key not in config:
        raise ValueError(f"Config key '{config_key}' not found in cascade_models.yaml")
    
    return config[config_key]


def load_judge_config(judge_key: str = JUDGE_MODEL_KEY) -> Dict[str, Any]:
    """Load judge model config from cascade_models.yaml.
    
    Args:
        judge_key: Key identifying the judge model in config.
        
    Returns:
        Dictionary of judge model configuration.
        
    Raises:
        ValueError: If judge_key is not found in the config.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "cascade_models.yaml"
    )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    judge_config = config.get("judge_models", {})
    if judge_key not in judge_config:
        raise ValueError(
            f"Judge key '{judge_key}' not found. Available: {list(judge_config.keys())}"
        )
    
    return judge_config.get(judge_key, {})


# AGENT FACTORIES

def create_agent(model_config: Dict[str, Any], model_key: str) -> WorkingAgent:
    """Create WorkingAgent from model config.
    
    Args:
        model_config: Configuration dictionary for the model.
        model_key: Identifier key for the model.
        
    Returns:
        Configured WorkingAgent instance.
    """
    return WorkingAgent(
        model_id=model_config['model_name'],
        role_name=model_key,
        system_instruction=SYSTEM_PROMPT,
        config=model_config,
        cascade_tier=model_config.get('tier', 'primary'),
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )


def create_judge_agent(judge_key: str = JUDGE_MODEL_KEY) -> ValidatorAgent:
    """Create judge agent from config.
    
    Args:
        judge_key: Key identifying the judge model in config.
        
    Returns:
        Configured ValidatorAgent instance.
    """
    judge_cfg = load_judge_config(judge_key)
    model_name = judge_cfg.get("model_name", "")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    print(f"Using judge: {judge_key} ({judge_cfg.get('short_model_name', model_name)})")
    
    return ValidatorAgent(
        model_name=model_name, 
        api_key=api_key, 
        threshold=ACCEPTABLE_SCORE
    )


# JUDGE EVALUATION

def judge_answer(
    judge_agent: ValidatorAgent, 
    question: str, 
    answer: str
) -> tuple[bool, str]:
    """Evaluate answer quality using judge agent.
    
    Uses ValidatorAgent.evaluate_single() to assess the answer quality.
    
    Args:
        judge_agent: The validator agent to use for evaluation.
        question: The original question.
        answer: The answer to evaluate.
        
    Returns:
        Tuple of (is_good, reason) where is_good is True if score >= ACCEPTABLE_SCORE.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        judge_result = loop.run_until_complete(
            judge_agent.evaluate_single(question=question, answer=answer)
        )
    except RuntimeError:
        judge_result = asyncio.run(
            judge_agent.evaluate_single(question=question, answer=answer)
        )
    
    if not judge_result:
        return False, "Judge failed to evaluate"
    
    score = judge_result.get("score", 0.0)
    verdict = judge_result.get("verdict", "Unknown")
    feedback = judge_result.get("feedback", "")
    
    is_good = score >= ACCEPTABLE_SCORE and verdict == "Valid"
    reason = f"Score: {score:.2f}, Verdict: {verdict}"
    
    if feedback:
        reason += f" - {feedback[:100]}"
    
    return is_good, reason


# PROMPT BUILDING

def build_refinement_prompt(
    question: str,
    prev_answer: str,
    issue: str,
    iteration: int
) -> str:
    """Build prompt that includes previous failed attempt.

    Args:
        question: The original question.
        prev_answer: The previous answer that was rejected.
        issue: Description of why the previous answer was rejected.
        iteration: Current iteration number (0-indexed).

    Returns:
        Formatted refinement prompt string.
    """
    return REFINEMENT_PROMPT_TEMPLATE.format(
        question=question,
        attempt=iteration + 1,
        prev_answer=prev_answer,
        issue=issue
    )


# CASCADE LOGIC

def run_cascade(
    question: str, 
    question_id: str, 
    models: Dict[str, Any],
    judge_key: str = JUDGE_MODEL_KEY
) -> Dict[str, Any]:
    """Try each model in cascade until judge approves an answer.
    
    Args:
        question: The question to answer.
        question_id: Unique identifier for the question.
        models: Dictionary of model configurations to try in order.
        
    Returns:
        Dictionary containing:
            - answer: The final answer (best or last)
            - iterations: Number of models tried
            - model: The model that produced the answer
            - success: Whether an acceptable answer was found
            - history: List of all attempts with details
    """
    model_names = list(models.keys())
    max_iter = len(model_names)
    prompt = question
    history = []
    answers_with_scores = []

    judge_agent = create_judge_agent()


    for i, model_key in enumerate(model_names):
        config = models[model_key]
        print(f"   [{i+1}/{max_iter}] {model_key}")

        agent = create_agent(config, model_key)
        result = agent.run_sync(prompt)
        answer = result.content

        if USAGE_TRACKER['last_cost'] > 0:
            total_real_cost = USAGE_TRACKER['last_cost']

            # Create a dict so the rest of your script (history logging) stays happy
            model_cost = {
                "input_cost": 0.0,  # Unknown split, but total is correct
                "output_cost": 0.0,
                "total_cost": total_real_cost
            }
            print(f"   -> Cost (Official): ${total_real_cost:.6f}")

        else:
            # Fallback Calculation
            real_input = USAGE_TRACKER['last_input'] or ((len(prompt) // 4) + 500)
            real_output = USAGE_TRACKER['last_output'] or (len(answer) // 4)

            model_cost = calculate_openrouter_cost(config, real_input, real_output)

            # FIXED PRINT STATEMENT BELOW:
            print(f"   -> Cost (Calculated): ${model_cost['total_cost']:.6f}")

        USAGE_TRACKER['last_cost'] = 0.0
        USAGE_TRACKER['last_input'] = 0
        USAGE_TRACKER['last_output'] = 0
        # Get judge score and verdict
        try:
            judge_result = None
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            judge_result = loop.run_until_complete(
                judge_agent.evaluate_single(question=question, answer=answer)
            )
        except RuntimeError:
            judge_result = asyncio.run(
                judge_agent.evaluate_single(question=question, answer=answer)
            )

        score = judge_result.get("score", 0.0) if judge_result else 0.0
        verdict = judge_result.get("verdict", "Unknown") if judge_result else "Unknown"
        feedback = judge_result.get("feedback", "") if judge_result else ""

        # Capture judge cost from tracker
        if USAGE_TRACKER['last_cost'] > 0:
            judge_cost = {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": USAGE_TRACKER['last_cost']
            }
            print(f"   -> Judge Cost (Official): ${judge_cost['total_cost']:.6f}")
        else:
            # Fallback calculation
            judge_cfg = load_judge_config(judge_key)
            judge_input = int((len(question.split()) + len(answer.split())) * 1.3)
            judge_output = int((len(feedback.split()) if feedback else 50) * 1.3)
            judge_cost = calculate_openrouter_cost(judge_cfg, judge_input, judge_output)
            print(f"   -> Judge Cost (Calculated): ${judge_cost['total_cost']:.6f}")


        is_good = score >= ACCEPTABLE_SCORE and verdict == "Valid"
        reason = f"Score: {score:.2f}, Verdict: {verdict}"
        if feedback:
            reason += f" - {feedback[:100]}"

        history.append({
            "iteration": i + 1,
            "model": model_key,
            "model_name": config['model_name'],
            "answer": answer,
            "passed": is_good,
            "reason": reason,
            "score": score,
            "verdict": verdict,
            "model_cost": model_cost,
            "judge_cost": judge_cost,
            "iteration_total_cost": model_cost["total_cost"] + judge_cost["total_cost"]
        })
        answers_with_scores.append({
            "answer": answer,
            "model": model_key,
            "model_name": config['model_name'],
            "score": score,
            "iteration": i + 1,
            "verdict": verdict
        })

        mlflow.log_metrics({
            f"iter_{i + 1}_length": len(answer),
            f"iter_{i + 1}_passed": 1 if is_good else 0,
            f"iter_{i + 1}_model_cost": model_cost["total_cost"],
            f"iter_{i + 1}_judge_cost": judge_cost["total_cost"]
        })

        print(f"   -> {reason}")

        if is_good:
            break
        else:
            prompt = build_refinement_prompt(question, answer, reason, i)

    total_cost = sum(h["iteration_total_cost"] for h in history)

    # Find the best valid answer (verdict == 'Valid'), highest score
    valid_answers = [a for a in answers_with_scores if a["verdict"] == "Valid"]
    if valid_answers:
        best = max(valid_answers, key=lambda x: x["score"])
        success = best["score"] >= ACCEPTABLE_SCORE
    else:
        # If none are valid, fall back to highest score overall
        best = max(answers_with_scores, key=lambda x: x["score"]) if answers_with_scores else {"answer": "",
                                                                                               "model": model_names[-1],
                                                                                               "model_name": models[
                                                                                                   model_names[-1]][
                                                                                                   "model_name"],
                                                                                               "score": 0.0,
                                                                                               "iteration": max_iter,
                                                                                               "verdict": "Unknown"}
        success = False
    return {
        "answer": best["answer"],
        "iterations": best["iteration"],
        "model": best["model"],
        "model_name": best["model_name"],
        "success": success,
        "history": history,
        "total_cost": total_cost,
        "cost_breakdown": {
            "total_model_cost": sum(h["model_cost"]["total_cost"] for h in history),
            "total_judge_cost": sum(h["judge_cost"]["total_cost"] for h in history)
        }
    }


# MAIN EVALUATION

def run_evaluation() -> None:
    """Run simple flow evaluation on dataset."""
    models = load_models(MODEL_CONFIG_KEY)
    num_models = len(models)
    
    exp_name = f"SimpleFlow_V2"
    exp_id, full_exp_name, version = create_versioned_experiment(exp_name)
    mlflow.set_experiment(full_exp_name)
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"Experiment: {full_exp_name}")
    print(f"Config: {MODEL_CONFIG_KEY} ({num_models} models)")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Judge: {JUDGE_MODEL_KEY}")
    print(f"Acceptable Score: {ACCEPTABLE_SCORE}")
    print(f"{'='*60}\n")
    
    success_count = 0
    total_iterations = 0
    total_cost_all = 0.0
    total_records = len(RECORDS["records"])
    num_questions = NUM_QUESTIONS if NUM_QUESTIONS > 0 else total_records
    

    for idx, record in enumerate(RECORDS["records"][:num_questions]):
        question = record["inputs"]["question"]
        question_id = record["inputs"]["question_id"]
        category = record["inputs"].get("category", "unknown")

        print(f"[{idx+1}/{num_questions}] {question_id} ({category})")

        with mlflow.start_run(run_name=question_id):
            mlflow.log_params({
                "question_id": question_id,
                "category": category,
                "config_key": MODEL_CONFIG_KEY,
                "judge_model": JUDGE_MODEL_KEY,
                "acceptable_score": ACCEPTABLE_SCORE,
                "num_models": num_models,
                "version": version
            })

            try:
                result = run_cascade(question, question_id, models)

                total_cost_all += result["total_cost"]

                best_response = {
                    "score": result.get("score", 0.0),  # Won't exist, will be 0.0
                    "verdict": "Valid" if result["success"] else "Invalid",
                    "iteration": result["iterations"],
                    "model": result["model"],
                    "model_name": result["model_name"]
                }

                # CREATE MANUAL TRACE FOR BEST RESPONSE
                with mlflow.start_span(name="best_response", span_type=SpanType.CHAIN) as span:
                    span.set_inputs({
                        "question_id": question_id,
                        "question": question
                    })
                    span.set_outputs({
                        "answer": result["answer"],
                        "model": result["model_name"]
                    })
                    span.set_attributes({
                        "score": best_response["score"],
                        "verdict": best_response["verdict"],
                        "iteration": result["iterations"],
                        "success": result["success"]
                    })
                mlflow.log_metrics({
                    "iterations": result["iterations"],
                    "success": 1 if result["success"] else 0,
                    "answer_length": len(result["answer"]),
                    "total_cost": result["total_cost"],
                    "model_cost": result["cost_breakdown"]["total_model_cost"],
                    "judge_cost": result["cost_breakdown"]["total_judge_cost"]

                })
                mlflow.log_dict(result["history"], "history.json")

                # JSONL logging
                jsonl_entry = {
                    "question_id": question_id,
                    "category": category,
                    "question": question,
                    "answer": result["answer"],
                    "iterations": result["iterations"],
                    "model": result["model"],
                    "model_name": result["model_name"],
                    "success": result["success"],
                    "total_cost": result["total_cost"],
                    "history": result["history"]
                }
                # with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
                #     f_jsonl.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

                if result["success"]:
                    success_count += 1
                    print(f"   Final: {result['answer'][:80]}...\n")
                else:
                    print(f"   Exhausted all models\n")

                total_iterations += result["iterations"]

            except Exception as e:
                print(f"   Error: {e}\n")
                mlflow.log_metrics({"success": 0})
    
    # Print results summary
    avg_iter = total_iterations / num_questions if num_questions > 0 else 0
    rate = (success_count / num_questions * 100) if num_questions > 0 else 0
    
    print(f"{'='*60}")
    print(f"Results: {success_count}/{num_questions} ({rate:.1f}%)")
    print(f"Avg iterations: {avg_iter:.2f}")
    print(f"Total cost: ${total_cost_all:.4f}")
    print(f"Avg cost per question: ${total_cost_all / num_questions:.4f}")
    print(f"MLflow: {MLFLOW_URI}/#/experiments/{exp_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_evaluation()
