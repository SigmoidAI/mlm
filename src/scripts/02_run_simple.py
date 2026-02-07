"""
02_run_simple.py - Iterative Cascade Evaluation

Flow:
1. Load cascade models (small -> large) from YAML config
2. For each question, try Model 1 first
3. Check answer quality (length, clarity)
4. If good -> done. If bad -> refine prompt and try next model
5. Track everything in MLflow
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load .env file BEFORE importing modules that need env vars
load_dotenv()

# Add src to path for direct script execution
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add parent of src (mlm root) for proper package resolution
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import warnings

import mlflow.pydantic_ai
import yaml
from mlflow.genai.datasets import get_dataset

import mlflow

# Import WorkingAgent from agents module
from src.agents.pydantic_agent import WorkingAgent, ValidatorAgent

warnings.filterwarnings("ignore", category=ResourceWarning)

try:
    import openai._base_client as _oai_client
    for _wrapper in ['SyncHttpxClientWrapper', 'AsyncHttpxClientWrapper']:
        if hasattr(_oai_client, _wrapper):
            setattr(getattr(_oai_client, _wrapper), '__del__', lambda self: None)
except (ImportError, AttributeError):
    pass  # Older openai versions don't need this fix


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

MLFLOW_URI = "http://127.0.0.1:5000"
MODEL_CONFIG_KEY = "simple_flow"  # Use simple_flow models
DATASET_ID = "d-bb04783ae0654ead9e35c474580d71b2"


mlflow.set_tracking_uri(MLFLOW_URI)

try:
    mlflow.pydantic_ai.autolog()
    print("PydanticAI autologging enabled")
except Exception as e:
    print(f"Autologging unavailable: {e}")


# -------------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------------

dataset = get_dataset(dataset_id=DATASET_ID)
records = dataset.to_dict()
print(f"Loaded {len(records['records'])} records")


# -------------------------------------------------------------------------
# MLflow Helpers
# -------------------------------------------------------------------------

def create_versioned_experiment(base_name: str) -> tuple[str, str, int]:
    """Create experiment with auto-incrementing version."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    existing = client.search_experiments(
        filter_string=f"name LIKE '{base_name}_v%'",
        order_by=["creation_time DESC"]
    )
    
    versions = []
    for exp in existing:
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


# -------------------------------------------------------------------------
# Model Configuration
# -------------------------------------------------------------------------

def load_models(config_key: str = "simple_flow") -> Dict[str, Any]:
    """Load models from cascade_models.yaml by config key."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "cascade_models.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config_key not in config:
        raise ValueError(f"Config key '{config_key}' not found in cascade_models.yaml")
    
    return config[config_key]


def create_agent(model_config: Dict[str, Any], model_key: str) -> WorkingAgent:
    """Create WorkingAgent from model config."""
    return WorkingAgent(
        model_id=model_config['model_name'],
        role_name=model_key,
        system_instruction="You are a helpful AI assistant. Provide detailed, accurate answers.",
        config=model_config,
        cascade_tier=model_config.get('tier', 'primary'),
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )




# -------------------------------------------------------------------------
# Judge Agent Quality Check
# -------------------------------------------------------------------------

import yaml

def load_judge_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "cascade_models.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    judge_config = config.get("judge_models", {})
    judge_key = "judge_model_1"
    return judge_config.get(judge_key, {})

def create_judge_agent():
    judge_cfg = load_judge_config()
    model_name = judge_cfg.get("model_name", "")
    endpoint = judge_cfg.get("endpoint", {})
    api_key = endpoint.get("api_key", "")
    return ValidatorAgent(model_name=model_name, api_key=api_key)

def judge_answer(judge_agent, question, answer):
    """Judge agent evaluates answer and returns (is_good, reason)."""
    from src.models.schemas import Prompt, AgentResponse
    prompt = Prompt(content=question, model_tier="simple")
    answers = {"worker": AgentResponse(content=answer, author_id="worker")}
    judge_result = judge_agent.evaluate_multiple(prompt=prompt, question=question, answers=answers)
    # judge_result should be a dict with 'best_answer' and 'best_confidence_score'
    if not judge_result or "best_answer" not in judge_result:
        return False, "Judge failed"
    best_score = judge_result["best_answer"].get("best_confidence_score", 0)
    reason = judge_result["best_answer"].get("best_reason", "")
    is_good = best_score >= 0.9
    return is_good, reason or f"Score: {best_score}"


def build_refinement_prompt(question: str, prev_answer: str, issue: str, iteration: int) -> str:
    """Build prompt that includes previous failed attempt."""
    return f"""Previous answer was not satisfactory.

QUESTION:
{question}

PREVIOUS ANSWER (Attempt {iteration + 1}):
{prev_answer}

ISSUE: {issue}

Please provide a better, more complete answer."""


# -------------------------------------------------------------------------
# Cascade Logic
# -------------------------------------------------------------------------

def run_cascade(question: str, question_id: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """Try each model until judge says answer is good."""
    model_names = list(models.keys())
    max_iter = len(model_names)
    prompt = question
    history = []
    last_answer = ""
    judge_agent = create_judge_agent()
    for i, model_key in enumerate(model_names):
        config = models[model_key]
        print(f"   [{i+1}/{max_iter}] {model_key}")
        agent = create_agent(config, model_key)
        result = agent.run_sync(prompt)
        answer = result.content
        last_answer = answer
        is_good, reason = judge_answer(judge_agent, question, answer)
        history.append({
            "iteration": i + 1,
            "model": model_key,
            "answer": answer,
            "passed": is_good,
            "reason": reason
        })
        mlflow.log_metrics({
            f"iter_{i+1}_length": len(answer),
            f"iter_{i+1}_passed": 1 if is_good else 0
        })
        if is_good:
            print(f"   -> {reason}")
            return {"answer": answer, "iterations": i + 1, "model": model_key, "success": True, "history": history}
        print(f"   -> {reason}")
        prompt = build_refinement_prompt(question, answer, reason, i)
    return {"answer": last_answer, "iterations": max_iter, "model": model_names[-1], "success": False, "history": history}


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def run_evaluation():
    """Run simple flow evaluation on dataset."""
    
    models = load_models(MODEL_CONFIG_KEY)
    num_models = len(models)
    
    exp_name = f"SimpleFlow_{num_models}Models"
    exp_id, full_exp_name, version = create_versioned_experiment(exp_name)
    mlflow.set_experiment(full_exp_name)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {full_exp_name}")
    print(f"Config: {MODEL_CONFIG_KEY} ({num_models} models)")
    print(f"Models: {', '.join(models.keys())}")
    print(f"{'='*60}\n")
    
    success_count = 0
    total_iterations = 0
    num_questions = 2  # Limit for testing; set to len(records["records"]) for full run
    
    for idx, record in enumerate(records["records"][:num_questions]):
        question = record["inputs"]["question"]
        question_id = record["inputs"]["question_id"]
        category = record["inputs"].get("category", "unknown")
        
        print(f"[{idx+1}/{num_questions}] {question_id} ({category})")
        
        with mlflow.start_run(run_name=question_id):
            mlflow.log_params({
                "question_id": question_id,
                "category": category,
                "config_key": MODEL_CONFIG_KEY,
                "num_models": num_models,
                "version": version
            })
            
            try:
                result = run_cascade(question, question_id, models)
                
                mlflow.log_metrics({
                    "iterations": result["iterations"],
                    "success": 1 if result["success"] else 0,
                    "answer_length": len(result["answer"])
                })
                mlflow.log_dict(result["history"], "history.json")
                
                if result["success"]:
                    success_count += 1
                    print(f"   Final: {result['answer'][:80]}...\n")
                else:
                    print(f"   Exhausted all models\n")
                
                total_iterations += result["iterations"]
                
            except Exception as e:
                print(f"   Error: {e}\n")
                mlflow.log_metrics({"success": 0})
    
    avg_iter = total_iterations / num_questions if num_questions > 0 else 0
    rate = (success_count / num_questions * 100) if num_questions > 0 else 0
    
    print(f"{'='*60}")
    print(f"Results: {success_count}/{num_questions} ({rate:.1f}%)")
    print(f"Avg iterations: {avg_iter:.2f}")
    print(f"MLflow: {MLFLOW_URI}/#/experiments/{exp_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_evaluation()
