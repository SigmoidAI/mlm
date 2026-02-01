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
from typing import Dict, Any

import mlflow
import mlflow.pydantic_ai
from mlflow.genai.datasets import get_dataset
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
load_dotenv()

import warnings
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
CASCADE_LEVEL = 4
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

def load_cascade_models(level: int = 1) -> Dict[str, Any]:
    """Load cascade_lvl_X from cascade_models.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "cascade_models.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    key = f'cascade_lvl_{level}'
    if key not in config:
        raise ValueError(f"Cascade level {level} not found")
    
    return config[key]


def create_agent(model_config: Dict[str, Any]) -> Agent:
    """Create PydanticAI agent from model config."""
    from openai import AsyncOpenAI
    
    # OpenRouter requires these headers for proper API access
    client = AsyncOpenAI(
        base_url=model_config['endpoint']['api_base_url'],
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        default_headers={
            "HTTP-Referer": "http://localhost:5000", 
            "X-Title": "MLM Cascade Evaluation",   
        }
    )
    
    model = OpenAIChatModel(
        model_config['model_name'],
        provider=OpenAIProvider(openai_client=client),
    )
    
    # -------------------------------------------------------------------------
    # FREE MODELS: Use str (no tool calling)
    # -------------------------------------------------------------------------
    return Agent(
        model,
        output_type=str,
        system_prompt="You are a helpful AI assistant. Provide detailed, accurate answers."
    )
    
    # -------------------------------------------------------------------------
    # PAID MODELS: Use AgentResponse schema for structured output
    # -------------------------------------------------------------------------
    # from models.schemas import AgentResponse
    # return Agent(
    #     model,
    #     output_type=AgentResponse,
    #     system_prompt="You are a helpful AI assistant. Provide detailed, accurate answers."
    # )



# -------------------------------------------------------------------------
# Quality Check - Later this will be replaced by a Judge Model
# -------------------------------------------------------------------------

def check_answer_quality(answer: str, iteration: int, max_iter: int) -> tuple[bool, str]:
    """Simple quality check. Returns (is_acceptable, reason)."""
    if len(answer) < 50:
        return False, "Too short"
    
    if len(answer) > 2000:
        return False, "Too long"
    
    uncertainty = ["i don't know", "i cannot answer", "i'm not sure", "unable to provide"]
    if any(phrase in answer.lower() for phrase in uncertainty):
        return False, "Contains uncertainty"
    
    if iteration >= max_iter - 1:
        return True, "Max iterations reached"
    
    return True, "Passed"


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
    """Try each model until we get an acceptable answer."""
    model_names = list(models.keys())
    max_iter = len(model_names)
    
    prompt = question
    history = []
    last_answer = ""
    
    for i, model_key in enumerate(model_names):
        config = models[model_key]
        print(f"   [{i+1}/{max_iter}] {model_key}")
        
        agent = create_agent(config)
        result = agent.run_sync(prompt)
        # Get answer from result (handles different PydanticAI versions)
        if hasattr(result, 'output'):
            answer = result.output
        elif hasattr(result, 'data'):
            answer = result.data
        else:
            answer = str(result)
        
        # If answer is an object with .content, extract it
        if hasattr(answer, 'content'):
            answer = answer.content
        
        last_answer = answer
        
        is_good, reason = check_answer_quality(answer, i, max_iter)
        
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
    """Run cascade evaluation on dataset."""
    
    models = load_cascade_models(CASCADE_LEVEL)
    num_models = len(models)
    
    exp_name = f"Cascade_Lvl{CASCADE_LEVEL}_{num_models}Models"
    exp_id, full_exp_name, version = create_versioned_experiment(exp_name)
    mlflow.set_experiment(full_exp_name)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {full_exp_name}")
    print(f"Cascade Level: {CASCADE_LEVEL} ({num_models} models)")
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
                "cascade_level": CASCADE_LEVEL,
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
