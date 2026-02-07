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
from mlflow.genai.datasets import EvaluationDataset, search_datasets

# Local Imports
from src.agents.pydantic_agent import WorkingAgent, ValidatorAgent

# Suppress Warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

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
DATASET_EXPERIMENT_NAME: str = "DATASET_Arena_Hard_V2"
MLFLOW_DATASET_NAME: str = "arena_hard_v2_0"

# Model Configuration
MODEL_CONFIG_KEY: str = "simple_flow"
JUDGE_MODEL_KEY: str = "judge_model_1"  # Options: judge_model_1, judge_model_2, judge_model_3

# Evaluation Thresholds
ACCEPTABLE_SCORE: float = 0.9

# Prompt Templates
SYSTEM_PROMPT: str = "You are a helpful AI assistant. Provide detailed, accurate answers."

REFINEMENT_PROMPT_TEMPLATE: str = """Previous answer was not satisfactory.

QUESTION:
{question}

PREVIOUS ANSWER (Attempt {attempt}):
{prev_answer}

ISSUE: {issue}

Please provide a better, more complete answer."""


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


# DATASET LOADING

def get_experiment_dataset(
    experiment_name: str = DATASET_EXPERIMENT_NAME, 
    dataset_name: str = MLFLOW_DATASET_NAME
) -> Optional[Dict[str, Any]]:
    """Retrieve experiment dataset by MLFlow experiment name.

    Args:
        experiment_name: Name of the MLFlow Experiment containing a dataset.
        dataset_name: Name of the Dataset in MLFlow Experiment.

    Returns:
        Dict representation of the dataset from MLFlow Experiment, or None if not found.
    """
    experiment_id = create_or_get_experiment(experiment_name=experiment_name)
    print(f"Looking for dataset in experiment ID: {experiment_id}")
    
    datasets_list: list[EvaluationDataset] = search_datasets(experiment_ids=experiment_id)
    if not datasets_list:
        print(f"No datasets found in experiment ID: {experiment_id}")
        return None
    
    print(f"Found {len(datasets_list)} dataset(s) in experiment")
    for dataset in datasets_list:
        if dataset.name == dataset_name:
            print(f"Dataset '{dataset_name}' found!")
            return dataset.to_dict()
    
    print(f"Dataset '{dataset_name}' not found in experiment")
    return None


# Load dataset at module level
_dataset_dict = get_experiment_dataset()
if _dataset_dict is None:
    raise RuntimeError(
        f"Could not load dataset '{MLFLOW_DATASET_NAME}' "
        f"from experiment '{DATASET_EXPERIMENT_NAME}'"
    )
RECORDS = _dataset_dict
print(f"Loaded {len(RECORDS['records'])} records")


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
    models: Dict[str, Any]
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
        
        print(f"   -> {reason}")
        
        if is_good:
            return {
                "answer": answer, 
                "iterations": i + 1, 
                "model": model_key, 
                "success": True, 
                "history": history
            }
        
        prompt = build_refinement_prompt(question, answer, reason, i)
    
    return {
        "answer": last_answer, 
        "iterations": max_iter, 
        "model": model_names[-1], 
        "success": False, 
        "history": history
    }


# MAIN EVALUATION

def run_evaluation() -> None:
    """Run simple flow evaluation on dataset."""
    models = load_models(MODEL_CONFIG_KEY)
    # num_models = len(models)
    num_models = 2
    
    exp_name = f"SimpleFlow_{num_models}Models"
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
    num_questions = len(RECORDS["records"])
    
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
    
    # Print results summary
    avg_iter = total_iterations / num_questions if num_questions > 0 else 0
    rate = (success_count / num_questions * 100) if num_questions > 0 else 0
    
    print(f"{'='*60}")
    print(f"Results: {success_count}/{num_questions} ({rate:.1f}%)")
    print(f"Avg iterations: {avg_iter:.2f}")
    print(f"MLflow: {MLFLOW_URI}/#/experiments/{exp_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_evaluation()
