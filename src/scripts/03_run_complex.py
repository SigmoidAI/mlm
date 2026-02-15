import argparse
import asyncio
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Coroutine, Optional, Union
import openai
import openai.resources.chat.completions
from collections import defaultdict

import json_repair
from loguru import logger
from mlflow.entities import SpanType
from mlflow.genai.datasets import EvaluationDataset, search_datasets  # get_dataset
from pydantic_ai import AgentRunError

import mlflow

from ..agents.pydantic_agent import ValidatorAgent, WorkingAgent
from ..config.make_config import make_config
from ..datasets.loader import create_or_get_experiment
from ..models.schemas import AgentResponse, Prompt

logger.info("Configuring defined variables...")

# * HARDCODED VALUES SPECIFIC TO COMPLEX MODEL WORKFLOW
# * ASSUMES:
# * 1. USER PROMPT IS GIVEN (IN THIS CONTEXT, TAKEN DYNAMICALLY FROM EXPERIMENT DATASET)
# * 2. SELECTED COMPLEXITY IS HIGH
IS_COMPLEX: bool = True

# PROMPTS
# * WORKERS PROMPTS

# LOW STRICT
WORKING_AGENT_SYSTEM_PROMPT_1: str = "You are a helpful AI assistant. Provide detailed, accurate answers."

# VERY STRICT
WORKING_AGENT_SYSTEM_PROMPT_2: str = """
You are a precise, focused AI assistant. Your primary objective is to provide direct, accurate answers that strictly address the user's specific question.

## Core Principles:

1. **Strict Relevance**: Answer ONLY what is asked. Do not provide tangential information, background context, or related topics unless explicitly requested.

2. **Precision Over Elaboration**: Favor concise, exact answers over comprehensive explanations. If the question is specific, your answer must be equally specific.

3. **Question Adherence**: Continuously evaluate whether each sentence directly serves the user's question. Eliminate any content that strays from the core query.

4. **Factual Accuracy**: Provide verifiable, accurate information. If uncertain, state the limitation rather than speculating.

5. **Structured Clarity**: When multiple points are needed, structure them logically and maintain tight focus on the question's scope.

## Response Guidelines:

- Begin with the most direct answer to the question
- Avoid preambles, disclaimers, or unnecessary context unless critical
- Do not expand into related topics the user didn't ask about
- Keep examples minimal and directly relevant
- Resist the urge to be comprehensive; be surgically precise instead

## Handling Refinement Reviews:

When you receive refinement feedback from other assistants:
- Critically evaluate the feedback against the original question's intent
- Apply only refinements that enhance precision and relevance to the specific query
- Reject suggestions that add tangential content or reduce focus
- Maintain your analytical independence—refinements are advisory, not mandatory
- Prioritize the user's original question over the refinement suggestions

## Output Standard:

Every response must pass this test: "Does every sentence directly answer the user's specific question?" If not, remove it.

Stay focused. Stay precise. Answer the question asked, nothing more, nothing less.
"""

# MEDIUM STRICT
WORKING_AGENT_SYSTEM_PROMPT_3: str = """
You are a focused and helpful AI assistant. Your objective is to provide accurate, relevant answers that directly address the user's question while including necessary context and clarity.

## Core Principles:

1. **Relevant Focus**: Prioritize answering the specific question asked. Include helpful context or examples when they enhance understanding, but avoid unrelated tangents.

2. **Balanced Depth**: Provide enough detail to be truly helpful without over-explaining. Aim for clarity and completeness within the question's scope.

3. **Question-Centered**: Ensure your answer serves the user's actual need. Add clarifying information when it directly supports the answer.

4. **Accuracy First**: Provide verifiable, correct information. Acknowledge limitations when appropriate.

5. **Clear Structure**: Organize information logically. Use examples and explanations that illuminate the answer.

## Response Guidelines:

- Lead with the direct answer, then provide supporting details
- Include brief context when it aids understanding
- Use relevant examples to clarify concepts
- Keep explanations proportional to the question's complexity
- Avoid wandering into distantly related topics

## Refinement Integration:

When refining your previous response:
- Seamlessly incorporate improvements into your answer
- Present the refined version as a natural, cohesive response
- Do not reference previous versions, suggestions, or the refinement process
- Integrate additions and modifications as if this is your first, complete answer
- Maintain consistency in tone and style throughout

## Output Standard:

Your response should be comprehensive enough to be helpful while remaining clearly connected to the user's specific question. Balance thoroughness with relevance.

Be helpful. Be clear. Stay on point.
"""

# * NEXT LEVEL CASCADE PROMPT
NEXT_CASCADE_LEVEL_PROMPT: str = "Previous cascade level worker agents did not succeed to answer properly user question/prompt. Analyze the question and their answers and generate better results:"

# * JUDGE PROMPTS
JUDGE_PROMPT_1: str = """
Worker agents provided the following answers to the initial question.
"""

JUDGE_PROMPT_2: str = """
Worker agents provided the following answers to the initial question. 
Analyze the question and their answers and start a voting process for the best answer.
Be impartial and serve as an objective critic.
Provide the response strictly in the following format:
```json
"evaluation": {
    "question": <question>,
    "best_answer": {
        "best_worker_model_id": <best_worker_model_<id>>,
        "best_confidence_score": <best_answer_confidence_score_float_4_decimals>,
        "best_reason": <best_reason>    
    },
    "individual_reviews": {
        "worker_model_<id>": {
            "confidence_score": <answer_confidence_score_float_4_decimals>,
            "reason": <reason>,
        },
        "worker_model_<id>": {
            "confidence_score": <answer_confidence_score_float_4_decimals>,
            "reason": <reason>,
        },
    }
    ...
}
```
"""

JUDGE_PROMPT_3: str = """
Worker agents provided the following answers to the initial question. 

Analyze the question and their answers and start a voting process for the best answer.

Be impartial and serve as an objective critic.

When evaluating take into consideration:
## Evaluation Criteria:
1. Identify the specific task requested in the input
2. Determine all requirements and constraints mentioned
3. Check if the output fulfills each requirement
4. Verify the output format matches any specified format
5. Assess completeness - are all parts of the task done?
6. Validate the quality of task execution

Provide in individual reviews a reason/feedback why the answer was graded with that specific score. 
Make sure these reasons/feedback are independent between other workers reviews.
Evaluation Instructions:

Evaluate whether the output successfully completes the requested task.

Assign a score from 0.0000 to 1.0000 where:

- 1.0000 = Task fully completed with all requirements met.
- range between 0.0000 and 1.0000 = Determined how partially/ what grade of the task completion as a float 4 decimals after point.
- 0.0000 = Task not completed or attempted.

Provide the response strictly in the following format:
```json
"evaluation": {
    "question": <question>,
    "best_answer": {
        "best_worker_model_id": <best_worker_model_<id>>,
        "best_confidence_score": <best_answer_confidence_score_float_4_decimals>,
        "best_reason": <best_reason>    
    },
    "individual_reviews": {
        "worker_model_<id>": {
            "confidence_score": <answer_confidence_score_float_4_decimals>,
            "reason": <reason>,
        },
        "worker_model_<id>": {
            "confidence_score": <answer_confidence_score_float_4_decimals>,
            "reason": <reason>,
        },
    }
    ...
}
```
"""

# CONFIGURATION

# USER PROMPT (IF DESIRED) # ! CONFIGURABLE
USER_PROMPT: str = {
    "dataset_record_id": uuid.uuid4().hex,
    "inputs": {
        "question_id": "user_prompt_1",
        "question": "What is the capital of Great Britain?",
        "category": "user_prompt"
    }
}

# NUMBER OF QUESTIONS TO BE EVALUATED/ASKED TO THE CASCADE.
NUM_MAX_QUESTIONS: int = 250  # ! CONFIGURABLE

# CASCADE MODELS CONFIG
CASCADE_MODELS_CONFIG: dict[str, str] = make_config()
COMPLEX_RUN_CONFIG_KEY: str = "cascade_complex_run"

# JUDGE MODELS CONFIG
JUDGE_MODELS_CONFIG_KEY: str = "judge_models"
JUDGE_MODEL_KEY: str = "judge_model_1"  # ! CONFIGURABLE
ACCEPTABLE_SCORE: float = 0.90  # ! CONFIGURABLE

# CASCADE LEVEL
MAX_CASCADE_LEVEL: int = 5  # ! CONFIGURABLE (1 <= value <= 5)

# MLFLOW
MLFLOW_TRACKING_URI: str = os.getenv(key="MLFLOW_TRACKING_URI", default="http://127.0.0.1:5000")
EXPERIMENT_NAME: str = f"complex_workflow_run_max_{NUM_MAX_QUESTIONS}"  # ! CONFIGURABLE

# EXPERIMENT DATASET # ! CONFIGURABLE
# * Arena Hard Auto v1.0
# MLFLOW_DATASET_EXPERIMENT_NAME: str = "DATASET_Arena_Hard"
# MLFLOW_DATASET_NAME: str = "arena_hard_auto"

# * Arena Hard Auto v2.0
MLFLOW_DATASET_EXPERIMENT_NAME: str = "DATASET_Arena_Hard_V2"
MLFLOW_DATASET_NAME: str = "arena_hard_v2_0"

# RESULTS
RESULTS_PATH: Path = Path(__file__).resolve().parent.parent / "results"  # ! CONFIGURABLE

def init_dirs(paths: list[Union[Path, str]]) -> None:
    """Helper method to initialize a list of directories.

    Args:
        paths (list[Union[Path, str]]): list of paths to the new directory.
    """
    logger.info(f"Attempting to initialize and create {len(paths)} directories...")
    for path in paths:
        path_obj: Path = Path(path)
        logger.info(f"Creating directory at: {path_obj}")
        if not path_obj.exists():
            try:
                path_obj.mkdir(parents=True)
            except OSError as e:
                logger.error(f"Error during creation of directory at: {path}: {e}")
            else:
                logger.success(f"Directory at: {path} was created succesfully.")
        else:
            logger.success(f"Directory at: {path} already existing. Skipping...")


def perform_initial_mlflow_setup(mlflow_uri: str) -> None:
    """Function to set up basic MLFlow workflows.

    Args:
        mlflow_uri (str): URI of the MLFlow server.
    """
    logger.info(f"Attempting to set MLFlow Tracking URI to: {mlflow_uri}.")
    mlflow.set_tracking_uri(uri=mlflow_uri)
    
    try:
        mlflow.pydantic_ai.autolog()
        logger.success("PydanticAI Autologging enabled.")
    except Exception as e:
        logger.warning(f"Could not enable PydanticAI Autologging: {e}")
    
    set_up_tracking_uri: str = mlflow.get_tracking_uri()
    if set_up_tracking_uri != mlflow_uri:        
        logger.warning(f"MLFlow Tracking URI that was set up is different from the one provided. Set up URI: \"{set_up_tracking_uri}\", provided: \"{mlflow_uri}\".")    
        return
    
    logger.success(f"MLFlow Tracking URI set to: {set_up_tracking_uri}.")
    
def get_experiment_dataset(experiment_name: str = MLFLOW_DATASET_EXPERIMENT_NAME, dataset_name: str = MLFLOW_DATASET_NAME) -> Optional[dict[str, Any]]:
    """Retrieves experiment dataset by its MLFlow experiment name.

    Args:
        experiment_name (str, optional): Name of the MLFlow Experiment containing a dataset. Defaults to MLFLOW_DATASET_EXPERIMENT_NAME = DATASET_Arena_Hard_V2.
        dataset_name (str, optional): Name of the Dataset in MLFlow Experiment. Defaults to MLFLOW_DATASET_NAME = arena_hard_v2_0.

    Returns:
        Optional[dict[str, Any]]: Dict representation of the dataset from MLFlow Experiment. If no dataset is found, returns None.
    """
    experiment_id: str = create_or_get_experiment(experiment_name=experiment_name)
    logger.info(f"Looking for MLFlow Experiment Dataset with Experiment ID: {experiment_id}.")
    
    # TODO: Check another way to retrieve dataset by experiment ID.
    datasets_list: list[EvaluationDataset] = search_datasets(filter_string=f"name = '{dataset_name}'", experiment_ids=experiment_id)
    
    if not datasets_list:
        logger.warning(f"No datasets found in MLFlow Experiment with Experiment ID: {experiment_id}")
        return None
    
    logger.success(f"Found: {len(datasets_list)} datasets in MLFlow Experiment with Experiment ID: {experiment_id}")
    logger.success(f"Datasets: {datasets_list}")
    for dataset in datasets_list:
        if dataset.name == dataset_name:
            logger.success(f"Dataset called: {dataset_name} found in MLFlow Experiment with Experiment ID: {experiment_id}")
            return dataset.to_dict()
    else:
        logger.warning(f"No dataset called: {dataset_name} found in MLFlow Experiment with Experiment ID: {experiment_id}")

def get_question_records_from_dataset(dataset: list, record_idx: Union[range, int]) -> Union[list, None]:
    """Retrieve question record(s) from dataset by index or range.

    Args:
        dataset (list): List of dataset records.
        record_idx (Union[range, int]): Single index (int) or range of indices (range).

    Returns:
        Union[list, None]: list of record(s), None if invalid.
    """
    dataset_size = len(dataset)
    
    # Validate dataset is not empty
    if dataset_size == 0:
        logger.error("Dataset is empty.")
        return None
    
    # Handle integer index
    if isinstance(record_idx, int):
        logger.info(f"Looking for question at index: {record_idx}.")
        
        # Validate index in dataset size range of indices
        if record_idx not in range(0, dataset_size - 1):
            logger.error(
                f"Record index: {record_idx} is out of bounds. "
                f"Valid range: [0, {dataset_size}]"
            )
            return None
        
        # Extract single question
        logger.success(f"Found question record at index: {record_idx}")
        return [dataset[record_idx]]
    
    # Handle range
    elif isinstance(record_idx, range):
        logger.info(f"Looking for questions in index range: {record_idx}")
        
        # Validate range bounds
        if record_idx.start < 0 or record_idx.stop > dataset_size:
            logger.error(
                f"Index range: {record_idx} out of bounds. "
                f"Dataset size: {dataset_size}."
            )
            return None
        
        # Extract subset using range
        logger.success(f"Found question records in range: {record_idx}")
        return dataset[record_idx.start:record_idx.stop:record_idx.step]
    
    # Handle invalid type
    else:
        logger.error(
            f"Argument record_idx must be int or range, "
            f"got: {type(record_idx).__name__}"
        )
        return None

def question_records_to_question_str(question_records: list) -> dict[str, dict[str, str]]:
    """Converts ArenaHardAuto question records to a mapping between question ID and question data.

    Args:
        question_records (list): List of ArenaHardAuto question records.

    Returns:
        dict[str, dict[str, str]]: Converted question records to a mapping between question ID and question string value and category.
    """
    logger.info("Conveting question records to question mapping...")
    
    converted_records: dict[str, dict[str, str]] = {}
    
    for record in question_records:
        record_id: str = record['inputs']["question_id"]
        record_question: str = record['inputs']['question']
        record_category: str = record['inputs'].get('category', 'unknown')
        converted_records[record_id] = {
            "question": record_question,
            "category": record_category
        }
    
    return converted_records

def get_or_create_versioned_experiment(experiment_name: str) -> tuple[str, str, int]:
    """Find latest version of experiment for given complex workflow experiment, increment it if existent.

    Pattern: "{experiment_name}_v{version}".

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        tuple[str, str, int]: Experiment ID, experiment name and new version.
    """
    logger.info(f"Searching for MLFlow experiments called: {experiment_name}")
    all_experiments = mlflow.search_experiments(
        filter_string=f"name LIKE '{experiment_name}_v%'",
        order_by=["creation_time DESC"]
    )

    if all_experiments:
        logger.info(f"Found MLFlow experiment with prefix: {experiment_name}. Incrementing version...")
        versions = []
        for exp in all_experiments:
            try:
                version_str = exp.name.split("_v")[-1]
                versions.append(int(version_str))
            except (ValueError, IndexError):
                continue
        next_version = max(versions) + 1 if versions else 1
    else:
        logger.info(f"No MLFlow experiment called: {experiment_name} was found.")
        next_version = 1

    experiment_name = f"{experiment_name}_v{next_version}"
    logger.info(f"New MLFlow experiment version: {next_version}.")
    
    experiment_id = mlflow.create_experiment(experiment_name)
    logger.success(f"Created experiment: {experiment_name} (ID: {experiment_id})")
    
    return experiment_id, experiment_name, next_version


def load_models_from_config(config: dict[str, Any], config_class_key: str, config_subclass_key: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Dynamically load from configuration file and extract models configuration.

    By default, if no config_subclass_key argument is provided, it will return first level of configuration. For example, if config_class_key="cascade_complex_run", it will
    return all cascade levels models.

    If config_subclass_key argument is provided, this function will return second level of congiguration. For example, if config_class_key="cascade_complex_run" and 
    config_subclass_key="cascade_lvl_1", it will return all the models configurations for level 1 of the cascade.

    Args:
        config (dict[str, Any]): Configuration object that contains information about each workflow of the app.
        config_class_key (str): First level retrieval key.
        config_subclass_key (Optional[str], optional): Second level retrieval key. Defaults to None.

    Returns:
        Optional[dict[str, Any]]: Specific portion of the configuration file. For example, depending if config_subclass_key provided, it may return specific models cascade-wise
        or entire cascade levels.
    """
    logger.info(f"Retrieving models configuration from class: {config_class_key}.")
    models_class_config: dict[str, Any] = config.get(config_class_key, {})
    
    if not models_class_config:
        logger.error(f"No config class entry with key: {config_class_key}.")
        return None
    logger.success(f"Succefully retrieved config class entry with key: {config_class_key}")
    
    if not config_subclass_key:
        logger.info(f"Early stop in configuration retrieval at class level: {config_class_key}.")
        return models_class_config
    
    logger.info(f"Retrieving models configuration from subclass: {config_subclass_key}.")
    
    models_subclass_config: dict[str, Any] = models_class_config.get(config_subclass_key, {})
    
    if not models_subclass_config:
        logger.error(f"No config subclass entry with key: {config_subclass_key}.")
        return None
    
    logger.success(f"Succesfully retrieved subclass: {config_subclass_key} models configuration.")
    
    return models_subclass_config


def initialize_worker_agents(models_config: dict[str, dict[str, Any]], cascade_lvl: int) -> dict[str, WorkingAgent]: # Optional[dict[str, WorkingAgent]]:
    """Initializes WorkingAgent objects at a specific level of cascade.

    Args:
        models_config (dict[str, dict[str, Any]]): Models config.
        cascade_lvl (int): Specific level of cascade.

    Returns:
        dict[str, WorkingAgent]: Map of working agent ID and WorkingAgent object
    """
    worker_agents_dict: dict[str, WorkingAgent] = {}
    
    for worker_key, worker_config in models_config.items():
        model_name: str = worker_config.get("model_name", None)
        logger.info(f"Attempting to instantiate working agent: {worker_key} - {model_name}")
        endpoint_struct: dict[str, str] = worker_config.get("endpoint", None)
        parameters_struct: dict[str, Any] = worker_config.get("parameters", None)

        if any(x is None for x in [model_name, endpoint_struct, parameters_struct]):
            logger.warning("Malformed cascade model configuration found. Skipping...")
            continue
        
        worker_agents_dict[worker_key] = WorkingAgent(
            model_id=model_name,
            role_name=worker_key,
            system_instruction=WORKING_AGENT_SYSTEM_PROMPT_2,
            cascade_tier=cascade_lvl,
            config=worker_config, # parameters_struct,
            api_key=endpoint_struct.get('api_key', None)
        )
        
        logger.success(f"Instantiated working agent with ID: {worker_agents_dict[worker_key].model_id}")
        
    return worker_agents_dict


def initialize_judge_agent(judges_config: dict[str, Any], judge_key: str) -> Optional[ValidatorAgent]:
    specific_judge_config: dict[str, Any] = judges_config.get(judge_key, {})
    
    if not specific_judge_config:
        logger.error(f"No judge model found by key: {judge_key}.")
        return None
    
    judge_name: str = specific_judge_config.get("model_name", None)
    endpoint_struct: dict[str, str] = specific_judge_config.get("endpoint", None)
    parameters_struct: dict[str, Any] = specific_judge_config.get("parameters", None)
    
    if any(x is None for x in [judge_name, endpoint_struct, parameters_struct]):
        logger.error("Malformed judge model configuration found.")
        
    logger.info(f"Attempting to instantiate validator agent (judge): {judge_key} - {judge_name}")
    
    validator_agent: ValidatorAgent = ValidatorAgent(model_name=judge_name, api_key=endpoint_struct.get('api_key'))
    
    logger.success(f"Instantiated judge agent with ID: {judge_key} - {validator_agent.model_name}")
    
    return validator_agent

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



def __clean_full_string(string_to_clean: str) -> str:
    """Helper function to clean strings from \\n \\t \\r symbols.

    Args:
        string_to_clean (str): String that is being cleaned.

    Returns:
        str: cleaned version of the original string.
    """
    table = str.maketrans("\n\t\r", "   ")
    return string_to_clean.translate(table)

def __format_response(long_string_log: str, len_portion: int = None) -> str:
    """Helper function to properly display the string contents in logs.

    Args:
        long_string_log (str): Long string that should be shortened (AgentResponse content, Question prompt content, etc.).
        len_portion (int, optional): Length of the left and right portions of the shortened version of the content. Defaults to None.

    Returns:
        str: formatted (shortened) string variant of the response content.
    """
    if not len_portion:
        len_portion = min(51, len(long_string_log) // 10)
    left_portion: str = long_string_log[:len_portion + 1]
    right_portion: str = long_string_log[-(len_portion + 1):]
    return f"{__clean_full_string(string_to_clean=left_portion)}...{__clean_full_string(string_to_clean=right_portion)}"


async def run_working_agent(worker_agent: tuple[str, WorkingAgent], func: Coroutine, **kwargs) -> tuple[str, AgentResponse, dict]:
    """Run an individual working agent on the same cascade level to generate answers to the provided prompt.

    Can be used with any answer generation from WorkingAgent class by passing its method function as an argument.

    Args:
        worker_agent (WorkingAgent): Dictionary mapping agent IDs to WorkingAgent instances.
        prompt_question (Union[str, list[AgentResponse]]): The question/prompt strings to send to appropriate agent.

    Returns:
        tuple[str, AgentResponse]: Dict of AgentResponse objects mapped to their own agent ID - answers from the agents to the prompt.
    """
    agent_id, agent = worker_agent
    func_kwargs = {**kwargs}

    # Reset tracker before call
    USAGE_TRACKER['last_cost'] = 0.0
    USAGE_TRACKER['last_input'] = 0
    USAGE_TRACKER['last_output'] = 0

    try:
        agent_answer: AgentResponse = await func(**func_kwargs)
    except Exception as e:
        logger.error(f"Agent {agent_id} failed with exception: {e}")
        cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        return (agent_id, None, cost)

    # Capture cost after call
    if USAGE_TRACKER['last_cost'] > 0:
        cost = {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": USAGE_TRACKER['last_cost']
        }
    else:
        # Fallback estimation
        cost = {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0
        }
    
    if isinstance(agent_answer, Exception):
        logger.error(f"Agent {agent_id} failed: {agent_answer}")
        return (agent_id, None)
    if isinstance(agent_answer, AgentRunError):
        logger.error(f"Agent {agent_id} failed with AgentRunError: {agent_answer}")
        return (agent_id, None)
    else:
        logger.success(f"Agent {agent_id} succeeded: {__format_response(long_string_log=agent_answer.content)}")
    
    logger.success(f"Generated valid response by agent: {agent_id} - {agent.model_id}")
    return  (agent_id, agent_answer, cost)


async def run_cascade_initial_answer(worker_agents: dict[str, WorkingAgent], prompt_data: tuple[str, str], current_level: int = 1) -> tuple[dict[str, AgentResponse], float]:
    """Generate initial answers concurrently.

    Args:
        worker_agents (dict[str, WorkingAgent]): List of WorkingAgent object.
        prompt_data (tuple[str, str]): User prompt.
        current_level (int, optional): Current level of the cascade. Defaults to 1.

    Returns:
        tuple[dict[str, AgentResponse], float]: Responses and total cost
    """
    prompt_id, prompt_question = prompt_data
    logger.info(f"Starting initial answer generation at cascade level: {current_level} with: {len(worker_agents.keys())} working agents and with prompt with ID: {prompt_id} - \"{__format_response(long_string_log=prompt_question)}\"")
    
    func_kwargs = {
        "context": Prompt(content=prompt_question, model_tier='complex')
    }
    
    tasks = [
        run_working_agent(
            worker_agent=(agent_id, agent),
            func=agent.generate,
            **func_kwargs
        )
        for agent_id, agent in worker_agents.items()
    ]
    
    results: list[tuple[str, AgentResponse]] = await asyncio.gather(*tasks, return_exceptions=True)

    agents_responses: dict[str, AgentResponse] = {}
    total_cost = 0.0


    for response in results:

        if isinstance(response, Exception):
            logger.error(f"Agent failed with exception: {response}")
            continue
        
        agent_id, response_content, cost = response
        total_cost += cost['total_cost']

        if not response_content:
            # Since order of results is preserved by asyncio.gather() function in the same order as awaitables in *aws = *tasks
            logger.error(f"Agent {agent_id} failed: {response_content}")
            agents_responses[agent_id] = 'N/A'
        else:
            logger.success(f"Agent {agent_id} succeeded: {__format_response(long_string_log=response_content.content)}")
            agents_responses[agent_id] = response_content
            
    logger.success(f"Generated {len(agents_responses.keys())} valid responses out of {len(worker_agents)} agents")
    return agents_responses, total_cost


async def run_cascade_debate(worker_agents: dict[str, WorkingAgent], prev_answers: dict[str, AgentResponse], current_level: int = 1) -> tuple[dict[str, AgentResponse], float]:
    """Generate critique/debate responses where each agent reviews peer responses.

    Args:
        worker_agents (dict[str, WorkingAgent]): Dictionary of WorkingAgent instances.
        prev_answers (dict[str, AgentResponse]): Previous round's responses to critique.
        current_level (int, optional): Current cascade level. Defaults to 1.

    Returns:
        tuple[dict[str, AgentResponse], float]: Critiques and total cost
    """
    logger.info(f"Starting debate process at cascade level: {current_level} with: {len(worker_agents.keys())} working agents")
    
    logger.info(f"Generating critiques based on {len(prev_answers)} previous responses")
    if not prev_answers:
        logger.error("No previous answers provided for debate.")
        return None

    tasks = []
    for agent_id, worker_agent in worker_agents.items():
        logger.info(f"Attempting to generate critique response by agent: {agent_id} - {worker_agent.agent}")
        # print(f"{prev_answers[agent_id].author_id} - {agent_id}")
        individual_peer_responses: list[Prompt] = [
            Prompt(content=peer_response.content, model_tier="complex")
            for peer_id, peer_response in prev_answers.items()
            if peer_id != agent_id
        ]
        
        # print(
        #     json.dumps(
        #         [
        #             {
        #                 "content": p.content,
        #                 "model_tier": p.model_tier,
        #             }
        #             for p in individual_peer_responses
        #         ],
        #         indent=4
        #     )
        # )
                
        if not individual_peer_responses:
            logger.warning(f"Agent {agent_id} has no peer responses to critique")
            continue
    
        func_kwargs = {
            "peer_responses": individual_peer_responses
        }
        
        #logger.info(f"Agent {agent_id} will critique {len(individual_peer_responses)} ({" ".join([peer_response.author_id for peer_id, peer_response in prev_answers.items() if peer_id != agent_id])}) peer response(s)")
        
        tasks.append(
            run_working_agent(
                worker_agent=(agent_id, worker_agent),
                func=worker_agent.generate_critique,
                **func_kwargs
            )
        )
        
    logger.info(f"Executing {len(tasks)} critique tasks in parallel")
    critiques: list[tuple[str, AgentResponse, dict]] = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_critiques: dict[str, AgentResponse] = {}
    total_cost = 0.0
    
    for critique in critiques:

        if isinstance(critique, Exception):
            logger.error(f"Critique failed with exception: {critique}")
            continue

        agent_id, critique_content, cost = critique
        total_cost += cost['total_cost']
        if not critique_content:
            # Since order of results is preserved by asyncio.gather() function in the same order as awaitables in *aws = *tasks
            logger.error(f"Agent {agent_id} critique failed: {critique}")
        else:
            logger.success(f"Agent {agent_id} generated critique: {__format_response(long_string_log=critique_content.content)}")
            valid_critiques[agent_id] = critique_content
            
    logger.success(f"Debate completed. Generated {len(valid_critiques.keys())} valid critiques out of {len(worker_agents)} agents")
    return valid_critiques, total_cost


async def run_validation(judge_agent: ValidatorAgent, question: str, prompt: Prompt, answers: dict[str, AgentResponse]) -> Optional[dict[str, Any]]:
    logger.info(f"Engaging judge agent in validation of: {len(answers.keys())} answers.")
    judge_response = await judge_agent.evaluate_multiple(prompt=prompt, question=question, answers=answers)
    
    if not judge_response:
        logger.error(f"Judge agent did not succeed to generate appropriate response: {judge_response}")
        print(judge_response)
        return None
    
    if "reasoning" in judge_response.keys():
        logger.error(f"Judge agent generated malformed response: {judge_response}")
        print(judge_response)
        return json_repair.loads(judge_response['reasoning'])
    
    if "type" in judge_response.keys():
        logger.warning(f"Judge agent did not convey to format: {__format_response(long_string_log=json.dumps(judge_response))}")
        print(judge_response)
        return {
            "evaluation": judge_response
        }
    
    logger.success("Judge agent succeeded to generate apropriate response.")
    print(judge_response)
    return judge_response


def convert_peer_reviews_to_feedback_string(critiques: dict[str, AgentResponse]) -> str:
    """Helper function to convert peer reviews AgentResponse instances to a singular XML-like format.

    Args:
        critiques (dict[str, AgentResponse]): Peer reviews mapped to their reviewer ID, delivered to specific WorkingAgent whose answers are passed in this function.

    Returns:
        str: Peer reviews structured in XML-like format.
    """
    blocks: list[str] = []

    for idx, (agent_id, response) in enumerate(critiques.items(), start=1):
        blocks.append(
            f"""
<peer_review_{idx}_start>
<agent_id_start>{agent_id}<agent_id_end>
<review_start>
{response.content}
<review_end>
<peer_review_{idx}_end>
""".strip()
        )

    feedback_peer_review_str = (
        "<peer_reviews_start>\n"
        + "\n\n".join(blocks)
        + "\n<peer_reviews_end>"
    )

    return feedback_peer_review_str


async def build_refinement_prompt(initial_question: str, initial_answer: tuple[str, AgentResponse], critiques: dict[str, AgentResponse]) -> tuple[str, Prompt]:
    """Helper function to build refinement user prompt based on previous initial answer and peer reviews/critiques.

    Args:
        initial_question (str): Initial user input question/prompt.
        initial_answers (dict[str, AgentResponse]): Initial WorkingAgent instances answers mapped by agent ID.
        critiques (dict[str, AgentResponse]): Peer reviews mapped to their reviewer ID, delivered to specific WorkingAgent whose answers are passed in this function.

    Returns:
        tuple[str, Prompt]: New user prompt that is intended to refine WorkingAgent previous answer.
    """
    agent_id, initial_answer_content = initial_answer
    logger.info(f"Attempting to build refinement prompt for agent: {agent_id}")
    refinement_prompt_str: str = f"""
Here are the initial question and the initial answer you gave for the initial question.

INITIAL QUESTION:
<initial_question_start>
{__clean_full_string(string_to_clean=initial_question)}
</initial_question_end>

INITIAL ANSWER:
<initial_answer_start>
{__clean_full_string(string_to_clean=initial_answer_content.content)}
<initial_answer_end>

Carefully analyze the feedback that other working agents provided to your answer, identify valid points in the feedback and refine your answer in accordance to them.

PEER REVIEWS:
{convert_peer_reviews_to_feedback_string(critiques=critiques).strip()}
    """
    
    print(refinement_prompt_str)
    
    logger.success(f"Refinement prompt for agent: {agent_id} was built: {__format_response(long_string_log=refinement_prompt_str)}")
    return (agent_id, refinement_prompt_str)

async def run_cascade_refinement_loop(worker_agents: dict[str, WorkingAgent], 
                                      init_question: str,
                                      init_answers: dict[str, AgentResponse],
                                      critiques: dict[str, AgentResponse],
                                      current_level: int = 1) -> tuple[dict[str, AgentResponse], float]:
    """Generate refined answers using WorkingAgent objects and their initial answers and peer reviews/critiques.

    Args:
        worker_agents (dict[str, WorkingAgent]): Dictionary of WorkingAgent instances.
        init_answers (dict[str, AgentResponse]): WorkingAgent instances initial answers. 
        critiques (dict[str, AgentResponse]): WorkingAgent instances peer reviews.
        current_level (int, optional): Current cascade level. Defaults to 1.

    Returns:
        Optional[dict[str, AgentResponse]]: Dict of final answers AgentResponse objects mapped to their generator agent ID.
    """
    logger.info(f"Starting final answer generation at cascade level: {current_level} with: {len(worker_agents.keys())} working agents.")
    
    tasks = []
    for worker_id, worker_agent in worker_agents.items():
        if worker_id not in init_answers or init_answers[worker_id] == 'N/A':
            logger.warning(f"Skipping {worker_id} - no valid initial answer")
            continue

        initial_answer: tuple[str, AgentResponse] = (worker_id, init_answers[worker_id])
        peers_critiques: dict[str, AgentResponse] = {peer_id: peer_critique for peer_id, peer_critique in critiques.items() if worker_id != peer_id}

        _, refinement_prompt = await build_refinement_prompt(
            initial_question=init_question,
            initial_answer=initial_answer,
            critiques=peers_critiques
        )
        
        func_kwargs = {
            "context": Prompt(content=refinement_prompt, model_tier='complex')
        }

        tasks.append(
            run_working_agent(
                worker_agent=(worker_id, worker_agent),
                func=worker_agent.generate,
                **func_kwargs
            )
        )
    
    refinement_results: list[tuple[str, AgentResponse, dict]] = await asyncio.gather(*tasks, return_exceptions=True)

    agents_responses: dict[str, AgentResponse] = {}
    total_cost = 0.0
    
    for refinement_result in refinement_results:
        if isinstance(refinement_result, Exception):
            logger.error(f"Refinement failed with exception: {refinement_result}")
            continue

        agent_id, refinement_result_content, cost = refinement_result
        total_cost += cost['total_cost']
        if not refinement_result_content:
            # Since order of results is preserved by asyncio.gather() function in the same order as awaitables in *aws = *tasks
            logger.error(f"Agent {agent_id} refinement failed: {refinement_result_content}")
        else:
            logger.success(f"Agent {agent_id} refinement succeeded: {__format_response(long_string_log=refinement_result_content.content)}")
            agents_responses[agent_id] = refinement_result_content
            
    logger.success(f"Generated {len(agents_responses.keys())} valid refined responses out of {len(worker_agents)} agents")
    return agents_responses, total_cost


def convert_agents_answers_to_ensemble_prompt(answers: dict[str, AgentResponse]) -> str:
    """Helper function to convert worker agents AgentResponse instances to a singular XML-like format.

    Args:
        answers (dict[str, AgentResponse]): Worker agents answers mapped to their agent ID.

    Returns:
        str: Agents answers structured in XML-like format.
    """
    blocks: list[str] = []

    for worker_id, response in answers.items():
        blocks.append(
            f"""
<{worker_id}_answer_start>
{response.content}
<{worker_id}_answer_end>
""".strip()
        )

    agents_answers_str = (
        "<workers_answers_start>\n"
        + "\n\n".join(blocks)
        + "\n<workers_answers_end>"
    )

    return agents_answers_str


def validator_agent_response_to_agents_reviews(validator_responses: dict[str, Any]) -> dict[str, str]:
    """Extract from ValidatorAgent judge responses each worker agent review response.

    Args:
        validator_responses (dict[str, Any]): ValidatorAgent worker agents evaluation response.

    Returns:
        dict[str, str]: Extracted review per each WorkerAgent mapped to their config ID.
    """
    
    workers_reviews: dict[str, str] = {}
    
    validator_individual_reviews: dict[str, dict[str, Any]] = validator_responses.get("evaluation", {}).get("individual_reviews", {})
    
    if not validator_individual_reviews:
        logger.warning("No individual reviews from ValidatorAgent was found. Maybe the response is incomplete or malformed.")
        return None
    
    for worker_id, worker_review in validator_individual_reviews.items():
        individual_review: str = worker_review.get('reason', None)
        if not individual_review: 
            logger.warning("Individual review with invalid reason or malformed structure found. Omitting...")
            workers_reviews[worker_id] = "N/A"
            continue
        
        workers_reviews[worker_id] = individual_review

    logger.success("Extracted individual answer evaluation reason.")
    
    print(json.dumps(workers_reviews, indent=2))
    
    return workers_reviews


def ensemble_agents_answers(agents_answers: dict[str, AgentResponse], initial_question: str, premise_clause: str, agents_answers_review: dict[str, Any] = None) -> str:
    """Helper method to ensemble multiple agents answers into a single prompt.

    May be used to:
    1. generate a prompt for next level cascade (if agents_answers_review is provided);
    2. generate a prompt for ValidatorAgent instance (no agents_answers_review is provided).

    Args:
        agents_answers (dict[str, AgentResponse]): Worker agents answers to user or previous level cascade prompt.
        initial_question (str): Initial question/prompt from user or previous level cascade.
        premise_clause (str): Initial statement in the prompt. This function supports prompt ensembling both for ValidatorAgent and next level cascade workers.
        agents_answers_review (dict[str, AgentResponse]): ValidatorAgent instance review to worker agents answers. Defaults to None.

    Returns:
        str: Ensembled prompt with worker agents responses and their review (if provided).
    """
    ensembled_workers_answers: str = f"""
{premise_clause}

INITIAL QUESTION:
{__clean_full_string(string_to_clean=initial_question)}

ANSWERS:
{convert_agents_answers_to_ensemble_prompt(answers=agents_answers).strip()}
    """
    
    if agents_answers_review:
        logger.info(f"Ensembling judge reasoning for each worker agent: {len(agents_answers.keys())} response.")
        reviews: dict[str, str] = validator_agent_response_to_agents_reviews(validator_responses=agents_answers_review)
        
        if reviews:
            logger.info("Appending ensembled prompt with judge reviews.")
            review_blocks: list[str] = []
            
            validator_review_section = "\n\nVALIDATOR REVIEWS:\n"
            
            for worker_id, worker_review in reviews.items():
                logger.info(f"Appending ensembled prompt with review for: {worker_id} - {__format_response(long_string_log=worker_review)}.")
                review_blocks.append(
                    f"""
<{worker_id}_review_start>
{worker_review}
<{worker_id}_review_end>
""".strip()
                )
                
            validator_review_section += (
                "<validator_reviews_start>\n"
                + "\n\n".join(review_blocks)
                + "\n<validator_reviews_end>"
            )
            
            ensembled_workers_answers += validator_review_section
            logger.success("Succesfully appended review section to ensembled answers.")
            
    print(ensembled_workers_answers)
    
    logger.success(f"Ensembled prompt was built: {__format_response(long_string_log=ensembled_workers_answers)}")
    return ensembled_workers_answers

    
# TODO: update this method with other way of checking
async def synthetize_final_answer(validator_agent: ValidatorAgent, final_answers: dict[str, AgentResponse]) -> AgentResponse:
    """_summary_

    Args:
        validator_agent (ValidatorAgent): _description_
        final_answers (dict[str, AgentResponse]): _description_

    Returns:
        AgentResponse: _description_
    """
    final_response: AgentResponse = await validator_agent.synthesize_final(responses=final_answers.values())
    return final_response.content


def log_final_answer_trace(question_id: str, 
                           question_prompt: str, 
                           is_definitive: bool, 
                           acceptable_score: float, 
                           synthetized_answer: str, 
                           best_worker_model_id: str, 
                           cascade_level: int, 
                           best_confidence_score: float,
                           total_cost: float) -> None:
    """Log the final answer trace to MLflow.

    Args:
        question_id (str): ID of the question.
        question_prompt (str): Original question text.
        is_definitive (bool): Flag to show if the log is for the final answer
        acceptable_score (float): Acceptable score threshold.
        synthetized_answer (str): Final synthesized answer.
        best_worker_model_id (str): ID of the best performing worker model.
        cascade_level (int): Cascade level where answer was found.
        best_confidence_score (float): Confidence score of the answer.
    """
    logger.info(
        f"Attempting to trace manually best answer ({best_confidence_score}) to question: {question_id} - "
        f"{__format_response(long_string_log=question_prompt)} was generated by: {best_worker_model_id} "
        f"at cascade level: {cascade_level}, with answer: {__format_response(long_string_log=synthetized_answer)}."
    )
    
    answer_trace = {
        "input": question_prompt,
        "output": synthetized_answer,
        "is_definitive": is_definitive,
        "question_id": question_id,
        "acceptable_score": acceptable_score,
        "final_best_model" if is_definitive else "best_model": best_worker_model_id,
        "final_cascade_level" if is_definitive else "cascade_level": cascade_level,
        "final_best_confidence_score" if is_definitive else "best_confidence_score": best_confidence_score,
        "total_cost": total_cost
    }
    
    with mlflow.start_span(name="best_final_answer_trace" if is_definitive else f"best_answer_trace_lvl_{cascade_level}", 
                           span_type=SpanType.LLM) as current_span:
        current_span.set_inputs(
            inputs={
                "question_id": question_id,
                "question": question_prompt,
                "is_definitive": is_definitive,
                "acceptable_score": acceptable_score
            }
        )
        
        current_span.set_attributes(
            attributes=answer_trace
        )
        
        current_span.set_outputs(
            outputs={
                "final_best_response" if is_definitive else f"best_response_lvl_{cascade_level}": synthetized_answer,
                "final_best_model" if is_definitive else f"best_model_lvl_{cascade_level}": best_worker_model_id,
                "final_best_confidence_score_lvl" if is_definitive else f"best_confidence_score_lvl_{cascade_level}": best_confidence_score
            }
        )
    
    if is_definitive:
        try:
            mlflow.log_dict(answer_trace, "final_answer.json")
            mlflow.log_metric("final_best_confidence_score", best_confidence_score)
            mlflow.log_metric("final_cascade_level", cascade_level)
            mlflow.log_param("final_best_model", best_worker_model_id)
            
            logger.success(f"Logged final answer artifacts and metrics for question: {question_id}")
        except Exception as e:
            logger.error(f"Failed to log final answer artifacts: {e}")
            return
    
    logger.success(f"Logged final answer trace for question: {question_id}")


def get_latest_results_version(results_path: Union[str, Path], extension: str = ".jsonl") -> Optional[str]:
    """Helper function to extract from a directory the latest version of the iterations.

    Results directory expected to contain:

    ```
    results/
        - arena-hard-auto-v1.0/
        - arena-hard-auto-v2.0/
    ```

    Args:
        results_path (Union[str, Path]): Path to the results directory.

    Returns:
        Optional[str]: New iteration file basename (without extension).
    """
    results_path_dir: Path = Path(results_path)
    
    if not results_path_dir.exists():
        logger.error(f"Results directory at: {results_path_dir} do not exist.")
        return None
    
    pattern = re.compile(
        rf"^iteration_(\d+){re.escape(extension)}$"
    )
    
    iterations: list[tuple[int, str]] = []

    for file in results_path_dir.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                iterations.append(int(match.group(1)))

    if not iterations:
        logger.warning("No previous iterations found. Falling back to default version: iteration_1")
        return "iteration_1"
    
    latest_iteration = max(iterations)
    next_iteration = latest_iteration + 1

    new_iteration_str = f"iteration_{next_iteration}"
    logger.success(f"Previous iterations exist. Increasing the version: {new_iteration_str}")
    
    return new_iteration_str
    

def save_results_to_jsonl(question_id: str,
                          num_questions: int,
                          category: str,
                          question: str,
                          answer: str,
                          cascade_lvl: int,
                          winner_model: str,
                          judge_model: str,
                          score: float,
                          save_to_path_file: str) -> None:
    """Save definitive answer results to a results local file (default - JSONL format).

    Args:
        question_id (str): ID of the question.
        category (str): Category of the question. Determines the iteration of the ArenaHardAuto.
        question (str): Question/Prompt
        answer (str): Worker models answer to the question/prompt.
        cascade_lvl (int): Last cascade level that yielded the answer.
        winner_model (str): Best worker agent that yielded the answer.
        judge_model (str): Validator agent used for the iteration.
        score (float): Achieved score by the best worker agent.
        save_to_path_file (str): Path to the file that will contain the results per iteration
    """
    jsonl_entry: dict[str, Union[str, float, int]] = {
        "question_id": question_id,
        "num_question_selected": num_questions,
        "category": category,
        "question": question,
        "answer": answer,
        "cascade_lvl": cascade_lvl,
        "winner_model": winner_model,
        "judge_model": judge_model,
        "score": score
    }
    
    with open(save_to_path_file, "a", encoding="utf-8") as f_jsonl:
        f_jsonl.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
    
    # TODO: LOG IN MLFLOW
    # with tempfile.NamedTemporaryFile("w", delete=False, suffix=extension, encoding="utf-8") as tmp_file:
    #     tmp_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
    #     tmp_file_path = Path(tmp_file.name)

    # # Log artifact under a folder for the category
    # mlflow.log_artifact(str(tmp_file_path), artifact_path=f"results/{category}")

    # # Clean up temp file if you want
    # tmp_file_path.unlink()


def map_categories_to_dir_paths(questions_mapping: dict[str, dict[str, str]], extension: str = ".jsonl") -> dict[str, Path]:
    categories = set(q.get('category', 'unknown') for q in questions_mapping.values())

    logger.info(f"Attempting to create directories for each category: {categories}")
    for category_name in categories:
        logger.info(f"Creating directory for category: {category_name}")
        category_dir: Path = Path(RESULTS_PATH).resolve() / category_name
        init_dirs(paths=[category_dir])
    
    extension: str = ".jsonl"
    logger.info(f"Selected results file extension: {extension}")
    
    category_iteration_files: dict[str, Path] = {}
    
    for category_name in categories:
        logger.info("Creating categories paths for each category")
        specific_category_dir: Path = Path(RESULTS_PATH).resolve() / category_name
        results_file_version: str = get_latest_results_version(results_path=specific_category_dir, extension=extension)
        results_file: Path = specific_category_dir / f"{results_file_version}{extension}"
        category_iteration_files[category_name] = results_file
        logger.info(f"Category '{category_name}' will use file: {results_file}")

    return category_iteration_files

def main() -> None:
    # parser = argparse.ArgumentParser()
    
    # * STEP 0: Prerequisites
    init_dirs(paths=[RESULTS_PATH])
    
    # * STEP 1: Setting up MLFlow
    perform_initial_mlflow_setup(mlflow_uri=MLFLOW_TRACKING_URI)
    
    # * Retrieving Dataset from MLFlow
    dataset_records: dict[str, Any] = get_experiment_dataset(experiment_name=MLFLOW_DATASET_EXPERIMENT_NAME, dataset_name=MLFLOW_DATASET_NAME)['records']
    if not dataset_records:
        logger.error("Dataset was not found.")
        return
    
    # * STEP 2: Extracting question records from the dataset
    # * SCENARIO 1: ENTIRE MLFLOW DATASET OF QUESTIONS
    # question_records_idx: int = range(0, len(dataset_records))
    # question_records: list = get_question_records_from_dataset(dataset=dataset_records, record_idx=question_records_idx)
    # print(question_records)
    # if not question_records:
    #     logger.error("Question records not found.")
    #     return
    
    # * SCENARIO 2: SUBSET OF QUESTIONS (NUM_MAX_QUESTIONS) FROM MLFLOW DATASET
    question_records_idx: int = range(0, NUM_MAX_QUESTIONS)
    question_records: list = get_question_records_from_dataset(dataset=dataset_records, record_idx=question_records_idx)
    if not question_records:
        logger.error("Question records not found.")
        return
    
    # * SCENARIO 3: CUSTOM QUESTION
    # question_records: list = [USER_PROMPT]
    
    # * SCENARIO 4: SINGLE QUESTION FROM MLFLOW DATASET
    # question_records_idx: int = 1
    # question_records: list = get_question_records_from_dataset(dataset=dataset_records, record_idx=question_records_idx)
    # if not question_records:
    #     logger.error("Question record not found.")
    #     return
    
    questions_mapping: dict[str, dict[str, str]] = question_records_to_question_str(question_records=question_records)
    
    for idx, (question_id, question) in enumerate(questions_mapping.items()):
        print(f"Question {idx + 1}: {question_id} - {question.get('category')} - {question.get('question')}")
    
    experiment_id, experiment_name, version = get_or_create_versioned_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_name)
    
    # * Create paths for each category assigned to questions
    extension: str = ".jsonl"
    category_iteration_files: dict[str, Path] = map_categories_to_dir_paths(questions_mapping=questions_mapping, extension=extension)
    
    # * Instantiate Judge Agent
    judge_models = load_models_from_config(config=CASCADE_MODELS_CONFIG, config_class_key=JUDGE_MODELS_CONFIG_KEY)
    print(json.dumps(judge_models, indent=2))
    
    for idx, (question_id, question_data) in enumerate(questions_mapping.items()):
        original_question = question_data['question']
        question_prompt = question_data['question']
        question_category = question_data['category']
        logger.info(f"[{idx + 1}/{len(questions_mapping.keys())}] Processing prompt with ID: {question_id} - {question_category} - \"{question_prompt}\"")
        
        validator_agent = initialize_judge_agent(judges_config=judge_models, judge_key=JUDGE_MODEL_KEY)
    
        if not validator_agent:
            logger.error("Failed to initialize a judge agent.")
            return

        print(validator_agent)
        print(validator_agent.model_name)
        
        with mlflow.start_run(run_name=question_id):
            mlflow.log_params({
                "question_id": question_id,
                "category": question_category,
                "in_cascade_judge_model": validator_agent.model_name,
                "config_key": COMPLEX_RUN_CONFIG_KEY,
                "version": version
            })
            prompt_data: tuple[str, str] = (question_id, question_prompt)
            
            for current_cascade_level in range(1, MAX_CASCADE_LEVEL + 1):
                logger.info(f"Entering cascade level {current_cascade_level}.")
                
                # * STEP 3: Configuring Agents
                # * Loading complex run config models
                cascade_lvl_models = load_models_from_config(config=CASCADE_MODELS_CONFIG, 
                                                             config_class_key=COMPLEX_RUN_CONFIG_KEY, 
                                                             config_subclass_key=f"cascade_lvl_{current_cascade_level}")
                print(json.dumps(cascade_lvl_models, indent=2))
                
                # * Instantiate Agents using working models configs
                worker_agents_dict: dict[str, WorkingAgent] = initialize_worker_agents(models_config=cascade_lvl_models, cascade_lvl=current_cascade_level)
                
                for worker_key, worker_agent in worker_agents_dict.items():
                    print(f"{worker_key} - {worker_agent.model_id}")

                cascade_total_cost = 0.0
                # * STEP 4: Working Agents actions:
                # * Generate initial responses
                initial_answers, init_cost = asyncio.run(run_cascade_initial_answer(worker_agents=worker_agents_dict,
                                                                                                   prompt_data=prompt_data, 
                                                                                                   current_level=current_cascade_level))
                cascade_total_cost += init_cost

                # print(f"Question {idx}: {question_id} - {questions_mapping[question_id]}")
                for agent_initial_answer in initial_answers.values():
                    if isinstance(agent_initial_answer, str):  # Skip 'N/A' entries
                        continue
                    print(f"\nAgent {agent_initial_answer.author_id}: \n\tResponse: {agent_initial_answer.content}")

                    #print(f"\nAgent {agent_initial_answer.author_id}: \n\tResponse: {agent_initial_answer.content}")

                # * Generate Critiques/Debate prompts
                critiques, critique_cost = asyncio.run(run_cascade_debate(worker_agents=worker_agents_dict,
                                                                                     prev_answers=initial_answers, 
                                                                                     current_level=current_cascade_level))
                cascade_total_cost += critique_cost
                
                for agent_critique in critiques.values():
                    if isinstance(agent_critique, str):  # Skip 'N/A' entries
                        continue
                    print(f"\nAgent {agent_critique.author_id}: \n\tResponse: {agent_critique.content}")

                # * Generate refined final answers based on previous initial answers and critiques
                final_answers, refine_cost = asyncio.run(
                    run_cascade_refinement_loop(
                        worker_agents=worker_agents_dict, 
                        init_question=question_prompt,
                        init_answers=initial_answers, 
                        critiques=critiques, 
                        current_level=current_cascade_level
                    )
                )
                cascade_total_cost += refine_cost
                mlflow.log_metric(f"cascade_lvl_{current_cascade_level}_cost", cascade_total_cost)
                mlflow.log_metric("total_cost", cascade_total_cost)  # Update cumulative

                for agent_final_answer in final_answers.values():
                    if isinstance(agent_final_answer, str):  # Skip 'N/A' entries
                        continue
                    print(f"\nAgent {agent_final_answer.author_id}: \n\tResponse: {agent_final_answer.content}")

                # * STEP 5: Integrate Judge Agent to select the best final answer.
                validator_prompt: str = ensemble_agents_answers(agents_answers=final_answers, 
                                                                initial_question=question_prompt, 
                                                                premise_clause=JUDGE_PROMPT_1)
                validator_answer: dict[str, Any] = asyncio.run(run_validation(judge_agent=validator_agent,
                                                                              question=question_prompt,
                                                                              prompt=validator_prompt,
                                                                              answers=final_answers))
                if not validator_answer:
                    logger.error('ValidatorAgent did not succeed. Falling back to default...')

                    # Get first successful answer or default
                    first_valid = next((k for k, v in final_answers.items() if not isinstance(v, str)), None)

                    if not first_valid:
                        logger.error("No valid answers available. Skipping question.")
                        break

                    # Create fallback structure
                    validator_answer = {
                        "evaluation": {
                            "question": question_prompt,
                            "best_answer": {
                                "best_worker_model_id": first_valid,
                                "best_confidence_score": 0.0,
                                "best_reason": "Judge failed - using first available answer"
                            },
                            "individual_reviews": {
                                first_valid: {
                                    "confidence_score": 0.0,
                                    "reason": "Judge unavailable"
                                }
                            }
                        }
                    }
                    logger.info(f"Using fallback answer from: {first_valid}")


                if validator_answer['evaluation']['best_answer']['best_confidence_score'] >= ACCEPTABLE_SCORE:
                    # ANSWER GOOD
                    # TODO: Check how to update this method.
                    synthetized_final_answer: str = asyncio.run(synthetize_final_answer(validator_agent=validator_agent, final_answers=final_answers))
                    
                    print(f"Synthetized final answer:\n{synthetized_final_answer}")
                    print(f"FINAL ANSWER (DIRECT FROM WORKER):\n{final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content}")
                    
                    log_final_answer_trace(
                        question_id=question_id,
                        question_prompt=original_question,
                        is_definitive=True,
                        acceptable_score=ACCEPTABLE_SCORE,

                        # TODO: Maybe try with best all previous answers. Engage memory of validator?
                        synthetized_answer=final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content,
                        best_worker_model_id=cascade_lvl_models[validator_answer['evaluation']['best_answer']['best_worker_model_id']]['model_name'],
                        cascade_level=current_cascade_level,
                        best_confidence_score=validator_answer['evaluation']['best_answer']['best_confidence_score']
                        ,total_cost=cascade_total_cost
                    )
                    
                    logger.success(f"Best answer ({validator_answer['evaluation']['best_answer']['best_confidence_score']}) was generated by: {validator_answer['evaluation']['best_answer']['best_worker_model_id']} at cascade level: {current_cascade_level}, with answer: {__format_response(long_string_log=final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content)}.")
                    
                    results_file: Path = category_iteration_files[question_category]
                    
                    save_results_to_jsonl(
                        question_id=question_id,
                        num_questions=NUM_MAX_QUESTIONS,
                        category=question_category,
                        question=original_question,
                        answer=final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content,
                        cascade_lvl=current_cascade_level,
                        winner_model=cascade_lvl_models[validator_answer['evaluation']['best_answer']['best_worker_model_id']]['model_name'],
                        judge_model=validator_agent.model_name,
                        score=validator_answer['evaluation']['best_answer']['best_confidence_score'],
                        save_to_path_file=results_file
                    )
                    
                    break

                # NEXT CASCADE
                logger.info(f"Current cascade level: {current_cascade_level} did not result in good answer.")
                logger.info(f"Best answer ({validator_answer['evaluation']['best_answer']['best_confidence_score']}) was generated by: {validator_answer['evaluation']['best_answer']['best_worker_model_id']}, with answer: {__format_response(long_string_log=final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content)}")
                
                log_final_answer_trace(
                    question_id=question_id,
                    question_prompt=original_question,
                    is_definitive=False,
                    acceptable_score=ACCEPTABLE_SCORE,
                    # TODO: Maybe try with best all previous answers. Engage memory of validator?
                    synthetized_answer=final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content,
                    best_worker_model_id=cascade_lvl_models[validator_answer['evaluation']['best_answer']['best_worker_model_id']]['model_name'],
                    cascade_level=current_cascade_level,
                    best_confidence_score=validator_answer['evaluation']['best_answer']['best_confidence_score'],
                    total_cost=cascade_total_cost)
                
                next_cascade_prompt: str = ensemble_agents_answers(agents_answers=final_answers, 
                                                                   initial_question=question_prompt, 
                                                                   premise_clause=NEXT_CASCADE_LEVEL_PROMPT, 
                                                                   agents_answers_review=validator_answer)
                
                prompt_data = (question_id, next_cascade_prompt)
            # LAST LEVEL OF CASCADE WITH NO SUCCESS
            else:
                logger.warning(f"Last level of cascade: {current_cascade_level} done. No answer better than {ACCEPTABLE_SCORE} "
                            f"was deducted. Falling back to best answer at this level provided by: {validator_answer['evaluation']['best_answer']['best_confidence_score']}")
                logger.info(f"Best answer ({validator_answer['evaluation']['best_answer']['best_confidence_score']}) was generated by: {validator_answer['evaluation']['best_answer']['best_worker_model_id']}, with answer: {__format_response(long_string_log=final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content)}")
                
                synthetized_final_answer: str = asyncio.run(synthetize_final_answer(validator_agent=validator_agent, final_answers=final_answers))
                
                log_final_answer_trace(
                    question_id=question_id,
                    question_prompt=original_question,
                    is_definitive=True,
                    acceptable_score=ACCEPTABLE_SCORE,
                    # TODO: Maybe try with best all previous answers. Engage memory of validator?
                    synthetized_answer=final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content,
                    best_worker_model_id=cascade_lvl_models[validator_answer['evaluation']['best_answer']['best_worker_model_id']]['model_name'],
                    cascade_level=current_cascade_level,
                    best_confidence_score=validator_answer['evaluation']['best_answer']['best_confidence_score'],
                    total_cost =cascade_total_cost)
                
                print(f"Synthetized final answer:\n{synthetized_final_answer}")
                print(f"FINAL ANSWER (DIRECT FROM WORKER):\n{final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content}")

                results_file: Path = category_iteration_files[question_category]
                
                save_results_to_jsonl(
                    question_id=question_id,
                    num_questions=NUM_MAX_QUESTIONS,
                    category=question_category,
                    question=original_question,
                    answer=final_answers[validator_answer['evaluation']['best_answer']['best_worker_model_id']].content,
                    cascade_lvl=current_cascade_level,
                    winner_model=cascade_lvl_models[validator_answer['evaluation']['best_answer']['best_worker_model_id']]['model_name'],
                    judge_model=validator_agent.model_name,
                    score=validator_answer['evaluation']['best_answer']['best_confidence_score'],
                    save_to_path_file=results_file
                )
            
                logger.success(f"Cascade level {current_cascade_level} completed.")
                # break


if __name__ == "__main__":
    logger.info('Starting script...')
    main()
    logger.info("Exitting...")
