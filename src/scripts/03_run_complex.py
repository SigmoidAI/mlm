import argparse
import asyncio
import json
import uuid
from typing import Any, Optional, Union

from loguru import logger
from mlflow.genai.datasets import EvaluationDataset, search_datasets  # get_dataset

import mlflow

from ..agents.pydantic_agent import WorkingAgent
from ..config.make_config import make_config
from ..datasets.loader import create_or_get_experiment
from ..models.schemas import AgentResponse, Prompt

logger.info("Configuring defined variables...")

# * HARDCODED VALUES SPECIFIC TO COMPLEX MODEL WORKFLOW
# * ASSUMES:
# * 1. USER PROMPT IS GIVEN (IN THIS CONTEXT, TAKEN DYNAMICALLY FROM EXPERIMENT DATASET)
# * 2. SELECTED COMPLEXITY IS HIGH
USER_PROMPT: str = {
    "dataset_record_id": uuid.uuid4(),
    "inputs": {
        "question": "What is the capital of Great Britain?"
    }
}
IS_COMPLEX: bool = True

# EXPERIMENT DATASET
DATASET_EXPERIMENT_NAME: str = "DATASET_Arena_Hard"
MLFLOW_DATASET_NAME: str = "arena_hard_auto"

# CONFIGURATION

# WORKER PROMPT
WORKING_AGENT_PROMPT: str = "You are a helpful AI assistant. Provide detailed, accurate answers."

# CASCADE MODELS CONFIG
CASCADE_MODELS_CONFIG: dict[str, str] = make_config()
COMPLEX_RUN_CONFIG_KEY: str = "cascade_complex_run"

# CASCADE LEVEL
CASCADE_LEVEL: int = 1

# MLFLOW
MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000" # os.getenv(key="MLFLOW_TRACKING_URI")
EXPERIMENT_NAME: str = "complex_workflow_run"

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
    
def get_experiment_dataset(experiment_name: str = DATASET_EXPERIMENT_NAME, dataset_name: str = MLFLOW_DATASET_NAME) -> Optional[dict[str, Any]]:
    """Retrieves experiment dataset by its MLFlow experiment name.

    Args:
        experiment_name (str, optional): Name of the MLFlow Experiment containing a dataset. Defaults to DATASET_EXPERIMENT_NAME = DATASET_Arena_Hard.
        dataset_name (str, optional): Name of the Dataset in MLFlow Experiment. Defaults to MLFLOW_DATASET_NAME = arena-hard-auto.

    Returns:
        Optional[dict[str, Any]]: Dict representation of the dataset from MLFlow Experiment. If no dataset is found, returns None.
    """
    experiment_id: str = create_or_get_experiment(experiment_name=experiment_name)
    logger.info(f"Looking for MLFlow Experiment Dataset with Experiment ID: {experiment_id}.")
    
    # TODO: Check another way to retrieve dataset by experiment ID.
    datasets_list: list[EvaluationDataset] = search_datasets(experiment_ids=experiment_id)
    if not datasets_list:
        logger.warning(f"No datasets found in MLFlow Experiment with Experiment ID: {experiment_id}")
        return None
    
    logger.success(f"Found: {len(datasets_list)} datasets in MLFlow Experiment with Experiment ID: {experiment_id}")
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
    logger.success(f"Created experiment: {experiment_name}_v{next_version} (ID: {experiment_id})")
    
    return experiment_id, experiment_name, next_version

# TODO: check alternative models loading mechanism. Not sure if it is viable.
# def load_models(config: dict[str, Any], config_key: str = "cascade_complex_run") -> dict[str, Any]:
#     return config[config_key]

def load_cascade_level_specific_models(config: dict[str, Any], 
                                       cascade_level: int, 
                                       config_key: str) -> Optional[dict[str, Any]]:
    """Dynamically load from configuration file and extract specific cascade level models.

    Args:
        config (dict[str, Any]): Configuration file in dict format.
        cascade_level (int, optional): Cascade level to look for models.
        config_key (str, optional): Key of the complex cascade subset of models.

    Returns:
        Optional[tuple[dict[str, Any], int]]: Specific cascade models configuration at a specific cascade level and corresponding cascade level as a tuple. None if config entry or cascade models not found.
    """
    logger.info(f"Retrieving cascade models configuration: {config_key} at level: {cascade_level}.")
    cascade_models_config: dict[str, Any] = config.get(config_key, {})
    
    if not cascade_models_config:
        logger.error(f"No config entry with key: {config_key}.")
        return None
    logger.success(f"Succefully retrieved config entry with key: {config_key}")
    
    key = f"cascade_lvl_{cascade_level}"
    logger.info(f"Looking for cascade level: {key} in configuration file.")

    if key not in cascade_models_config:
        logger.error(f"No cascade level: {key} found in configuration file.")
        return None
    
    logger.success(f"Succesfully retrieved cascade models configuration at level: {cascade_level}")
    return cascade_models_config[key]


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
        logger.info(f"Attempting to instantiate working agent: {worker_key}")
        model_name: str = worker_config.get("model_name", None)
        endpoint_struct: dict[str, str] = worker_config.get("endpoint", None)
        parameters_struct: dict[str, Any] = worker_config.get("parameters", None)

        if any(x is None for x in [model_name, endpoint_struct, parameters_struct]):
            logger.warning("Malformed cascade model configuration found. Skipping...")
            continue
            # return None
        
        worker_agents_dict[worker_key] = WorkingAgent(
            # model_id=f"{model_name}/{uuid.uuid4()}", # uuid.uuid4(),
            model_id=model_name,
            role_name=worker_key,
            system_instruction=WORKING_AGENT_PROMPT,
            cascade_tier=cascade_lvl,
            config=worker_config, # parameters_struct,
            api_key=endpoint_struct.get('api_key', None)
        )
        
        logger.success(f"Instantiated working agent with ID: {worker_agents_dict[worker_key].model_id}")
        
    return worker_agents_dict

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
    table = str.maketrans("\n\t\r", "   ")
    return f"{left_portion.translate(table)}...{right_portion.translate(table)}"

async def run_working_agents(worker_agents: dict[str, WorkingAgent], prompt_question: str) -> list[AgentResponse]:
    """Run all working agents on the same cascade level in parallel to generate answers to the same prompt.

    Args:
        worker_agents (dict[str, WorkingAgent]): Dictionary mapping agent IDs to WorkingAgent instances.
        prompt (str): The question/prompt string to send to all agents

    Returns:
        list[AgentResponse]: List of AgentResponse objects - answers from the agents to the prompt.
    """
    prompt_obj = Prompt(content=prompt_question, model_tier="complex")
    logger.info(f"Created Prompt object from prompt question: {__format_response(long_string_log=prompt_question)}")
    
    logger.info("Created coroutines for working agents answer generation.")
    tasks = [
        worker_agent.generate(context=prompt_obj) for worker_agent in worker_agents.values()
    ]
    
    answers: list[AgentResponse] = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_answers: list[AgentResponse] = []
    for idx, response in enumerate(answers):
        agent_id = list(worker_agents.keys())[idx]
        if isinstance(response, Exception):
            # Since order of results is preserved by asyncio.gather() function in the same order as awaitables in *aws = *tasks
            logger.error(f"Agent {agent_id} failed: {response}")
        else:
            logger.success(f"Agent {agent_id} succeeded: {__format_response(long_string_log=response.content)}")
            valid_answers.append(response)
    logger.success(f"Generated {len(valid_answers)} valid responses out of {len(worker_agents)} agents")
    return valid_answers


async def run_cascade_initial_answer(worker_agents: dict[str, WorkingAgent], prompt_data: tuple[str, str], current_level: int = 1) -> Optional[list[AgentResponse]]:
    """Generate initial answers concurrently.

    Args:
        worker_agents (dict[str, WorkingAgent]): List of WorkingAgent object.
        prompt_data (tuple[str, str]): User prompt.
        current_level (int, optional): Current level of the cascade. Defaults to 1.

    Returns:
        Optional[list[AgentResponse]]: List of AgentResponse objects to the user prompt.
    """
    prompt_id, prompt_question = prompt_data
    logger.info(f"Starting initial answer generation at cascade level: {current_level} with: {len(worker_agents.keys())} working agents and with prompt with ID: {prompt_id} - \"{__format_response(long_string_log=prompt_question)}\"")
    
    logger.info("Invoking working agents")
    responses = await run_working_agents(
        worker_agents=worker_agents,
        prompt_question=prompt_question
    )

    if not responses:
        logger.error("No valid responses received. Stopping.")
        return None
    
    logger.success(f"Initial cascade phase of answer generation completed. Processed prompt with ID: {prompt_id}.")
    return responses

async def run_cascade_debate(
    worker_agents: dict,
    prev_answers: list[AgentResponse],
    current_level: int = 1
) -> Optional[list[AgentResponse]]:
    """Generate critique/debate responses where each agent reviews peer responses.

    Args:
        worker_agents (dict): Dictionary of WorkingAgent instances.
        prev_answers (list[AgentResponse]): Previous round's responses to critique.
        current_level (int, optional): Current cascade level. Defaults to 1.

    Returns:
        Optional[list[AgentResponse]]: List of critique AgentResponse objects
    """
    logger.info(
        f"Starting debate process at cascade level: {current_level} "
        f"with: {len(worker_agents.keys())} working agents"
    )
    logger.info(f"Generating critiques based on {len(prev_answers)} previous responses")
    
    if not prev_answers:
        logger.error("No previous answers provided for debate.")
        return None
    
    critique_tasks = []
    agent_ids_order = []
    
    for agent_id, worker_agent in worker_agents.items():
        # Get peer responses (excluding own response if present)
        peer_responses = [r for r in prev_answers if r.author_id != agent_id]
        
        if not peer_responses:
            logger.warning(f"Agent {agent_id} has no peer responses to critique")
            continue
        
        logger.info(f"Agent {agent_id} will critique {len(peer_responses)} peer response(s)")
        agent_ids_order.append(agent_id)
        critique_tasks.append(worker_agent.generate_critique(peer_responses))
    
    if not critique_tasks:
        logger.error("No critique tasks created. Stopping debate.")
        return None
    
    logger.info(f"Executing {len(critique_tasks)} critique tasks in parallel")
    critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
    
    valid_critiques = []
    for idx, critique in enumerate(critiques):
        agent_id = agent_ids_order[idx]
        if isinstance(critique, Exception):
            logger.error(f"Agent {agent_id} critique failed: {critique}")
        else:
            logger.success(f"Agent {agent_id} generated critique")
            valid_critiques.append(critique)
    
    if not valid_critiques:
        logger.error("No valid critiques generated.")
        return None
    
    logger.success(
        f"Debate completed. Generated {len(valid_critiques)} valid critiques "
        f"out of {len(worker_agents)} agents"
    )
    return valid_critiques
    

def main() -> None:
    # parser = argparse.ArgumentParser()
    
    # * STEP 1: Setting up MLFlow
    perform_initial_mlflow_setup(mlflow_uri=MLFLOW_TRACKING_URI)
    
    # * Retrieving Dataset from MLFlow
    dataset_records: dict[str, Any] = get_experiment_dataset(experiment_name=DATASET_EXPERIMENT_NAME)['records']
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
    
    # * SCENARIO 2: SUBSET OF QUESTIONS (10) FROM MLFLOW DATASET
    # question_records_idx: int = range(0, 11)
    # question_records: list = get_question_records_from_dataset(dataset=dataset_records, record_idx=question_records_idx)
    # if not question_records:
    #     logger.error("Question records not found.")
    #     return
    
    # * SCENARIO 3: CUSTOM QUESTION
    # question_records: list = [USER_PROMPT]
    
    # * SCENARIO 4: SINGLE QUESTION FROM MLFLOW DATASET
    question_records_idx: int = 1
    question_records: list = get_question_records_from_dataset(dataset=dataset_records, record_idx=question_records_idx)
    if not question_records:
        logger.error("Question record not found.")
        return
    
    questions_mapping: dict[str, dict[str, str]] = question_records_to_question_str(question_records=question_records)
    
    for idx, (question_id, question) in enumerate(questions_mapping.items()):
        print(f"Question {idx + 1}: {question_id} - {question.get('category')} - {question.get('question')}")
    
    # TODO: Uncomment experiment setting
    experiment_id, experiment_name, version = get_or_create_versioned_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_name)
    
    for current_cascade_level in range(1, CASCADE_LEVEL + 1):
        logger.info(f"Entering cascade level {current_cascade_level}.")
        
        # * STEP 3: Configuring Agents
        
        # * Loading complex run config models
        cascade_lvl_models = load_cascade_level_specific_models(config=CASCADE_MODELS_CONFIG, cascade_level=current_cascade_level, config_key=COMPLEX_RUN_CONFIG_KEY)
        print(json.dumps(cascade_lvl_models, indent=2))
        
        # * Instantiate Agents using working models configs
        worker_agents_dict: dict[str, WorkingAgent] = initialize_worker_agents(models_config=cascade_lvl_models, cascade_lvl=CASCADE_LEVEL)
        for worker_key, worker_agent in worker_agents_dict.items():
            print(worker_key, worker_agent.model_id)
        
        # * STEP 4: Working Agents actions
        for idx, (question_id, question_data) in enumerate(questions_mapping.items()):
            question_prompt = question_data['question']
            question_category = question_data['category']
            logger.info(f"[{idx + 1}/{len(questions_mapping.keys())}] Processing prompt with ID: {question_id} - {question_category} - \"{question_prompt}\"")
            
            with mlflow.start_run(run_name=question_id):
                mlflow.log_params({
                    "question_id": question_id,
                    "category": question_category,
                    "config_key": COMPLEX_RUN_CONFIG_KEY,
                    "num_models": len(worker_agents_dict.keys()),
                    "models": [model_name['short_model_name'] for model_name in cascade_lvl_models.values()],
                    "version": version
                })
                
                prompt_data: tuple[str, str] = (question_id, question_prompt)
                
                # * Generate initial responses    
                answers: list[AgentResponse] = asyncio.run(run_cascade_initial_answer(worker_agents=worker_agents_dict, prompt_data=prompt_data, current_level=current_cascade_level))
                
                print(f"Question {idx}: {question_id} - {questions_mapping[question_id]}")
                for agents_answers in answers:
                    print(f"\nAgent {agents_answers.author_id}: \n\tResponse: {agents_answers.content}")
            
                # * Generate Critiques/Debate prompts
                answers: list[AgentResponse] = asyncio.run(run_cascade_debate(worker_agents=worker_agents_dict, prev_answers=answers, current_level=current_cascade_level))
                
                for agents_answers in answers:
                    print(f"\nAgent {agents_answers.author_id}: \n\tResponse: {agents_answers.content}")
        
        logger.success(f"Cascade level {current_cascade_level} completed.")


if __name__ == "__main__":
    logger.info('Starting script...')
    main()
    logger.info("Exitting...")
