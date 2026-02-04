import argparse
import json
import uuid
from typing import Any, Optional, Union

from loguru import logger
from mlflow.genai.datasets import EvaluationDataset, search_datasets  # get_dataset

import mlflow

from ..agents.pydantic_agent import WorkingAgent
from ..config.make_config import make_config
from ..datasets.loader import create_or_get_experiment

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

def question_records_to_question_str(question_records: list) -> dict[str, str]:
    """Converts ArenaHardAuto question records to a mapping between question ID and question.

    Args:
        question_records (list): List of ArenaHardAuto question records.

    Returns:
        dict[str, str]: Converted question records to a mapping between question ID and question string value.
    """
    logger.info("Conveting question records to question mapping...")
    return {record['dataset_record_id']: record['inputs']['question'] for record in question_records}

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
        dict[str, WorkingAgent]: Map of working agent ID and 
    """
    worker_agents_dict: dict[str, WorkingAgent] = {}
    
    for worker_key, worker_config in models_config.items():
        logger.info(f"Attempting to instantiate working agent: {worker_key}")
        model_name: str = worker_config.get("model_name", None)
        # short_model_name: str = worker_config.get("short_model_name", "")
        endpoint_struct: dict[str, str] = worker_config.get("endpoint", None)
        parameters_struct: dict[str, Any] = worker_config.get("parameters", None)

        if any(x is None for x in [model_name, endpoint_struct, parameters_struct]):
            logger.warning("Malformed cascade model configuration found. Skipping...")
            continue
            # return None
        
        worker_agents_dict[worker_key] = WorkingAgent(
            model_id=f"{model_name}/{uuid.uuid4()}", # uuid.uuid4(),
            role_name=worker_key,
            system_instruction=WORKING_AGENT_PROMPT,
            cascade_tier=cascade_lvl,
            config=worker_config, # parameters_struct,
            api_key=endpoint_struct.get('api_key', None)
        )
        
        logger.success(f"Instantiated working agent with ID: {worker_agents_dict[worker_key].model_id}")
        
    return worker_agents_dict

async def run_cascade_loop(worker_agents: dict[str, WorkingAgent], prompts: dict[str, str], current_level: int = 1) -> None:
    # TODO: implement
    logger.error("Method not implemented.")
    # raise NotImplementedError


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
    
    questions_mapping: dict[str, str] = question_records_to_question_str(question_records=question_records)
    
    for idx, (question_id, question) in enumerate(questions_mapping.items()):
        print(f"Question {idx + 1}: {question_id} - {question}")
        
    # * STEP 3: Configuring Agents
    
    # TODO: Uncomment experiment setting
    # experiment_id, experiment_name, version = get_or_create_versioned_experiment(experiment_name=EXPERIMENT_NAME)
    # mlflow.set_experiment(experiment_name)
    
    # * Loading complex run config models
    cascade_lvl_models = load_cascade_level_specific_models(config=CASCADE_MODELS_CONFIG, cascade_level=CASCADE_LEVEL, config_key=COMPLEX_RUN_CONFIG_KEY)
    print(f"Current Cascade Level: {CASCADE_LEVEL}")
    print(json.dumps(cascade_lvl_models, indent=2))
    
    # * Instantiate Agents using working models configs
    worker_agents_dict: dict[str, WorkingAgent] = initialize_worker_agents(models_config=cascade_lvl_models, cascade_lvl=CASCADE_LEVEL)
    for worker_key, worker_agent in worker_agents_dict.items():
        print(worker_key, worker_agent.model_id)
    
    run_cascade_loop(worker_agents=worker_agents_dict, prompts=questions_mapping, current_level=CASCADE_LEVEL)
    

if __name__ == "__main__":
    logger.info('Starting script...')
    main()
    logger.info("Exitting...")
