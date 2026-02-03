import argparse
from typing import Any, Optional, Union

from loguru import logger
from mlflow.genai.datasets import EvaluationDataset, get_dataset, search_datasets

import mlflow

from ..config.make_config import make_config
from ..datasets.loader import create_or_get_experiment

logger.info("Configuring defined variables...")

# * HARDCODED VALUES SPECIFIC TO COMPLEX MODEL WORKFLOW
# * ASSUMES:
# * 1. USER PROMPT IS GIVEN (IN THIS CONTEXT, TAKEN DYNAMICALLY FROM EXPERIMENT DATASET)
# * 2. SELECTED COMPLEXITY IS HIGH
USER_PROMPT: str = ...
IS_COMPLEX: bool = True

# EXPERIMENT DATASET
DATASET_EXPERIMENT_NAME: str = "DATASET_Arena_Hard"
MLFLOW_DATASET_NAME: str = "arena_hard_auto"

# CONFIGURATION

# CASCADE MODELS CONFIG
CASCADE_MODELS_CONFIG: dict[str, str] = make_config()

# CASCADE LEVEL
CASCADE_LEVEL: int = 1

# MLFLOW URI
MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000" # os.getenv(key="MLFLOW_TRACKING_URI")

def set_up_mlflow(mlflow_uri: str) -> None:
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
    
    # # Previous version of dataset retrieval - not working with experiment_id
    # dataset = get_dataset(dataset_id=experiment_id)
    # records = dataset.to_dict()
    
    # return records

def get_question_records_from_dataset(dataset: list, record_idx: Union[range, int]) -> Union[dict, list, None]:
    """Retrieve question record(s) from dataset by index or range.

    Args:
        dataset (list): List of dataset records.
        record_idx (Union[range, int]): Single index (int) or range of indices (range).

    Returns:
        Union[dict, list, None]: list of record(s), None if invalid.
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
                f"Dataset size: {dataset_size}. Returning available records."
            )
            return None
        
        # Extract subset using range
        logger.success(f"Found question records in range: {record_idx}")
        return dataset[record_idx.start:record_idx.step:record_idx.stop]
    
    # Handle invalid type
    else:
        logger.error(
            f"Argument record_idx must be int or range, "
            f"got: {type(record_idx).__name__}"
        )
        return None



def main() -> None:
    # parser = argparse.ArgumentParser()
    
    # * STEP 1: Setting up MLFlow
    set_up_mlflow(mlflow_uri=MLFLOW_TRACKING_URI)
    
    # * Retrieving Dataset from MLFlow
    dataset_records: dict[str, Any] = get_experiment_dataset(experiment_name=DATASET_EXPERIMENT_NAME)['records']
    if not dataset_records:
        logger.error("Dataset was not found.")
        return
    
    # * STEP 2: Extracting question records from the dataset
    
    # * SCENARIO 1: ENTIRE MLFLOW DATASET OF QUESTIONS
    # question_records_idx: int = range(0, len(dataset_records))
    # question_records = get_questions_from_dataset(dataset=dataset_records, record_idx=range(0, len(dataset_records)))
    # print(question_records)
    # if not question_records:
    #     logger.error("Question records not found.")
    #     return
    
    # * SCENARIO 2: SUBSET OF QUESTIONS FROM MLFLOW DATASET
    # question_records_idx: int = range(0, 11)
    # question_records = get_questions_from_dataset(dataset=dataset_records, record_idx=range(0, 11))
    # print(question_records)
    # if not question_records:
    #     logger.error("Question records not found.")
    #     return
    
    # * SCENARIO 3: CUSTOM QUESTION
    # question_record = [USER_PROMPT]
    
    # * SCENARIO 4: SINGLE QUESTION FROM MLFLOW DATASET
    question_records_idx: int = 1
    question_record = get_question_records_from_dataset(dataset=dataset_records, record_idx=question_records_idx)[0]
    if not question_record:
        logger.error("Question record not found.")
        return
    
    print(f"Question {question_records_idx}: {question_record['inputs']['question']}")
    


if __name__ == "__main__":
    logger.info('Starting script...')
    main()
    logger.info("Exitting...")
