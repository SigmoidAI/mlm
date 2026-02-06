import argparse
import asyncio
import json
import uuid
from typing import Any, Coroutine, Optional, Union

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
    logger.success(f"Created experiment: {experiment_name} (ID: {experiment_id})")
    
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
    
    # table = str.maketrans("\n\t\r", "   ")
    # return f"{left_portion.translate(table)}...{right_portion.translate(table)}"

# async def run_working_multiple_agents(worker_agents: dict[str, WorkingAgent], prompt_questions: dict[str, str]) -> dict[str, AgentResponse]:
#     """Run all working agents on the same cascade level in parallel to generate answers to the same prompt.

#     Args:
#         worker_agents (dict[str, WorkingAgent]): Dictionary mapping agent IDs to WorkingAgent instances.
#         prompt_questions (dict[str, str]): The question/prompt strings to send to appropriate agent.

#     Returns:
#         dict[str, AgentResponse]: Dict of AgentResponse objects mapped to their own agent ID - answers from the agents to the prompt.
#     """
#     tasks = []
#     for worker_id, worker_agent in worker_agents.items():
#         designated_prompt: Prompt = Prompt(content=prompt_questions.get(worker_id), model_tier='complex')
#         logger.info(f"Created Prompt object from prompt question: \"{__format_response(long_string_log=designated_prompt.content)}\" for worker agent: {worker_agent.model_id}")
#         tasks.append(worker_agent.generate(context=designated_prompt))
    
#     logger.info("Created coroutines for working agents answer generation.")
    
#     answers: list[AgentResponse] = await asyncio.gather(*tasks, return_exceptions=True)
    
#     agents_responses: dict[str, AgentResponse] = {}
#     agent_ids: list[str] = list(worker_agents.keys())
    
#     for idx, response in enumerate(answers):
#         agent_id = agent_ids[idx]
#         if isinstance(response, Exception):
#             # Since order of results is preserved by asyncio.gather() function in the same order as awaitables in *aws = *tasks
#             logger.error(f"Agent {agent_id} failed: {response}")
#         else:
#             logger.success(f"Agent {agent_id} succeeded: {__format_response(long_string_log=response.content)}")
#             agents_responses[agent_id] = response
            
#     logger.success(f"Generated {len(agents_responses.keys())} valid responses out of {len(worker_agents)} agents")
#     return agents_responses


async def run_working_agent(worker_agent: tuple[str, WorkingAgent], func: Coroutine, **kwargs) -> tuple[str, AgentResponse]:
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
    
    agent_answer: AgentResponse = await func(**func_kwargs)
    
    if isinstance(agent_answer, Exception):
        logger.error(f"Agent {agent_id} failed: {agent_answer}")
        return (agent_id, None)
    else:
        logger.success(f"Agent {agent_id} succeeded: {__format_response(long_string_log=agent_answer.content)}")
    
    logger.success(f"Generated valid response by agent: {agent_id} - {agent.model_id}")
    return (agent_id, agent_answer)


async def run_cascade_initial_answer(worker_agents: dict[str, WorkingAgent], prompt_data: tuple[str, str], current_level: int = 1) -> Optional[dict[str, AgentResponse]]:
    """Generate initial answers concurrently.

    Args:
        worker_agents (dict[str, WorkingAgent]): List of WorkingAgent object.
        prompt_data (tuple[str, str]): User prompt.
        current_level (int, optional): Current level of the cascade. Defaults to 1.

    Returns:
        Optional[dict[str, AgentResponse]]: Mapping of AgentResponse objects to the user prompt.
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
    
    for response in results:
        agent_id, response_content = response
        if not response_content:
            # Since order of results is preserved by asyncio.gather() function in the same order as awaitables in *aws = *tasks
            logger.error(f"Agent {agent_id} failed: {response_content}")
        else:
            logger.success(f"Agent {agent_id} succeeded: {__format_response(long_string_log=response_content.content)}")
            agents_responses[agent_id] = response_content
            
    logger.success(f"Generated {len(agents_responses.keys())} valid responses out of {len(worker_agents)} agents")
    return agents_responses

# async def run_cascade_initial_answer(worker_agents: dict[str, WorkingAgent], prompt_data: tuple[str, str], current_level: int = 1) -> Optional[dict[str, AgentResponse]]:
#     """Generate initial answers concurrently.

#     Args:
#         worker_agents (dict[str, WorkingAgent]): List of WorkingAgent object.
#         prompt_data (tuple[str, str]): User prompt.
#         current_level (int, optional): Current level of the cascade. Defaults to 1.

#     Returns:
#         Optional[dict[str, AgentResponse]]: Mapping of AgentResponse objects to the user prompt.
#     """
#     prompt_id, prompt_question = prompt_data
#     logger.info(f"Starting initial answer generation at cascade level: {current_level} with: {len(worker_agents.keys())} working agents and with prompt with ID: {prompt_id} - \"{__format_response(long_string_log=prompt_question)}\"")
    
#     prompt_question_dict: dict[str, str] = {
#         agent_id: prompt_question for agent_id in worker_agents.keys()
#     }
    
#     logger.info("Invoking working agents")
#     responses: dict[str, AgentResponse] = await run_working_multiple_agents(
#         worker_agents=worker_agents,
#         prompt_questions=prompt_question_dict
#     )

#     if not responses:
#         logger.error("No valid responses received. Stopping.")
#         return None
    
#     logger.success(f"Initial cascade phase of answer generation completed. Processed prompt with ID: {prompt_id}.")
#     return responses


async def run_cascade_debate(worker_agents: dict[str, WorkingAgent], prev_answers: dict[str, AgentResponse], current_level: int = 1) -> Optional[dict[str, AgentResponse]]:
    """Generate critique/debate responses where each agent reviews peer responses.

    Args:
        worker_agents (dict[str, WorkingAgent]): Dictionary of WorkingAgent instances.
        prev_answers (dict[str, AgentResponse]): Previous round's responses to critique.
        current_level (int, optional): Current cascade level. Defaults to 1.

    Returns:
        Optional[dict[str, AgentResponse]]: Dict of critique AgentResponse objects mapped to their generator agent ID.
    """
    logger.info(f"Starting debate process at cascade level: {current_level} with: {len(worker_agents.keys())} working agents")
    
    logger.info(f"Generating critiques based on {len(prev_answers)} previous responses")
    if not prev_answers:
        logger.error("No previous answers provided for debate.")
        return None

    tasks = []
    for agent_id, worker_agent in worker_agents.items():
        logger.info(f"Attempting to generate critique response by agent: {agent_id} - {worker_agent.agent}")
        print(f"{prev_answers[agent_id].author_id} - {agent_id}")
        individual_peer_responses: list[Prompt] = [
            Prompt(content=peer_response.content, model_tier="complex")
            for peer_id, peer_response in prev_answers.items()
            if peer_id != agent_id
        ]
        
        print(
            json.dumps(
                [
                    {
                        "content": p.content,
                        "model_tier": p.model_tier,
                    }
                    for p in individual_peer_responses
                ],
                indent=4
            )
        )
                
        if not individual_peer_responses:
            logger.warning(f"Agent {agent_id} has no peer responses to critique")
            continue
    
        func_kwargs = {
            "peer_responses": individual_peer_responses
        }
        
        logger.info(f"Agent {agent_id} will critique {len(individual_peer_responses)} ({" ".join([peer_response.author_id for peer_id, peer_response in prev_answers.items() if peer_id != agent_id])}) peer response(s)")
        
        tasks.append(
            run_working_agent(
                worker_agent=(agent_id, worker_agent),
                func=worker_agent.generate_critique,
                **func_kwargs
            )
        )
        
    logger.info(f"Executing {len(tasks)} critique tasks in parallel")
    critiques: list[tuple[str, AgentResponse]] = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_critiques: dict[str, AgentResponse] = {}
    
    for critique in critiques:
        agent_id, critique_content = critique
        if not critique_content:
            # Since order of results is preserved by asyncio.gather() function in the same order as awaitables in *aws = *tasks
            logger.error(f"Agent {agent_id} critique failed: {critique}")
        else:
            logger.success(f"Agent {agent_id} generated critique: {__format_response(long_string_log=critique_content.content)}")
            valid_critiques[agent_id] = critique_content
            
    logger.success(f"Debate completed. Generated {len(valid_critiques.keys())} valid critiques out of {len(worker_agents)} agents")
    return valid_critiques


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
                                      current_level: int = 1) -> Optional[dict[str, AgentResponse]]:
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
    
    refinement_results: list[tuple[str, AgentResponse]] = await asyncio.gather(*tasks, return_exceptions=True)

    agents_responses: dict[str, AgentResponse] = {}
    
    for refinement_result in refinement_results:
        agent_id, refinement_result_content = refinement_result
        if not refinement_result_content:
            # Since order of results is preserved by asyncio.gather() function in the same order as awaitables in *aws = *tasks
            logger.error(f"Agent {agent_id} refinement failed: {refinement_result_content}")
        else:
            logger.success(f"Agent {agent_id} refinement succeeded: {__format_response(long_string_log=refinement_result_content.content)}")
            agents_responses[agent_id] = refinement_result_content
            
    logger.success(f"Generated {len(agents_responses.keys())} valid refined responses out of {len(worker_agents)} agents")
    return agents_responses


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
    
    experiment_id, experiment_name, version = get_or_create_versioned_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_name)
    
    
    # * Instantiate Judge Agent
    # TODO: create ValidatorAgent.
    
    
    for current_cascade_level in range(1, CASCADE_LEVEL + 1):
        logger.info(f"Entering cascade level {current_cascade_level}.")
        
        # * STEP 3: Configuring Agents
        # * Loading complex run config models
        cascade_lvl_models = load_cascade_level_specific_models(config=CASCADE_MODELS_CONFIG, cascade_level=current_cascade_level, config_key=COMPLEX_RUN_CONFIG_KEY)
        print(json.dumps(cascade_lvl_models, indent=2))
        
        # * Instantiate Agents using working models configs
        worker_agents_dict: dict[str, WorkingAgent] = initialize_worker_agents(models_config=cascade_lvl_models, cascade_lvl=CASCADE_LEVEL)
        # for worker_key, worker_agent in worker_agents_dict.items():
        #     print(worker_key, worker_agent.agent)
        
        for idx, (question_id, question_data) in enumerate(questions_mapping.items()):
            question_prompt = question_data['question']
            question_category = question_data['category']
            logger.info(f"[{idx + 1}/{len(questions_mapping.keys())}] Processing prompt with ID: {question_id} - {question_category} - \"{question_prompt}\"")
            
            with mlflow.start_run(run_name=question_id):
                mlflow.log_params({
                    "question_id": question_id,
                    "category": question_category,
                    "config_key": COMPLEX_RUN_CONFIG_KEY,
                    "cascade_level": current_cascade_level,
                    "num_models": len(worker_agents_dict.keys()),
                    "models": [model_name['short_model_name'] for model_name in cascade_lvl_models.values()],
                    "version": version
                })
                
                prompt_data: tuple[str, str] = (question_id, question_prompt)
                
                # * STEP 4: Working Agents actions:
                # * Generate initial responses    
                initial_answers: dict[str, AgentResponse] = asyncio.run(run_cascade_initial_answer(worker_agents=worker_agents_dict, prompt_data=prompt_data, current_level=current_cascade_level))
                
                print(f"Question {idx}: {question_id} - {questions_mapping[question_id]}")
                for agent_initial_answer in initial_answers.values():
                    print(f"\nAgent {agent_initial_answer.author_id}: \n\tResponse: {agent_initial_answer.content}")
            
                # * Generate Critiques/Debate prompts
                critiques: dict[str, AgentResponse] = asyncio.run(run_cascade_debate(worker_agents=worker_agents_dict, prev_answers=initial_answers, current_level=current_cascade_level))
                
                for agent_critique in critiques.values():
                    print(f"\nAgent {agent_critique.author_id}: \n\tResponse: {agent_critique.content}")

                # * Generate refined final answers based on previous initial answers and critiques
                final_answers: dict[str, AgentResponse] = asyncio.run(
                    run_cascade_refinement_loop(
                        worker_agents=worker_agents_dict, 
                        init_question=question_prompt,
                        init_answers=initial_answers, 
                        critiques=critiques, 
                        current_level=current_cascade_level
                    )
                )
                
                for agent_final_answer in final_answers.values():
                    print(f"\nAgent {agent_final_answer.author_id}: \n\tResponse: {agent_final_answer.content}")
                
                # * STEP 5: Integrate Judge Agent to select the best final answer.
                
        logger.success(f"Cascade level {current_cascade_level} completed.")


if __name__ == "__main__":
    logger.info('Starting script...')
    main()
    logger.info("Exitting...")
