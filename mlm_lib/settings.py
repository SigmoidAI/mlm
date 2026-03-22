import os
"""
config.py — All configuration dataclasses for CascadeRunner.

Import what you need:
    from cascade_lib.config import CascadeRunnerConfig, SimpleConfig, ComplexConfig, ...
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RunType(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class TemperatureStrategy(str, Enum):
    """Controls how worker-agent temperature is assigned per question."""
    FIXED = "fixed"                  # single value for all workers / levels
    RANDOM = "random"                # random sample from hard_prompt_temp_range
    CATEGORY_AWARE = "category_aware"  # different ranges per question category


# ---------------------------------------------------------------------------
# Default prompts (module-level constants so callers can reuse/extend them)
# ---------------------------------------------------------------------------

DEFAULT_SIMPLE_SYSTEM_PROMPT: str = (
    "You are a helpful AI assistant. Provide detailed, accurate answers."
)

DEFAULT_COMPLEX_SYSTEM_PROMPT: str = (
    "You are a precise, focused AI assistant. Answer ONLY what is asked with verifiable, "
    "accurate information. When receiving peer-review feedback, apply only refinements that "
    "enhance precision and relevance. Stay focused and surgically precise."
)

DEFAULT_REFINEMENT_PROMPT_TEMPLATE: str = (
    "The previous answer did not fully meet the requirements.\n\n"
    "QUESTION:\n{question}\n\n"
    "PREVIOUS ANSWER (Attempt {attempt}):\n{prev_answer}\n\n"
    "FEEDBACK ON PREVIOUS ANSWER:\n{issue}\n\n"
    "Please revise your answer to address the feedback above. Ensure your response is "
    "thorough, accurate, and covers all aspects of the question."
)

DEFAULT_JUDGE_PROMPT: str = (
    "Worker agents provided the following answers to the initial question.\n"
    "Analyze the question and their answers and vote for the best answer.\n"
    "Be impartial and objective.\n\n"
    "Evaluation Criteria: task completion, accuracy, relevance, quality, "
    "non-deprecated solution components.\n\n"
    "Score each answer 0.0000–1.0000. Respond strictly as:\n"
    "```json\n"
    "\"evaluation\": {\n"
    "    \"question\": <question>,\n"
    "    \"best_answer\": {\n"
    "        \"best_worker_model_id\": <id>,\n"
    "        \"best_confidence_score\": <float_4dp>,\n"
    "        \"best_reason\": <reason>\n"
    "    },\n"
    "    \"individual_reviews\": {\n"
    "        \"worker_model_<id>\": {\"confidence_score\": <float_4dp>, \"reason\": <reason>}\n"
    "    }\n"
    "}\n"
    "```"
)

DEFAULT_NEXT_LEVEL_PROMPT_PREFIX: str = (
    "Previous cascade level worker agents did not succeed in answering the user question "
    "properly. Analyze the question and their answers and generate better results:"
)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class JudgeConfig:
    """
    Configuration for the judge / validator agent.

    Attributes
    ----------
    judge_model_key : str
        Key in the ``judge_models`` section of cascade_models.yaml.
    acceptable_score : float
        Minimum score (0–1) for an answer to be considered passing.
    agent : ValidatorAgent, optional
        Pass a pre-built ValidatorAgent to skip YAML lookup entirely.
    """
    judge_model_key: str = "judge_model_1"
    acceptable_score: float = 0.92
    agent: Optional[Any] = None          # ValidatorAgent; typed as Any to avoid circular import


@dataclass
class SimpleConfig:
    """
    Configuration for the Simple cascade workflow.

    The simple flow tries models in order (as defined in the YAML).
    After each failed attempt the prompt is enriched with feedback and
    passed to the next model. The best-scoring valid answer is returned.

    Attributes
    ----------
    model_config_key : str
        Top-level key in cascade_models.yaml, e.g. ``"simple_flow"``.
    model_overrides : dict
        Runtime overrides keyed by slot name (e.g. ``"model_1"``).
        Values are full model-config dicts that are merged (updated) over
        the YAML entry.  Pass a new slot name to add an extra model.
    system_prompt : str
        System instruction sent to every worker agent.
    refinement_prompt_template : str
        Template used when the previous answer did not pass.
        Available placeholders: ``{question}``, ``{attempt}``,
        ``{prev_answer}``, ``{issue}``.
    exhaust_all_models : bool
        When True, all models are tried regardless of early success.
        Useful for benchmarking; the highest-scoring valid answer wins.
    """
    model_config_key: str = "simple_flow"
    model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_prompt: str = DEFAULT_SIMPLE_SYSTEM_PROMPT
    refinement_prompt_template: str = DEFAULT_REFINEMENT_PROMPT_TEMPLATE
    exhaust_all_models: bool = False


@dataclass
class ComplexConfig:
    """
    Configuration for the Complex multi-level debate cascade.

    Each level runs: initial generation → peer debate → refinement → judge.
    If the judge score is below ``acceptable_score`` the ensemble of answers
    and reviews becomes the prompt for the next level.

    Attributes
    ----------
    model_config_key : str
        Top-level key in cascade_models.yaml, e.g. ``"cascade_complex_run"``.
    max_cascade_levels : int
        How many levels to attempt before giving up and returning the best
        answer found so far.
    level_model_overrides : dict
        Per-level, per-slot overrides.
        Format: ``{level_number: {slot_name: model_config_dict}}``.
    system_prompt : str
        System instruction sent to every worker agent at every level.
    temperature_strategy : TemperatureStrategy
        How worker temperatures are assigned — FIXED, RANDOM, or CATEGORY_AWARE.
    fixed_temperature : float
        Used only when strategy is FIXED.
    hard_prompt_temp_range : (float, float)
        (low, high) range for ``hard_prompt`` category questions.
    creative_temp_range : (float, float)
        (low, high) range for ``creative_writing`` category questions.
    judge_prompt : str
        Full prompt prefix sent to the judge together with worker answers.
    next_level_prompt_prefix : str
        Prefix prepended to the ensemble of answers when escalating to the
        next cascade level.
    """
    model_config_key: str = "cascade_complex_run"
    max_cascade_levels: int = 5
    level_model_overrides: Dict[int, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    system_prompt: str = DEFAULT_COMPLEX_SYSTEM_PROMPT
    temperature_strategy: TemperatureStrategy = TemperatureStrategy.CATEGORY_AWARE
    fixed_temperature: float = 0.7
    hard_prompt_temp_range: Tuple[float, float] = (0.3, 0.7)
    creative_temp_range: Tuple[float, float] = (0.6, 1.4)
    judge_prompt: str = DEFAULT_JUDGE_PROMPT
    next_level_prompt_prefix: str = DEFAULT_NEXT_LEVEL_PROMPT_PREFIX


@dataclass
class MLflowConfig:
    """
    MLflow experiment tracking settings.

    Attributes
    ----------
    tracking_uri : str
        URI of the MLflow tracking server.
    experiment_base_name : str
        Base name for experiments; a version suffix (_v1, _v2, …) is
        appended automatically when auto_version is True.
    auto_version : bool
        When True, a new versioned experiment is created for each
        CascadeRunner instance.
    enabled : bool
        Set to False to disable all MLflow logging (useful in tests or
        when running without an MLflow server).
    autolog_pydantic : bool
        Whether to call ``mlflow.pydantic_ai.autolog()`` on setup.
    """
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/")
    experiment_base_name: str = "CascadeRunner"
    auto_version: bool = True
    enabled: bool = True
    autolog_pydantic: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class CascadeRunnerConfig:
    """
    Top-level configuration object for CascadeRunner.

    All sub-configs have sensible defaults — only override what you need.

    Quick examples
    ~~~~~~~~~~~~~~
    Minimum (all defaults, simple flow):

        cfg = CascadeRunnerConfig()

    Complex run, 3 levels, strict threshold:

        cfg = CascadeRunnerConfig(
            run_type=RunType.COMPLEX,
            judge=JudgeConfig(acceptable_score=0.95),
            complex=ComplexConfig(max_cascade_levels=3),
        )

    Custom system prompts + no MLflow:

        cfg = CascadeRunnerConfig(
            simple=SimpleConfig(system_prompt="Be terse."),
            mlflow=MLflowConfig(enabled=False),
            openrouter_api_key="sk-or-...",
        )
    """
    run_type: RunType = RunType.SIMPLE

    # Path to cascade_models.yaml — auto-detected from the repo tree if None.
    models_yaml_path: Optional[str] = None

    judge: JudgeConfig = field(default_factory=JudgeConfig)
    simple: SimpleConfig = field(default_factory=SimpleConfig)
    complex: ComplexConfig = field(default_factory=ComplexConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # OpenRouter API key — falls back to OPENROUTER_API_KEY env var if None.
    openrouter_api_key: Optional[str] = None