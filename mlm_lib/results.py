"""
results.py — RunResult and supporting record dataclasses.

All data returned by CascadeRunner.run() and CascadeRunner.run_batch()
lives here so downstream code can import just what it needs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .settings import RunType


@dataclass
class IterationRecord:
    """One model attempt within a Simple cascade run."""
    iteration: int
    model_key: str
    model_name: str
    answer: str
    passed: bool
    score: float
    verdict: str
    reason: str
    model_cost: Dict[str, float]
    judge_cost: Dict[str, float]
    iteration_total_cost: float


@dataclass
class CascadeLevelRecord:
    """One full level (init → debate → refine → judge) within a Complex run."""
    level: int
    initial_answers: Dict[str, str]
    critiques: Dict[str, str]
    final_answers: Dict[str, str]
    judge_evaluation: Dict[str, Any]
    best_model_id: str
    best_score: float
    passed: bool
    level_cost: float


@dataclass
class RunResult:
    """
    Unified result returned by CascadeRunner.run().

    Fields common to both run types
    --------------------------------
    run_type            "simple" or "complex"
    question_id         identifier passed to run()
    question            original question text
    answer              best answer found
    success             True if score >= acceptable_score
    total_cost          cumulative USD cost (model + judge calls)

    Simple-only fields
    ------------------
    iterations          which model iteration produced the winner
    winning_model       slot name (e.g. "model_1")
    winning_model_name  full model string (e.g. "gpt-4o")
    iteration_history   list[IterationRecord]

    Complex-only fields
    -------------------
    winning_cascade_level   level where an acceptable answer was found
    best_confidence_score   judge score of the winning answer
    winning_model           slot name of the winning worker
    winning_model_name      full model string of the winning worker
    cascade_level_history   list[CascadeLevelRecord]
    """
    run_type: RunType
    question_id: str
    question: str
    answer: str
    success: bool
    total_cost: float

    # Simple
    iterations: int = 0
    winning_model: str = ""
    winning_model_name: str = ""
    iteration_history: List[IterationRecord] = field(default_factory=list)

    # Complex
    winning_cascade_level: int = 0
    best_confidence_score: float = 0.0
    cascade_level_history: List[CascadeLevelRecord] = field(default_factory=list)

    # Arbitrary extra metadata callers may attach
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.run_type == RunType.SIMPLE:
            return (
                f"[simple] id={self.question_id} success={self.success} "
                f"iter={self.iterations} model={self.winning_model_name} "
                f"cost=${self.total_cost:.4f}"
            )
        return (
            f"[complex] id={self.question_id} success={self.success} "
            f"level={self.winning_cascade_level} score={self.best_confidence_score:.4f} "
            f"model={self.winning_model_name} cost=${self.total_cost:.4f}"
        )