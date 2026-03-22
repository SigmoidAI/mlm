"""
runner.py — CascadeRunner

The single entry-point for both Simple and Complex cascade evaluation runs.
Import it via the package:

    from cascade_lib import CascadeRunner
"""

from __future__ import annotations

import asyncio
import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import yaml

from .settings import (
    CascadeRunnerConfig,
    RunType,
    TemperatureStrategy,
)
from .cost import COST_TRACKER
from .results import CascadeLevelRecord, IterationRecord, RunResult

# ---------------------------------------------------------------------------
# Local project imports — resolved at runtime so cascade_lib can live anywhere
# relative to the rest of the project tree.
# ---------------------------------------------------------------------------
from src.agents.pydantic_agent import ValidatorAgent, WorkingAgent
from src.models.schemas import AgentResponse, Prompt

# ---------------------------------------------------------------------------
# Suppress a known openai/httpx __del__ bug that produces noisy tracebacks
# when AsyncHttpxClientWrapper is garbage-collected before the event loop
# closes. This is a cosmetic issue only — no functionality is affected.
# https://github.com/openai/openai-python/issues/1426
# ---------------------------------------------------------------------------
try:
    import openai._base_client as _oai_base
    for _wrapper_name in ("SyncHttpxClientWrapper", "AsyncHttpxClientWrapper"):
        _wrapper = getattr(_oai_base, _wrapper_name, None)
        if _wrapper is not None:
            _wrapper.__del__ = lambda self: None
except Exception:
    pass


# ===========================================================================
# YAML helpers
# ===========================================================================

def _find_yaml() -> str:
    """
    Locate cascade_models.yaml by walking up from the current working
    directory (i.e. wherever the user runs the script from — typically
    the project root).  The library's own directory is never searched.

    Search order from cwd upward:
        <cwd>/config/cascade_models.yaml
        <cwd>/src/config/cascade_models.yaml
        <cwd>/../config/cascade_models.yaml
        <cwd>/../src/config/cascade_models.yaml
    """
    cwd = Path.cwd()
    candidates = [
        cwd / "config" / "cascade_models.yaml",
        cwd / "src" / "config" / "cascade_models.yaml",
        cwd.parent / "config" / "cascade_models.yaml",
        cwd.parent / "src" / "config" / "cascade_models.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "cascade_models.yaml not found. Searched:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\n\nEither run your script from the project root, or pass the path explicitly:\n"
        + "  CascadeRunner('simple', models_yaml_path='path/to/cascade_models.yaml')"
    )


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ===========================================================================
# CascadeRunner
# ===========================================================================

class CascadeRunner:
    """
    Unified entry-point for Simple and Complex cascade evaluation runs.

    Parameters
    ----------
    run_type : str | RunType
        ``"simple"`` or ``"complex"``.
    config : CascadeRunnerConfig, optional
        Full configuration object. If omitted, all defaults apply.
    **kwargs
        Convenience shortcuts applied over the defaults:

        ====================== ==============================================
        Kwarg                  Effect
        ====================== ==============================================
        openrouter_api_key     Override API key (env var fallback still works)
        acceptable_score       Judge threshold, float 0–1
        judge_model_key        Key in cascade_models.yaml judge_models section
        max_cascade_levels     Complex only — number of levels to attempt
        model_config_key       YAML top-level key for the chosen run type
        system_prompt          Worker system instruction
        mlflow_enabled         False to disable all MLflow logging
        experiment_name        Base name for the MLflow experiment
        models_yaml_path       Explicit path to cascade_models.yaml
        ====================== ==============================================

    Examples
    --------
    Minimal simple run::

        runner = CascadeRunner("simple")
        result = runner.run("What is 2 + 2?", question_id="math_1")
        print(result.answer)

    Complex run, custom threshold, no MLflow::

        runner = CascadeRunner(
            "complex",
            acceptable_score=0.95,
            max_cascade_levels=3,
            mlflow_enabled=False,
        )
        result = runner.run("Explain quantum entanglement.", question_id="phys_1")

    Fluent API::

        result = (
            CascadeRunner("simple")
            .set_api_key("sk-or-...")
            .set_acceptable_score(0.90)
            .set_system_prompt("Answer in exactly one sentence.")
            .set_judge_model("judge_model_2")
            .run("Capital of Japan?", question_id="geo_1")
        )

    Batch run::

        results = CascadeRunner("simple").run_batch(records, max_questions=50)
    """

    def __init__(
        self,
        run_type: Union[str, RunType] = RunType.SIMPLE,
        config: Optional[CascadeRunnerConfig] = None,
        **kwargs,
    ) -> None:
        self.run_type = RunType(run_type)
        self.cfg = config or CascadeRunnerConfig(run_type=self.run_type)
        self._apply_kwargs(kwargs)

        # Install the OpenRouter cost hook (idempotent)
        COST_TRACKER.install()

        # Resolve YAML path (lazy-loaded on first use)
        self._yaml_path: str = self.cfg.models_yaml_path or _find_yaml()
        self._raw_yaml: Optional[Dict[str, Any]] = None

        # MLflow is initialised lazily on the first run() call, not here.
        # This avoids crashing at construction time when the tracking
        # server is remote or temporarily unreachable.
        self._experiment_id: Optional[str] = None
        self._mlflow_ready: bool = False

    # -----------------------------------------------------------------------
    # Fluent configuration API
    # -----------------------------------------------------------------------

    def set_api_key(self, key: str) -> "CascadeRunner":
        """Set the OpenRouter API key at runtime."""
        self.cfg.openrouter_api_key = key
        return self

    def set_acceptable_score(self, score: float) -> "CascadeRunner":
        """Set the judge acceptance threshold (0–1)."""
        self.cfg.judge.acceptable_score = score
        return self

    def set_judge_model(self, key: str) -> "CascadeRunner":
        """Set which judge model key to use from cascade_models.yaml."""
        self.cfg.judge.judge_model_key = key
        return self

    def set_judge_agent(self, agent: ValidatorAgent) -> "CascadeRunner":
        """Inject a pre-built ValidatorAgent, bypassing YAML lookup entirely."""
        self.cfg.judge.agent = agent
        return self

    def set_system_prompt(self, prompt: str) -> "CascadeRunner":
        """Set the worker system prompt for the active run type."""
        if self.run_type == RunType.SIMPLE:
            self.cfg.simple.system_prompt = prompt
        else:
            self.cfg.complex.system_prompt = prompt
        return self

    def set_refinement_template(self, template: str) -> "CascadeRunner":
        """
        Override the refinement prompt template (Simple flow only).

        Available placeholders: ``{question}``, ``{attempt}``,
        ``{prev_answer}``, ``{issue}``.
        """
        self.cfg.simple.refinement_prompt_template = template
        return self

    def set_judge_prompt(self, prompt: str) -> "CascadeRunner":
        """Override the judge prompt prefix sent with worker answers (Complex only)."""
        self.cfg.complex.judge_prompt = prompt
        return self

    def set_next_level_prefix(self, prefix: str) -> "CascadeRunner":
        """Override the escalation prompt prefix used between levels (Complex only)."""
        self.cfg.complex.next_level_prompt_prefix = prefix
        return self

    def set_max_levels(self, n: int) -> "CascadeRunner":
        """Set the maximum number of cascade levels (Complex only)."""
        self.cfg.complex.max_cascade_levels = n
        return self

    def set_temperature_strategy(
        self,
        strategy: Union[str, TemperatureStrategy],
        fixed_value: Optional[float] = None,
        hard_prompt_range: Optional[Tuple[float, float]] = None,
        creative_range: Optional[Tuple[float, float]] = None,
    ) -> "CascadeRunner":
        """
        Configure the temperature strategy for Complex runs.

        Parameters
        ----------
        strategy            ``"fixed"``, ``"random"``, or ``"category_aware"``
        fixed_value         Temperature to use when strategy is ``"fixed"``
        hard_prompt_range   ``(low, high)`` for ``hard_prompt`` category
        creative_range      ``(low, high)`` for ``creative_writing`` category
        """
        self.cfg.complex.temperature_strategy = TemperatureStrategy(strategy)
        if fixed_value is not None:
            self.cfg.complex.fixed_temperature = fixed_value
        if hard_prompt_range is not None:
            self.cfg.complex.hard_prompt_temp_range = hard_prompt_range
        if creative_range is not None:
            self.cfg.complex.creative_temp_range = creative_range
        return self

    def override_model(self, slot: str, model_config: Dict[str, Any]) -> "CascadeRunner":
        """
        Override or add a model slot for Simple flow.

        Parameters
        ----------
        slot         Model slot name, e.g. ``"model_1"``.
        model_config Full model config dict merged over the YAML entry.
        """
        self.cfg.simple.model_overrides[slot] = model_config
        return self

    def override_level_model(
        self, level: int, slot: str, model_config: Dict[str, Any]
    ) -> "CascadeRunner":
        """
        Override or add a model slot at a specific Complex cascade level.

        Parameters
        ----------
        level        Cascade level number (1-indexed).
        slot         Worker slot name, e.g. ``"worker_model_1"``.
        model_config Full model config dict merged over the YAML entry.
        """
        self.cfg.complex.level_model_overrides.setdefault(level, {})[slot] = model_config
        return self

    def exhaust_all_models(self, value: bool = True) -> "CascadeRunner":
        """
        When True, Simple flow tries every model even after finding a passing
        answer — useful for benchmarking. The highest-scoring answer wins.
        """
        self.cfg.simple.exhaust_all_models = value
        return self

    def disable_mlflow(self) -> "CascadeRunner":
        """Disable all MLflow logging."""
        self.cfg.mlflow.enabled = False
        return self

    def set_experiment_name(self, name: str) -> "CascadeRunner":
        """Override the MLflow experiment base name."""
        self.cfg.mlflow.experiment_base_name = name
        return self

    # -----------------------------------------------------------------------
    # Public run API
    # -----------------------------------------------------------------------

    def run(
        self,
        question: str,
        question_id: str = "q0",
        category: str = "unknown",
        mlflow_run: bool = True,
    ) -> RunResult:
        """
        Execute the configured cascade for a single question.

        Parameters
        ----------
        question      The question / prompt text.
        question_id   Unique identifier used as the MLflow run name.
        category      Question category used by complex flow temperature
                      strategy (``"hard_prompt"``, ``"creative_writing"``…).
        mlflow_run    Wrap execution in ``mlflow.start_run``. Set False when
                      you are managing the run context externally.

        Returns
        -------
        RunResult
        """
        fn = self._run_simple if self.run_type == RunType.SIMPLE else self._run_complex

        if self.cfg.mlflow.enabled and mlflow_run:
            self._setup_mlflow()   # lazy — connects only when actually needed

        if self.cfg.mlflow.enabled and mlflow_run:
            with mlflow.start_run(run_name=question_id):
                self._log_common_params(question_id, category)
                result = fn(question, question_id, category)
                self._log_common_metrics(result)
            return result

        return fn(question, question_id, category)

    def run_batch(
        self,
        records: List[Dict[str, Any]],
        question_key: str = "question",
        id_key: str = "question_id",
        category_key: str = "category",
        max_questions: int = 0,
    ) -> List[RunResult]:
        """
        Run the cascade over a list of question records.

        Parameters
        ----------
        records        List of dicts; must contain the keys referenced by
                       ``question_key`` and ``id_key``.
        question_key   Dict key for the question text. Default ``"question"``.
        id_key         Dict key for the question ID. Default ``"question_id"``.
        category_key   Dict key for the category. Default ``"category"``.
        max_questions  When > 0, only the first N records are processed.
                       0 means process all.

        Returns
        -------
        List[RunResult]
        """
        subset = records[:max_questions] if max_questions > 0 else records
        results: List[RunResult] = []

        for idx, rec in enumerate(subset):
            q   = rec[question_key]
            qid = rec.get(id_key, f"q{idx}")
            cat = rec.get(category_key, "unknown")
            print(f"[{idx + 1}/{len(subset)}] {qid} ({cat})")
            try:
                results.append(self.run(q, qid, cat))
            except Exception as exc:
                print(f"  ERROR on {qid}: {exc}")

        self._print_batch_summary(results)
        return results

    # -----------------------------------------------------------------------
    # Simple flow
    # -----------------------------------------------------------------------

    def _run_simple(self, question: str, question_id: str, category: str) -> RunResult:
        models = self._load_simple_models()
        judge  = self._build_judge()

        slots    = list(models.keys())
        prompt   = question
        history: List[IterationRecord] = []
        candidates: List[Dict[str, Any]] = []

        for i, slot in enumerate(slots):
            cfg = models[slot]
            print(f"   [{i + 1}/{len(slots)}] {slot} ({cfg['model_name']})")

            agent = self._build_worker(cfg, slot, self.cfg.simple.system_prompt)

            COST_TRACKER.reset()
            agent_result = agent.run_sync(prompt)
            answer       = agent_result.content
            model_cost   = COST_TRACKER.pop_cost()

            COST_TRACKER.reset()
            judge_result = self._sync(judge.evaluate_single(question=question, answer=answer))
            judge_cost   = COST_TRACKER.pop_cost()

            score   = float(judge_result.get("score",   0.0))       if judge_result else 0.0
            verdict = str(judge_result.get("verdict",   "Unknown"))  if judge_result else "Unknown"
            feedback= str(judge_result.get("feedback",  ""))         if judge_result else ""

            passed = score >= self.cfg.judge.acceptable_score and verdict == "Valid"
            reason = f"Score: {score:.4f}, Verdict: {verdict}"
            if feedback:
                reason += f" — {feedback[:120]}"

            rec = IterationRecord(
                iteration          = i + 1,
                model_key          = slot,
                model_name         = cfg["model_name"],
                answer             = answer,
                passed             = passed,
                score              = score,
                verdict            = verdict,
                reason             = reason,
                model_cost         = model_cost,
                judge_cost         = judge_cost,
                iteration_total_cost = model_cost["total_cost"] + judge_cost["total_cost"],
            )
            history.append(rec)
            candidates.append(
                {"answer": answer, "model": slot, "model_name": cfg["model_name"],
                 "score": score, "verdict": verdict, "iteration": i + 1}
            )

            if self.cfg.mlflow.enabled:
                mlflow.log_metrics({
                    f"iter_{i + 1}_score":      score,
                    f"iter_{i + 1}_passed":     int(passed),
                    f"iter_{i + 1}_model_cost": model_cost["total_cost"],
                    f"iter_{i + 1}_judge_cost": judge_cost["total_cost"],
                })

            print(f"   → {reason}")

            if passed and not self.cfg.simple.exhaust_all_models:
                break
            if not passed:
                prompt = self.cfg.simple.refinement_prompt_template.format(
                    question   = question,
                    attempt    = i + 1,
                    prev_answer= answer,
                    issue      = reason,
                )

        total_cost = sum(r.iteration_total_cost for r in history)
        valid = [c for c in candidates if c["verdict"] == "Valid"]
        best  = (
            max(valid, key=lambda x: x["score"])
            if valid
            else (max(candidates, key=lambda x: x["score"])
                  if candidates
                  else {"answer": "", "model": "", "model_name": "",
                        "score": 0.0, "iteration": 0})
        )

        return RunResult(
            run_type         = RunType.SIMPLE,
            question_id      = question_id,
            question         = question,
            answer           = best["answer"],
            success          = bool(valid) and best["score"] >= self.cfg.judge.acceptable_score,
            total_cost       = total_cost,
            iterations       = best["iteration"],
            winning_model    = best["model"],
            winning_model_name = best["model_name"],
            iteration_history= history,
        )

    # -----------------------------------------------------------------------
    # Complex flow
    # -----------------------------------------------------------------------

    def _run_complex(self, question: str, question_id: str, category: str) -> RunResult:
        judge        = self._build_judge()
        level_history: List[CascadeLevelRecord] = []
        prompt_data: Tuple[str, str] = (question_id, question)
        grand_total  = 0.0
        final: Optional[RunResult] = None
        last_level_models: Dict[str, Any] = {}

        for level in range(1, self.cfg.complex.max_cascade_levels + 1):
            print(f"\n  ═══ Cascade Level {level} ═══")

            level_models = self._load_complex_level_models(level)
            if not level_models:
                print(f"  No models configured for level {level}. Skipping.")
                continue

            last_level_models = level_models
            self._apply_temperature(level_models, category)
            workers    = self._build_workers_dict(level_models, level)
            level_cost = 0.0

            # 1 — Initial answers (parallel)
            init_answers, c = self._sync(
                self._parallel_initial(workers, prompt_data, level)
            )
            level_cost += c

            # 2 — Peer debate / critique (parallel)
            critiques, c = self._sync(
                self._parallel_debate(workers, init_answers, question, level)
            )
            level_cost += c

            # 3 — Refinement (parallel)
            final_answers, c = self._sync(
                self._parallel_refinement(workers, question, init_answers, critiques, level)
            )
            level_cost += c

            if not final_answers:
                print(f"  Level {level}: no final answers produced. Skipping.")
                continue

            # 4 — Judge evaluation
            validator_prompt = self._ensemble(
                final_answers, prompt_data[1], self.cfg.complex.judge_prompt
            )
            judge_eval = (
                self._sync(self._validate(judge, prompt_data[1], validator_prompt, final_answers))
                or self._fallback_eval(prompt_data[1], final_answers)
            )

            grand_total += level_cost
            if self.cfg.mlflow.enabled:
                mlflow.log_metric(f"cascade_lvl_{level}_cost", level_cost)
                mlflow.log_metric("total_cost", grand_total)

            best_id = (
                judge_eval.get("evaluation", {})
                          .get("best_answer", {})
                          .get("best_worker_model_id", "")
                          .replace("_answer", "")
            )
            if best_id not in final_answers:
                best_id = next(
                    (k for k in final_answers if not isinstance(final_answers[k], str)), ""
                )

            best_score = float(
                judge_eval.get("evaluation", {})
                          .get("best_answer", {})
                          .get("best_confidence_score", 0.0)
            )
            passed = best_score >= self.cfg.judge.acceptable_score

            level_history.append(CascadeLevelRecord(
                level         = level,
                initial_answers = {
                    k: (v.content if hasattr(v, "content") else str(v))
                    for k, v in init_answers.items()
                },
                critiques     = {
                    k: (v.content if hasattr(v, "content") else str(v))
                    for k, v in critiques.items()
                },
                final_answers = {
                    k: (v.content if hasattr(v, "content") else str(v))
                    for k, v in final_answers.items()
                },
                judge_evaluation = judge_eval,
                best_model_id    = best_id,
                best_score       = best_score,
                passed           = passed,
                level_cost       = level_cost,
            ))

            best_answer = (
                final_answers[best_id].content
                if best_id and hasattr(final_answers.get(best_id), "content")
                else ""
            )

            if passed:
                print(f"  ✓ Level {level} passed (score={best_score:.4f})")
                final = RunResult(
                    run_type              = RunType.COMPLEX,
                    question_id           = question_id,
                    question              = question,
                    answer                = best_answer,
                    success               = True,
                    total_cost            = grand_total,
                    winning_cascade_level = level,
                    best_confidence_score = best_score,
                    winning_model         = best_id,
                    winning_model_name    = level_models.get(best_id, {}).get("model_name", ""),
                    cascade_level_history = level_history,
                )
                break

            print(f"  ✗ Level {level} did not pass (score={best_score:.4f}). Escalating…")
            next_prompt = self._ensemble(
                final_answers,
                question,
                self.cfg.complex.next_level_prompt_prefix,
                agents_answers_review=judge_eval,
            )
            prompt_data = (question_id, next_prompt)

        # Exhausted all levels without passing — return best effort
        if final is None:
            best_lvl = max(level_history, key=lambda r: r.best_score) if level_history else None
            answer = winning_model = winning_model_name = ""
            best_score = 0.0
            winning_level = 0
            if best_lvl:
                winning_level  = best_lvl.level
                best_score     = best_lvl.best_score
                winning_model  = best_lvl.best_model_id
                answer         = best_lvl.final_answers.get(winning_model) or next(
                    iter(best_lvl.final_answers.values()), ""
                )
                winning_model_name = last_level_models.get(winning_model, {}).get("model_name", "")
            final = RunResult(
                run_type              = RunType.COMPLEX,
                question_id           = question_id,
                question              = question,
                answer                = answer,
                success               = False,
                total_cost            = grand_total,
                winning_cascade_level = winning_level,
                best_confidence_score = best_score,
                winning_model         = winning_model,
                winning_model_name    = winning_model_name,
                cascade_level_history = level_history,
            )

        return final

    # -----------------------------------------------------------------------
    # Complex sub-tasks
    # -----------------------------------------------------------------------

    async def _parallel_initial(
        self,
        workers: Dict[str, WorkingAgent],
        prompt_data: Tuple[str, str],
        level: int,
    ) -> Tuple[Dict[str, AgentResponse], float]:
        _, text = prompt_data
        tasks = [
            self._call_worker(
                aid, agent, agent.generate,
                context=Prompt(content=text, model_tier="complex"),
            )
            for aid, agent in workers.items()
        ]
        return await self._gather_workers(tasks)

    async def _parallel_debate(
        self,
        workers: Dict[str, WorkingAgent],
        prev_answers: Dict[str, AgentResponse],
        original_question: str,
        level: int,
    ) -> Tuple[Dict[str, AgentResponse], float]:
        tasks = []
        for aid, agent in workers.items():
            if aid not in prev_answers or isinstance(prev_answers[aid], str):
                continue
            peers = [
                Prompt(content=v.content, model_tier="complex")
                for pid, v in prev_answers.items()
                if pid != aid and hasattr(v, "content")
            ]
            if not peers:
                continue
            tasks.append(
                self._call_worker(
                    aid, agent, agent.generate_critique,
                    question           = original_question,
                    peer_responses     = peers,
                    own_previous_answer= prev_answers[aid].content,
                )
            )
        if not tasks:
            return {}, 0.0
        return await self._gather_workers(tasks)

    async def _parallel_refinement(
        self,
        workers: Dict[str, WorkingAgent],
        original_question: str,
        init_answers: Dict[str, AgentResponse],
        critiques: Dict[str, AgentResponse],
        level: int,
    ) -> Tuple[Dict[str, AgentResponse], float]:
        if not critiques:
            return init_answers, 0.0
        tasks = []
        for wid, agent in workers.items():
            if wid not in init_answers or init_answers[wid] == "N/A":
                continue
            peer_critiques = {pid: c for pid, c in critiques.items() if pid != wid}
            ref_prompt = self._build_refinement_prompt(
                original_question, wid, init_answers[wid], peer_critiques
            )
            tasks.append(
                self._call_worker(
                    wid, agent, agent.generate,
                    context=Prompt(content=ref_prompt, model_tier="complex"),
                )
            )
        if not tasks:
            return init_answers, 0.0
        return await self._gather_workers(tasks)

    async def _call_worker(
        self, agent_id: str, agent: WorkingAgent, func, **kwargs
    ) -> Tuple[str, Optional[AgentResponse], Dict[str, float]]:
        COST_TRACKER.reset()
        try:
            resp: AgentResponse = await func(**kwargs)
        except Exception as exc:
            print(f"  Worker {agent_id} error: {exc}")
            return agent_id, None, {"total_cost": 0.0}
        cost = COST_TRACKER.pop_cost()
        return agent_id, resp, cost

    @staticmethod
    async def _gather_workers(tasks) -> Tuple[Dict[str, AgentResponse], float]:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        answers: Dict[str, AgentResponse] = {}
        total = 0.0
        for r in results:
            if isinstance(r, Exception):
                continue
            aid, resp, cost = r
            total += cost.get("total_cost", 0.0)
            if resp:
                answers[aid] = resp
        return answers, total

    async def _validate(
        self,
        judge: ValidatorAgent,
        question: str,
        prompt: str,
        answers: Dict[str, AgentResponse],
    ) -> Optional[Dict[str, Any]]:
        return await judge.evaluate_multiple(
            prompt=prompt, question=question, answers=answers
        )

    # -----------------------------------------------------------------------
    # Model / agent builders
    # -----------------------------------------------------------------------

    def _yaml(self) -> Dict[str, Any]:
        if self._raw_yaml is None:
            self._raw_yaml = _load_yaml(self._yaml_path)
        return self._raw_yaml

    def _load_simple_models(self) -> Dict[str, Any]:
        key = self.cfg.simple.model_config_key
        raw = self._yaml().get(key)
        if raw is None:
            raise ValueError(
                f"Simple model config key '{key}' not found in {self._yaml_path}.\n"
                f"Available keys: {list(self._yaml().keys())}"
            )
        models = copy.deepcopy(raw)
        for slot, override in self.cfg.simple.model_overrides.items():
            models.setdefault(slot, {}).update(override)
        return models

    def _load_complex_level_models(self, level: int) -> Optional[Dict[str, Any]]:
        key = self.cfg.complex.model_config_key
        raw = self._yaml().get(key, {}).get(f"cascade_lvl_{level}")
        if raw is None:
            return None
        models = copy.deepcopy(raw)
        for slot, override in self.cfg.complex.level_model_overrides.get(level, {}).items():
            models.setdefault(slot, {}).update(override)
        return models

    def _load_judge_cfg(self) -> Dict[str, Any]:
        key = self.cfg.judge.judge_model_key
        cfg = self._yaml().get("judge_models", {}).get(key)
        if cfg is None:
            available = list(self._yaml().get("judge_models", {}).keys())
            raise ValueError(
                f"Judge key '{key}' not found in {self._yaml_path}.\n"
                f"Available judge keys: {available}"
            )
        return cfg

    def _api_key(self) -> str:
        return self.cfg.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")

    def _build_worker(
        self, cfg: Dict[str, Any], slot: str, system_prompt: str
    ) -> WorkingAgent:
        return WorkingAgent(
            model_id          = cfg["model_name"],
            role_name         = slot,
            system_instruction= system_prompt,
            config            = cfg,
            cascade_tier      = cfg.get("tier", "primary"),
            api_key           = self._api_key(),
        )

    def _build_workers_dict(
        self, level_models: Dict[str, Any], level: int
    ) -> Dict[str, WorkingAgent]:
        return {
            slot: self._build_worker(cfg, slot, self.cfg.complex.system_prompt)
            for slot, cfg in level_models.items()
        }

    def _build_judge(self) -> ValidatorAgent:
        if self.cfg.judge.agent is not None:
            return self.cfg.judge.agent
        judge_cfg = self._load_judge_cfg()
        return ValidatorAgent(
            model_name= judge_cfg["model_name"],
            api_key   = self._api_key(),
            threshold = self.cfg.judge.acceptable_score,
        )

    # -----------------------------------------------------------------------
    # Temperature helpers
    # -----------------------------------------------------------------------

    def _apply_temperature(self, level_models: Dict[str, Any], category: str) -> None:
        strategy = self.cfg.complex.temperature_strategy
        for model_cfg in level_models.values():
            params = model_cfg.setdefault("parameters", {})
            if strategy == TemperatureStrategy.FIXED:
                params["temperature"] = self.cfg.complex.fixed_temperature
            elif strategy == TemperatureStrategy.RANDOM:
                lo, hi = self.cfg.complex.hard_prompt_temp_range
                params["temperature"] = round(random.uniform(lo, hi), 2)
            elif strategy == TemperatureStrategy.CATEGORY_AWARE:
                if category == "hard_prompt":
                    lo, hi = self.cfg.complex.hard_prompt_temp_range
                elif category == "creative_writing":
                    lo, hi = self.cfg.complex.creative_temp_range
                else:
                    continue  # leave whatever the YAML default is
                params["temperature"] = round(random.uniform(lo, hi), 2)

    # -----------------------------------------------------------------------
    # Prompt helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean(s: str) -> str:
        return s.translate(str.maketrans("\n\t\r", "   "))

    def _build_refinement_prompt(
        self,
        question: str,
        agent_id: str,
        init_answer: AgentResponse,
        critiques: Dict[str, AgentResponse],
    ) -> str:
        blocks = "\n\n".join(
            f"<peer_review_by_{pid}>\n{c.content}\n</peer_review_by_{pid}>"
            for pid, c in critiques.items()
            if hasattr(c, "content")
        )
        return (
            f"INITIAL QUESTION:\n{self._clean(question)}\n\n"
            f"YOUR INITIAL ANSWER:\n{self._clean(init_answer.content)}\n\n"
            f"PEER REVIEWS:\n{blocks}\n\n"
            "Review the feedback carefully, identify valid points, and produce a refined answer."
        )

    def _ensemble(
        self,
        answers: Dict[str, AgentResponse],
        question: str,
        premise: str,
        agents_answers_review: Optional[Dict[str, Any]] = None,
    ) -> str:
        blocks = "\n\n".join(
            f"<{aid}_answer>\n"
            f"{(v.content if hasattr(v, 'content') else str(v))}"
            f"\n</{aid}_answer>"
            for aid, v in answers.items()
        )
        out = f"{premise}\n\nINITIAL QUESTION:\n{self._clean(question)}\n\nANSWERS:\n{blocks}"

        if agents_answers_review:
            reviews = (
                agents_answers_review.get("evaluation", {}).get("individual_reviews", {})
            )
            if reviews:
                review_blocks = "\n\n".join(
                    f"<{wid}_review>\n{r.get('reason', 'N/A')}\n</{wid}_review>"
                    for wid, r in reviews.items()
                )
                out += f"\n\nVALIDATOR REVIEWS:\n{review_blocks}"
        return out

    @staticmethod
    def _fallback_eval(
        question: str, answers: Dict[str, AgentResponse]
    ) -> Dict[str, Any]:
        first = next((k for k in answers if not isinstance(answers[k], str)), None)
        return {
            "evaluation": {
                "question": question,
                "best_answer": {
                    "best_worker_model_id":    first or "",
                    "best_confidence_score":   0.0,
                    "best_reason":             "Judge unavailable — fallback to first valid answer",
                },
                "individual_reviews": {},
            }
        }

    # -----------------------------------------------------------------------
    # MLflow helpers
    # -----------------------------------------------------------------------

    def _setup_mlflow(self) -> None:
        if self._mlflow_ready:
            return

        # Prefer explicit config value; fall back to env var so that
        # MLFLOW_TRACKING_URI=http://34.118.9.137:5000 always works.
        uri = (
            self.cfg.mlflow.tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/")
        )
        mlflow.set_tracking_uri(uri)

        if self.cfg.mlflow.autolog_pydantic:
            try:
                mlflow.pydantic_ai.autolog()
            except Exception:
                pass

        try:
            if self.cfg.mlflow.auto_version:
                self._experiment_id = self._create_versioned_experiment()
            else:
                name = self.cfg.mlflow.experiment_base_name
                exp  = mlflow.get_experiment_by_name(name)
                if exp:
                    self._experiment_id = exp.experiment_id
                    mlflow.set_experiment(name)
                else:
                    self._experiment_id = mlflow.create_experiment(name)
                    mlflow.set_experiment(name)
            self._mlflow_ready = True
        except Exception as exc:
            print(f"  [MLflow] Could not connect to {uri}: {exc}")
            print("  [MLflow] Disabling MLflow logging for this run.")
            self.cfg.mlflow.enabled = False

    def _create_versioned_experiment(self) -> str:
        base = self.cfg.mlflow.experiment_base_name
        exps = mlflow.search_experiments(filter_string=f"name LIKE '{base}_v%'")
        versions = []
        for e in exps:
            try:
                versions.append(int(e.name.split("_v")[-1]))
            except ValueError:
                pass
        next_v = max(versions, default=0) + 1
        name   = f"{base}_v{next_v}"
        eid    = mlflow.create_experiment(name)
        mlflow.set_experiment(name)
        print(f"MLflow experiment: {name} (id={eid})")
        return eid

    def _log_common_params(self, question_id: str, category: str) -> None:
        mlflow.log_params({
            "question_id":      question_id,
            "category":         category,
            "run_type":         self.run_type.value,
            "judge_model_key":  self.cfg.judge.judge_model_key,
            "acceptable_score": self.cfg.judge.acceptable_score,
        })

    def _log_common_metrics(self, result: RunResult) -> None:
        mlflow.log_metrics({
            "success":       int(result.success),
            "total_cost":    result.total_cost,
            "answer_length": len(result.answer),
        })
        if result.run_type == RunType.SIMPLE:
            mlflow.log_metric("iterations", result.iterations)
        else:
            mlflow.log_metric("winning_level",         result.winning_cascade_level)
            mlflow.log_metric("best_confidence_score", result.best_confidence_score)

    # -----------------------------------------------------------------------
    # Async / sync bridge
    # -----------------------------------------------------------------------

    @staticmethod
    def _sync(coro):
        """Run a coroutine from sync context. Compatible with nested event loops."""
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def _apply_kwargs(self, kwargs: Dict[str, Any]) -> None:
        mapping = {
            "openrouter_api_key": ("cfg.openrouter_api_key",),
            "acceptable_score":   ("cfg.judge.acceptable_score",),
            "judge_model_key":    ("cfg.judge.judge_model_key",),
            "max_cascade_levels": ("cfg.complex.max_cascade_levels",),
            "mlflow_enabled":     ("cfg.mlflow.enabled",),
            "experiment_name":    ("cfg.mlflow.experiment_base_name",),
            "models_yaml_path":   ("cfg.models_yaml_path",),
            "system_prompt": None,   # handled separately
            "model_config_key": None,
        }
        for key, val in kwargs.items():
            if key == "system_prompt":
                if self.run_type == RunType.SIMPLE:
                    self.cfg.simple.system_prompt = val
                else:
                    self.cfg.complex.system_prompt = val
            elif key == "model_config_key":
                if self.run_type == RunType.SIMPLE:
                    self.cfg.simple.model_config_key = val
                else:
                    self.cfg.complex.model_config_key = val
            elif key == "openrouter_api_key":
                self.cfg.openrouter_api_key = val
            elif key == "acceptable_score":
                self.cfg.judge.acceptable_score = val
            elif key == "judge_model_key":
                self.cfg.judge.judge_model_key = val
            elif key == "max_cascade_levels":
                self.cfg.complex.max_cascade_levels = val
            elif key == "mlflow_enabled":
                self.cfg.mlflow.enabled = val
            elif key == "experiment_name":
                self.cfg.mlflow.experiment_base_name = val
            elif key == "models_yaml_path":
                self.cfg.models_yaml_path = val
            else:
                raise TypeError(f"CascadeRunner.__init__() got an unexpected keyword argument '{key}'")

    @staticmethod
    def _print_batch_summary(results: List[RunResult]) -> None:
        n = len(results)
        if n == 0:
            return
        succeeded  = sum(1 for r in results if r.success)
        total_cost = sum(r.total_cost for r in results)
        print(f"\n{'=' * 60}")
        print(f"Batch summary : {succeeded}/{n} succeeded ({100 * succeeded / n:.1f}%)")
        print(f"Total cost    : ${total_cost:.4f}   Avg: ${total_cost / n:.4f}")
        if results[0].run_type == RunType.SIMPLE:
            avg_iter = sum(r.iterations for r in results) / n
            print(f"Avg iterations: {avg_iter:.2f}")
        else:
            avg_lvl   = sum(r.winning_cascade_level for r in results) / n
            avg_score = sum(r.best_confidence_score for r in results) / n
            print(f"Avg level     : {avg_lvl:.2f}   Avg score: {avg_score:.4f}")
        print(f"{'=' * 60}\n")

    def __repr__(self) -> str:
        return (
            f"CascadeRunner("
            f"run_type={self.run_type.value!r}, "
            f"acceptable_score={self.cfg.judge.acceptable_score}, "
            f"judge={self.cfg.judge.judge_model_key!r}"
            f")"
        )