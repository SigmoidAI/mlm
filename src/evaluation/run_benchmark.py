import mlflow
import asyncio
import argparse
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import AssessmentSource, AssessmentSourceType
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List, Optional, Tuple
import os
import time

from .arena_judger import ArenaValidatorAgent
from ..config.make_config import make_config, _load_env

# ==============================================================================
# CONFIGURATION
# ==============================================================================
_load_env()
CASCADE_MODELS_CONFIG: dict[str, str] = make_config()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set.")

# Arena Hard weighted scoring (matches the leaderboard methodology)
ARENA_SCORE_MAP = {
    "A>>B": 1.0,
    "A>B":  0.5,
    "A=B":  0.0,
    "B>A": -0.5,
    "B>>A":-1.0,
}

# GPT-4.1 cost: $2/1M input tokens, $8/1M output tokens
GPT_IN_COST_PER_TOKEN  = 2.0 / 1_000_000
GPT_OUT_COST_PER_TOKEN = 8.0 / 1_000_000


# ==============================================================================
# HELPERS
# ==============================================================================
def get_experiment_runs(client: MlflowClient, experiment_name: str) -> Dict[str, Any]:
    """Fetch runs indexed by run_name (question_id)."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"]
    )
    return {run.data.tags.get("mlflow.runName"): run for run in runs}


def extract_traces(client: MlflowClient, run_id: str, experiment_id: str):
    """Pull the trace DataFrame for a given run."""
    return mlflow.search_traces(run_id=run_id, experiment_ids=[experiment_id])


def estimate_gpt_cost(question: str, answer: str) -> Dict[str, float]:
    """
    Estimate GPT cost based on text length.
    Approximation: 1 token ≈ 4 characters.
    Input (question) at $2/1M tokens, Output (answer) at $8/1M tokens.
    """
    input_tokens  = len(question) // 4
    output_tokens = len(answer)   // 4
    total_tokens  = input_tokens + output_tokens
    cost = input_tokens * GPT_IN_COST_PER_TOKEN + output_tokens * GPT_OUT_COST_PER_TOKEN

    return {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  total_tokens,
        "cost":          round(cost, 6),
    }


# Priority order for auto-detecting the answer key in experiment B traces.
# The first key found in the response dict will be used.
ANSWER_B_KEY_PRIORITY: List[str] = ["final_best_response", "answer", "output"]


def extract_answer_b(traces_b) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detect the answer key in experiment B traces by trying ANSWER_B_KEY_PRIORITY
    in order. Returns (answer_b, trace_id_b, matched_key) or (None, None, None).
    """
    for key in ANSWER_B_KEY_PRIORITY:
        matching = traces_b[
            traces_b["response"].apply(
                lambda r: isinstance(r, dict) and r.get(key) not in (None, "")
            )
        ]
        if not matching.empty:
            row = matching.iloc[0]
            return row["response"][key], row["trace_id"], key

    return None, None, None


# ==============================================================================
# BRADLEY-TERRY SCORING  (identical to lmarena/arena-hard-auto show_result.py)
# ==============================================================================
def compute_mle_elo(
    battles: pd.DataFrame,
    SCALE: float = 400.0,
    BASE: float = 10.0,
    INIT_RATING: float = 1000.0,
) -> pd.Series:
    """
    Fit a Bradley-Terry model via logistic regression on a battles DataFrame
    with columns: model_a, model_b, winner  (winner ∈ {"model_a","model_b","tie"}).

    Ties are split 0.5/0.5 (each tie becomes half-win for both sides).
    Returns a Series of Elo-scaled ratings indexed by model name.
    """
    models = pd.concat([battles["model_a"], battles["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # Build win/loss rows; ties count as 0.5 win for each side
    p = []
    y = []
    sample_weight = []

    for _, row in battles.iterrows():
        idx_a = models[row["model_a"]]
        idx_b = models[row["model_b"]]
        x = np.zeros(len(models))
        x[idx_a] = +1
        x[idx_b] = -1

        if row["winner"] == "model_a":
            p.append(x); y.append(1); sample_weight.append(1.0)
        elif row["winner"] == "model_b":
            p.append(x); y.append(0); sample_weight.append(1.0)
        else:  # tie → half-win each direction
            p.append(x); y.append(1); sample_weight.append(0.5)
            p.append(x); y.append(0); sample_weight.append(0.5)

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6, max_iter=10_000)
    lr.fit(np.array(p), np.array(y), sample_weight=np.array(sample_weight))

    elo = pd.Series(
        SCALE * np.log(BASE) / np.log(BASE) * lr.coef_[0] + INIT_RATING,
        index=models.index,
    )
    # Simpler, equivalent formulation used by lmarena:
    elo = pd.Series(
        INIT_RATING + SCALE * np.log10(np.exp(lr.coef_[0])),
        index=models.index,
    )
    return elo.sort_values(ascending=False)


def compute_bt_win_rate(
    battles: pd.DataFrame,
    model_b_name: str,
    baseline_name: str,
    n_bootstrap: int = 100,
    SCALE: float = 400.0,
    BASE: float = 10.0,
    INIT_RATING: float = 1000.0,
) -> Dict[str, float]:
    """
    Compute the Arena-Hard leaderboard score for model_b vs baseline.

    The score is the Bradley-Terry predicted win-rate of model_b against the
    baseline (expressed as a percentage, 50 = equal, >50 = model_b is better).
    Bootstrap is used to build 95% confidence intervals, exactly as lmarena does.

    Returns dict with keys: score, ci_lower, ci_upper, elo_b, elo_baseline.
    """
    rows = []
    for _, r in battles.iterrows():
        rows.append(r)

    def _bt_score(df: pd.DataFrame) -> float:
        try:
            elo = compute_mle_elo(df, SCALE=SCALE, BASE=BASE, INIT_RATING=INIT_RATING)
            if model_b_name not in elo.index or baseline_name not in elo.index:
                return float("nan")
            elo_b = elo[model_b_name]
            elo_base = elo[baseline_name]
            # Predicted win probability of model_b over baseline (BT formula)
            win_prob = 1.0 / (1.0 + BASE ** ((elo_base - elo_b) / SCALE))
            return win_prob * 100.0
        except Exception:
            return float("nan")

    # Point estimate on full data
    point_score = _bt_score(battles)

    # Bootstrap for CI
    bootstrap_scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = battles.sample(n=len(battles), replace=True, random_state=rng.integers(0, 2**31))
        s = _bt_score(sample)
        if not np.isnan(s):
            bootstrap_scores.append(s)

    bootstrap_scores = np.array(bootstrap_scores)
    ci_lower = float(np.quantile(bootstrap_scores, 0.05)) if len(bootstrap_scores) else float("nan")
    ci_upper = float(np.quantile(bootstrap_scores, 0.95)) if len(bootstrap_scores) else float("nan")

    # Full-data Elo for logging
    elo = compute_mle_elo(battles, SCALE=SCALE, BASE=BASE, INIT_RATING=INIT_RATING)

    return {
        "bt_score":    round(point_score, 2),
        "ci_lower":    round(ci_lower, 2),
        "ci_upper":    round(ci_upper, 2),
        "elo_model_b": round(float(elo.get(model_b_name, float("nan"))), 2),
        "elo_baseline":round(float(elo.get(baseline_name, float("nan"))), 2),
    }


def battles_from_results(df: pd.DataFrame, exp_1_name: str, exp_2_name: str) -> pd.DataFrame:
    """
    Convert per-question results DataFrame into the battles format expected by
    the Bradley-Terry model.  Each question produces TWO rows (fwd + rev),
    exactly as in the official Arena Hard pipeline.

    battles columns: model_a, model_b, winner
    winner ∈ {"model_a", "model_b", "tie"}
    """
    rows = []
    for _, r in df.iterrows():
        # forward game: model_a=exp1, model_b=exp2
        fwd_score = r["arena_score_fwd"]
        if fwd_score > 0:
            winner_fwd = "model_a"
        elif fwd_score < 0:
            winner_fwd = "model_b"
        else:
            winner_fwd = "tie"
        rows.append({"model_a": exp_1_name, "model_b": exp_2_name, "winner": winner_fwd})

        # reverse game: model_a=exp2, model_b=exp1  (scores are already flipped in arena_score_rev)
        rev_score = r["arena_score_rev"]   # negative means exp1 won in reverse
        if rev_score > 0:   # exp2 (now model_a) won the reverse game
            winner_rev = "model_a"
        elif rev_score < 0:
            winner_rev = "model_b"
        else:
            winner_rev = "tie"
        rows.append({"model_a": exp_2_name, "model_b": exp_1_name, "winner": winner_rev})

    return pd.DataFrame(rows)


# ==============================================================================
# TRACING WRAPPER
# ==============================================================================
@mlflow.trace(name="arena_judge_evaluation", span_type="LLM")
async def execute_traced_judge(
    validator, question, answer_a, answer_b, question_id,
    exp_1_name: str, exp_2_name: str
):
    """
    Wraps the validation step in an MLflow trace.
    """
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "question_id": question_id,
            "model_A": exp_1_name,
            "model_B": exp_2_name,
        })

    return await validator.validate(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        question_id=question_id
    )


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
async def run_benchmark(exp_1_name: str, exp_2_name: str):
    """
    mode: "simple" | "complex" | "generic"
    """
    comparison_exp_name = f"{exp_1_name}_vs_{exp_2_name}"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"📡 Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    print(f"⚔️  Comparing: {exp_1_name} vs {exp_2_name}\n")

    # ── 1. Fetch runs from both experiments ───────────────────────────────────
    runs_1 = get_experiment_runs(client, exp_1_name)
    runs_2 = get_experiment_runs(client, exp_2_name)

    common_names = sorted(set(runs_1.keys()) & set(runs_2.keys()))
    print(f"📂 {exp_1_name}: {len(runs_1)} runs")
    print(f"📂 {exp_2_name}: {len(runs_2)} runs")
    print(f"🔗 Overlapping question_ids: {len(common_names)}\n")

    if not common_names:
        print("⚠️  No overlapping runs found. Nothing to compare.")
        return

    # ── 2. Resolve experiment IDs ─────────────────────────────────────────────
    exp_1_id = client.get_experiment_by_name(exp_1_name).experiment_id
    exp_2_id = client.get_experiment_by_name(exp_2_name).experiment_id

    # ── 3. Create / set the comparison experiment ─────────────────────────────
    mlflow.set_experiment(comparison_exp_name)
    comparison_exp_id = client.get_experiment_by_name(comparison_exp_name).experiment_id

    # ── 4. Initialize judge ───────────────────────────────────────────────────
    validator = ArenaValidatorAgent(
        model_name="google/gemini-2.5-flash",
        api_key=OPENROUTER_API_KEY
    )

    # ── 5. Per-question judging loop ──────────────────────────────────────────
    results:  List[Dict[str, Any]] = []
    skipped:  List[str] = []

    cumulative_score_A = 0.0
    cumulative_score_B = 0.0
    per_question_run_ids: List[str] = []
    n_judged = 0

    for idx, run_name in enumerate(common_names, 1):
        print(f"\n[{idx}/{len(common_names)}] Judging: {run_name}")

        try:
            run_a = runs_1[run_name]
            run_b = runs_2[run_name]

            run_id_a = run_a.info.run_id
            run_id_b = run_b.info.run_id

            # Framework cost from OpenRouter hooks (logged on run_b)
            total_cost = run_b.data.metrics.get("total_cost", None)

            # ── Pull traces ───────────────────────────────────────────────────
            try:
                traces_a = extract_traces(client, run_id_a, exp_1_id)
                traces_b = extract_traces(client, run_id_b, exp_2_id)

                question_a = traces_a.iloc[0]["request"]["q"]
                answer_a   = traces_a.iloc[0]["response"]["output"]
                trace_id_a = traces_a.iloc[0]["trace_id"]
                gpt_cost_info = estimate_gpt_cost(question_a, answer_a)

                answer_b, trace_id_b, matched_key = extract_answer_b(traces_b)

                if answer_b is None:
                    print(f"   ⚠️  Skipping: None of {ANSWER_B_KEY_PRIORITY} found in exp B traces.")
                    skipped.append(run_name)
                    continue
                else:
                    print(f"   🔑 Using response key: '{matched_key}'")

            except Exception as e:
                print(f"   ❌ Trace extraction failed for {run_name}: {e}")
                skipped.append(run_name)
                continue

            question = question_a
            if not question:
                print(f"   ⚠️  Skipping: Could not extract question.")
                skipped.append(run_name)
                continue
            if not answer_a or not answer_b:
                print(f"   ⚠️  Skipping: Missing answer (A={bool(answer_a)}, B={bool(answer_b)})")
                skipped.append(run_name)
                continue

            # ── Per-question run ──────────────────────────────────────────────
            with mlflow.start_run(run_name=run_name) as comparison_run:
                comparison_run_id = comparison_run.info.run_id
                per_question_run_ids.append(comparison_run_id)

                # Judge — forward pass
                judge_start = time.time()

                try:
                    validation_fwd = await execute_traced_judge(
                        validator,
                        question=question,
                        answer_a=answer_a,
                        answer_b=answer_b,
                        question_id=f"{run_name}_fwd",
                        exp_1_name=exp_1_name,
                        exp_2_name=exp_2_name,
                    )
                except Exception as e:
                    print(f"   ❌ Forward judge call failed for {run_name}: {e}")
                    raise

                verdict_fwd   = validation_fwd.get("verdict", "A=B")
                reasoning_fwd = validation_fwd.get("reasoning", "")
                score_fwd     = ARENA_SCORE_MAP.get(verdict_fwd, 0.0)

                verdict   = verdict_fwd
                reasoning = reasoning_fwd

                # Judge — reverse pass
                try:
                    validation_rev = await execute_traced_judge(
                        validator,
                        question=question,
                        answer_a=answer_b,   # swapped
                        answer_b=answer_a,
                        question_id=f"{run_name}_rev",
                        exp_1_name=exp_1_name,
                        exp_2_name=exp_2_name,
                    )
                except Exception as e:
                    print(f"   ❌ Reverse judge call failed for {run_name}: {e}")
                    raise

                verdict_rev_raw = validation_rev.get("verdict", "A=B")
                reasoning_rev   = validation_rev.get("reasoning", "")
                score_rev_raw   = ARENA_SCORE_MAP.get(verdict_rev_raw, 0.0)
                score_rev       = -score_rev_raw

                arena_score     = (score_fwd + score_rev) / 2.0
                judge_latency_s = time.time() - judge_start
                judge_trace_id  = mlflow.get_last_active_trace_id()

                if arena_score > 0:
                    winner = exp_1_name
                elif arena_score < 0:
                    winner = exp_2_name
                else:
                    winner = "Tie"

                score_A = arena_score
                score_B = -arena_score

                cumulative_score_A += score_A
                cumulative_score_B += score_B
                n_judged += 1

                running_win_val_A = cumulative_score_A / n_judged
                running_win_val_B = cumulative_score_B / n_judged

                print(
                    f"   📊 Verdict: {verdict} → 🏆 {winner}  "
                    f"(arena={arena_score:+.1f} | win_val_A={running_win_val_A:+.3f} "
                    f"win_val_B={running_win_val_B:+.3f} | "
                    f"cost_est=${gpt_cost_info['cost']:.6f} | "
                    f"cost_fw=${f'{total_cost:.6f}' if total_cost is not None else 'N/A'})"
                )

                # Params
                mlflow.log_params({
                    "question_id":       run_name,
                    "model_A":           exp_1_name,
                    "model_B":           exp_2_name,
                    "verdict":           verdict,
                    "winner":            winner,
                    "source_run_id_A":   run_id_a,
                    "source_run_id_B":   run_id_b,
                    "source_trace_id_A": str(trace_id_a),
                    "source_trace_id_B": str(trace_id_b),
                    "judge_model":       "google/gemini-2.5-flash",
                    "judge_trace_id":    str(judge_trace_id) if judge_trace_id else "N/A",
                })

                # Metrics
                mlflow.log_metrics({
                    "arena_score_forward":      score_fwd,
                    "arena_score_reverse":      score_rev,
                    "arena_score_pairwise_avg": arena_score,
                    "arena_score":              arena_score,

                    "win_binary": 1 if winner == exp_1_name else (-1 if winner == exp_2_name else 0),

                    "cost_generic":      gpt_cost_info["cost"],
                    "gpt_input_tokens":  gpt_cost_info["input_tokens"],
                    "gpt_output_tokens": gpt_cost_info["output_tokens"],
                    "gpt_total_tokens":  gpt_cost_info["total_tokens"],

                    "cost_framework": total_cost if total_cost is not None else 0.0,

                    "answer_length_A":          len(answer_a),
                    "answer_length_B":          len(answer_b),
                    "reasoning_length_forward": len(reasoning_fwd),
                    "reasoning_length_reverse": len(reasoning_rev),

                    "judge_latency_s": round(judge_latency_s, 3),

                    "win_val_A": round(running_win_val_A, 4),
                    "win_val_B": round(running_win_val_B, 4),
                    "win_val":   round(running_win_val_A - running_win_val_B, 4),
                })

                # Artifacts
                mlflow.log_text(question,      "question.txt")
                mlflow.log_text(answer_a,      "response_A.txt")
                mlflow.log_text(answer_b,      "response_B.txt")
                mlflow.log_text(reasoning_fwd, "reasoning_forward.txt")
                mlflow.log_text(reasoning_rev, "reasoning_reverse.txt")

                # Feedback on source traces
                for trace_id, source_exp, label in [
                    (trace_id_a, exp_1_name, "A"),
                    (trace_id_b, exp_2_name, "B"),
                ]:
                    try:
                        mlflow.log_feedback(
                            trace_id=str(trace_id),
                            name="ArenaHardComparison",
                            value=(winner == source_exp) or (winner == "Tie"),
                            rationale=f"[{verdict}] {reasoning[:500]}",
                            source=AssessmentSource(
                                source_type=AssessmentSourceType.LLM_JUDGE,
                                source_id=f"arena_judge_{comparison_exp_name}"
                            ),
                            metadata={
                                "verdict":            verdict,
                                "winner":             winner,
                                "arena_score":        arena_score,
                                "opponent":           exp_2_name if label == "A" else exp_1_name,
                                "comparison_run_id":  comparison_run_id,
                                "question_id":        run_name,
                                "judge_trace_id":     str(judge_trace_id),
                                "gpt_cost_estimated": gpt_cost_info["cost"],
                                "gpt_cost_framework": total_cost if total_cost is not None else 0.0,
                            }
                        )
                    except Exception as e:
                        print(f"   ⚠️  Could not log feedback on trace {trace_id} ({label}): {e}")

            results.append({
                "question_id":        run_name,
                "verdict":            verdict,
                "winner":             winner,
                "arena_score":        arena_score,
                "arena_score_fwd":    score_fwd,
                "arena_score_rev":    score_rev,
                "answer_length_A":    len(answer_a),
                "answer_length_B":    len(answer_b),
                "reasoning_length":   len(reasoning),
                "judge_latency_s":    round(judge_latency_s, 3),
                "gpt_cost_estimated": gpt_cost_info["cost"],
                "gpt_cost_framework": total_cost if total_cost is not None else 0.0,
            })

        except Exception as e:
            print(f"   💥 Unexpected error on {run_name}, skipping: {type(e).__name__}: {e}")
            skipped.append(run_name)
            continue

    # ── 6. Summary run ────────────────────────────────────────────────────────
    if results:
        df = pd.DataFrame(results)
        total = len(df)

        exp1_wins = len(df[df["winner"] == exp_1_name])
        exp2_wins = len(df[df["winner"] == exp_2_name])
        ties      = len(df[df["winner"] == "Tie"])

        # ── Bradley-Terry score (same as lmarena leaderboard) ─────────────────
        # Build battles table: 2 rows per question (fwd + rev), then fit BT model
        print("\n⚙️  Computing Bradley-Terry scores (100 bootstrap rounds)...")
        battles = battles_from_results(df, exp_1_name, exp_2_name)

        bt_A = compute_bt_win_rate(
            battles, model_b_name=exp_1_name, baseline_name=exp_2_name, n_bootstrap=100
        )
        bt_B = compute_bt_win_rate(
            battles, model_b_name=exp_2_name, baseline_name=exp_1_name, n_bootstrap=100
        )

        # bt_score: predicted win-rate % vs the other model
        # 50% = equal, >50% = this model wins more often
        bt_score_A = bt_A["bt_score"]   # model A win-rate vs B
        bt_score_B = bt_B["bt_score"]   # model B win-rate vs A
        elo_A      = bt_A["elo_model_b"]
        elo_B      = bt_B["elo_model_b"]

        with mlflow.start_run(run_name=f"SUMMARY_{exp_1_name}_vs_{exp_2_name}"):
            mlflow.log_params({
                "model_A":           exp_1_name,
                "model_B":           exp_2_name,
                "judge_model":       "google/gemini-2.5-flash",
                "total_questions":   total,
                "total_skipped":     len(skipped),
                "exp_1_id":          exp_1_id,
                "exp_2_id":          exp_2_id,
                "comparison_exp_id": comparison_exp_id,
                "scoring_method":    "Bradley-Terry MLE + 100x bootstrap (lmarena methodology)",
            })

            mlflow.log_metrics({
                # ── Bradley-Terry scores (primary, matches lmarena leaderboard) ──
                "bt_score_A":    bt_score_A,         # model A win-rate % vs B
                "bt_score_B":    bt_score_B,         # model B win-rate % vs A
                "bt_ci_lower_A": bt_A["ci_lower"],   # 5th pct bootstrap
                "bt_ci_upper_A": bt_A["ci_upper"],   # 95th pct bootstrap
                "bt_ci_lower_B": bt_B["ci_lower"],
                "bt_ci_upper_B": bt_B["ci_upper"],
                "elo_A":         elo_A,
                "elo_B":         elo_B,

                # ── Raw counts ────────────────────────────────────────────────
                f"{exp_1_name}_wins":     exp1_wins,
                f"{exp_2_name}_wins":     exp2_wins,
                "ties":                   ties,
                f"{exp_1_name}_win_rate": round(exp1_wins / total * 100, 2),
                f"{exp_2_name}_win_rate": round(exp2_wins / total * 100, 2),
                "tie_rate":               round(ties / total * 100, 2),

                # ── Arena score stats (kept for backward compatibility) ────────
                "arena_score_mean": round(df["arena_score"].mean(), 4),
                "arena_score_min":  float(df["arena_score"].min()),
                "arena_score_max":  float(df["arena_score"].max()),

                # ── Answer quality ────────────────────────────────────────────
                "avg_answer_length_A":  round(df["answer_length_A"].mean(), 1),
                "avg_answer_length_B":  round(df["answer_length_B"].mean(), 1),
                "avg_reasoning_length": round(df["reasoning_length"].mean(), 1),

                # ── Latency & cost ────────────────────────────────────────────
                "total_judge_latency_s":    round(df["judge_latency_s"].sum(), 2),
                "avg_judge_latency_s":      round(df["judge_latency_s"].mean(), 2),
                "total_gpt_cost_estimated": round(df["gpt_cost_estimated"].sum(), 6),
                "total_gpt_cost_framework": round(df["gpt_cost_framework"].sum(), 6),
                "avg_gpt_cost_estimated":   round(df["gpt_cost_estimated"].mean(), 6),
                "avg_gpt_cost_framework":   round(df["gpt_cost_framework"].mean(), 6),
                "n_skipped":                len(skipped),
            })

            for v in ARENA_SCORE_MAP:
                mlflow.log_metric(f"verdict_{v}", int((df["verdict"] == v).sum()))

            if skipped:
                mlflow.log_text("\n".join(skipped), "skipped_questions.txt")

            # Save battles table as artifact (useful for re-running BT analysis)
            battles.to_csv("battles.csv", index=False)
            mlflow.log_artifact("battles.csv")
            os.remove("battles.csv")

            df.to_csv("benchmark_results.csv", index=False)
            mlflow.log_artifact("benchmark_results.csv")
            os.remove("benchmark_results.csv")

        # ── Console summary ───────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print(f"📊 BENCHMARK SUMMARY: {exp_1_name} vs {exp_2_name}")
        print("=" * 65)
        print(f"   Questions judged : {total}  |  skipped: {len(skipped)}")
        print()
        print(f"   {'Model':<35} {'BT Score':>8}  {'95% CI':>14}  {'Elo':>7}")
        print(f"   {'-'*35} {'-'*8}  {'-'*14}  {'-'*7}")
        print(
            f"   {exp_1_name:<35} {bt_score_A:>7.2f}%"
            f"  ({bt_A['ci_lower']:+.2f} / {bt_A['ci_upper']:+.2f})"
            f"  {elo_A:>7.1f}"
        )
        print(
            f"   {exp_2_name:<35} {bt_score_B:>7.2f}%"
            f"  ({bt_B['ci_lower']:+.2f} / {bt_B['ci_upper']:+.2f})"
            f"  {elo_B:>7.1f}"
        )
        print()
        print(f"   Raw wins  →  {exp_1_name}: {exp1_wins}  |  Tie: {ties}  |  {exp_2_name}: {exp2_wins}")
        print(f"   Total cost (estimated): ${df['gpt_cost_estimated'].sum():.4f}")
        print(f"   Total cost (framework): ${df['gpt_cost_framework'].sum():.4f}")
        if skipped:
            print(f"   ⚠️  Skipped IDs → skipped_questions.txt in summary run artifacts")
        print(f"\n🔗 View results: {MLFLOW_TRACKING_URI}/#/experiments")
    else:
        print("\n⚠️  No results produced.")
        if skipped:
            print(f"   Skipped questions ({len(skipped)}): {', '.join(skipped)}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arena-Hard benchmark: compare two MLflow experiments with an LLM judge."
    )
    parser.add_argument(
        "--model-a",
        dest="model_a",
        required=True,
        help="MLflow experiment name for model A (the baseline).",
    )
    parser.add_argument(
        "--model-b",
        dest="model_b",
        required=True,
        help="MLflow experiment name for model B (the challenger).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(
        exp_1_name=args.model_a,
        exp_2_name=args.model_b,
    ))