import mlflow
import asyncio
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import AssessmentSource, AssessmentSourceType
from typing import Dict, Any, List
import os
import time

from .arena_judger import ArenaValidatorAgent
from ..config.make_config import make_config,_load_env

# ==============================================================================
# CONFIGURATION
# ==============================================================================
_load_env()
CASCADE_MODELS_CONFIG: dict[str, str] = make_config()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set.")

EXP_1_NAME = os.getenv("BENCHMARK_NAME_1","Arena_Hard_GPT-4.1")
EXP_2_NAME = os.getenv("BENCHMARK_NAME_2","complex_workflow_run_max_100_v2")

COMPARISON_EXP_NAME = f"{EXP_1_NAME}_vs_{EXP_2_NAME}"

# Arena Hard weighted scoring (matches the leaderboard methodology)
ARENA_SCORE_MAP = {
    "A>>B": 1.0,
    "A>B": 0.5,
    "A=B": 0.0,
    "B>A": -0.5,
    "B>>A": -1.0,
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


def verdict_to_winner(verdict: str) -> str:
    if verdict in ("A>>B", "A>B"):
        return EXP_1_NAME
    elif verdict in ("B>A", "B>>A"):
        return EXP_2_NAME
    return "Tie"


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


# ==============================================================================
# TRACING WRAPPER
# ==============================================================================
@mlflow.trace(name="arena_judge_evaluation", span_type="LLM")
async def execute_traced_judge(validator, question, answer_a, answer_b, question_id):
    """
    Wraps the validation step in an MLflow trace.
    This ensures inputs (answers) and outputs (verdict) are visible in the Traces tab.
    """
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "question_id": question_id,
            "model_A": EXP_1_NAME,
            "model_B": EXP_2_NAME
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
async def run_benchmark():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"📡 Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    print(f"⚔️  Comparing: {EXP_1_NAME} vs {EXP_2_NAME}\n")

    # ── 1. Fetch runs from both experiments ──────────────────────────────────
    runs_1 = get_experiment_runs(client, EXP_1_NAME)
    runs_2 = get_experiment_runs(client, EXP_2_NAME)

    common_names = sorted(set(runs_1.keys()) & set(runs_2.keys()))
    print(f"📂 {EXP_1_NAME}: {len(runs_1)} runs")
    print(f"📂 {EXP_2_NAME}: {len(runs_2)} runs")
    print(f"🔗 Overlapping question_ids: {len(common_names)}\n")

    if not common_names:
        print("⚠️  No overlapping runs found. Nothing to compare.")
        return

    # ── 2. Resolve experiment IDs once ────────────────────────────────────────
    exp_1_id = client.get_experiment_by_name(EXP_1_NAME).experiment_id
    exp_2_id = client.get_experiment_by_name(EXP_2_NAME).experiment_id

    # ── 3. Create / set the comparison experiment ────────────────────────────
    mlflow.set_experiment(COMPARISON_EXP_NAME)
    comparison_exp_id = client.get_experiment_by_name(COMPARISON_EXP_NAME).experiment_id

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

        # ── Outer try/except: skip this question entirely on any error ────────
        try:
            run_a = runs_1[run_name]
            run_b = runs_2[run_name]

            run_id_a = run_a.info.run_id
            run_id_b = run_b.info.run_id

            # ── Framework cost from OpenRouter hooks (logged on run_b) ───────
            total_cost = run_b.data.metrics.get("total_cost", None)

            # ── Pull traces (contains request/response) ──────────────────────
            try:
                traces_a = extract_traces(client, run_id_a, exp_1_id)
                traces_b = extract_traces(client, run_id_b, exp_2_id)

                question_a = traces_a.iloc[0]["request"]["q"]
                answer_a   = traces_a.iloc[0]["response"]["output"]
                trace_id_a = traces_a.iloc[0]["trace_id"]
                gpt_cost_info = estimate_gpt_cost(question_a, answer_a)

                correct_traces = traces_b[
                    traces_b["response"].apply(
                        lambda r: isinstance(r, dict) and "final_best_response" in r
                    )
                ]

                if not correct_traces.empty:
                    answer_b   = correct_traces.iloc[0]["response"]["final_best_response"]
                    trace_id_b = correct_traces.iloc[0]["trace_id"]
                else:
                    print(f"   ⚠️  Skipping: Could not extract answer from exp B.")
                    skipped.append(run_name)
                    continue
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

            # ── Log everything into a per-question run ───────────────────────
            with mlflow.start_run(run_name=run_name) as comparison_run:
                comparison_run_id = comparison_run.info.run_id
                per_question_run_ids.append(comparison_run_id)

                # ── Judge (forward + reverse) ──────────────────────────────────
                judge_start = time.time()

                try:
                    validation_fwd = await execute_traced_judge(
                        validator,
                        question=question,
                        answer_a=answer_a,
                        answer_b=answer_b,
                        question_id=f"{run_name}_fwd"
                    )
                except Exception as e:
                    print(f"   ❌ Forward judge call failed for {run_name}: {e}")
                    raise  # bubble up to the outer handler to skip this question

                verdict_fwd   = validation_fwd.get("verdict", "A=B")
                reasoning_fwd = validation_fwd.get("reasoning", "")
                score_fwd     = ARENA_SCORE_MAP.get(verdict_fwd, 0.0)

                verdict   = verdict_fwd
                reasoning = reasoning_fwd

                try:
                    validation_rev = await execute_traced_judge(
                        validator,
                        question=question,
                        answer_a=answer_b,  # swapped
                        answer_b=answer_a,
                        question_id=f"{run_name}_rev"
                    )
                except Exception as e:
                    print(f"   ❌ Reverse judge call failed for {run_name}: {e}")
                    raise  # bubble up to the outer handler to skip this question

                verdict_rev_raw = validation_rev.get("verdict", "A=B")
                reasoning_rev   = validation_rev.get("reasoning", "")
                score_rev_raw   = ARENA_SCORE_MAP.get(verdict_rev_raw, 0.0)
                score_rev       = -score_rev_raw

                arena_score     = (score_fwd + score_rev) / 2.0
                judge_latency_s = time.time() - judge_start
                judge_trace_id  = mlflow.get_last_active_trace_id()

                if arena_score > 0:
                    winner = EXP_1_NAME
                elif arena_score < 0:
                    winner = EXP_2_NAME
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
                    f"cost_fw=${ f'{total_cost:.6f}' if total_cost is not None else 'N/A'})"
                )

                # --- Params ----------------------------------------------------
                mlflow.log_params({
                    "question_id":       run_name,
                    "model_A":           EXP_1_NAME,
                    "model_B":           EXP_2_NAME,
                    "verdict":           verdict,
                    "winner":            winner,
                    "source_run_id_A":   run_id_a,
                    "source_run_id_B":   run_id_b,
                    "source_trace_id_A": str(trace_id_a),
                    "source_trace_id_B": str(trace_id_b),
                    "judge_model":       "google/gemini-2.5-flash",
                    "judge_trace_id":    str(judge_trace_id) if judge_trace_id else "N/A",
                })

                # --- Metrics --------------------------------------------------
                mlflow.log_metrics({
                    "arena_score_forward":      score_fwd,
                    "arena_score_reverse":      score_rev,
                    "arena_score_pairwise_avg": arena_score,
                    "arena_score":              arena_score,

                    "win_binary": 1 if winner == EXP_1_NAME else (-1 if winner == EXP_2_NAME else 0),

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

                # --- Artifacts ------------------------------------------------
                mlflow.log_text(question,      "question.txt")
                mlflow.log_text(answer_a,      "response_A.txt")
                mlflow.log_text(answer_b,      "response_B.txt")
                mlflow.log_text(reasoning_fwd, "reasoning_forward.txt")
                mlflow.log_text(reasoning_rev, "reasoning_reverse.txt")

                # --- Feedback on SOURCE traces --------------------------------
                for trace_id, source_exp, label in [
                    (trace_id_a, EXP_1_NAME, "A"),
                    (trace_id_b, EXP_2_NAME, "B"),
                ]:
                    try:
                        mlflow.log_feedback(
                            trace_id=str(trace_id),
                            name="ArenaHardComparison",
                            value=(winner == source_exp) or (winner == "Tie"),
                            rationale=f"[{verdict}] {reasoning[:500]}",
                            source=AssessmentSource(
                                source_type=AssessmentSourceType.LLM_JUDGE,
                                source_id=f"arena_judge_{COMPARISON_EXP_NAME}"
                            ),
                            metadata={
                                "verdict":            verdict,
                                "winner":             winner,
                                "arena_score":        arena_score,
                                "opponent":           EXP_2_NAME if label == "A" else EXP_1_NAME,
                                "comparison_run_id":  comparison_run_id,
                                "question_id":        run_name,
                                "judge_trace_id":     str(judge_trace_id),
                                "gpt_cost_estimated": gpt_cost_info["cost"],
                                "gpt_cost_framework": total_cost if total_cost is not None else 0.0,
                            }
                        )
                    except Exception as e:
                        print(f"   ⚠️  Could not log feedback on trace {trace_id} ({label}): {e}")

            # Accumulate for summary (only reached if no exception above)
            results.append({
                "question_id":        run_name,
                "verdict":            verdict,
                "winner":             winner,
                "arena_score":        arena_score,
                "answer_length_A":    len(answer_a),
                "answer_length_B":    len(answer_b),
                "reasoning_length":   len(reasoning),
                "judge_latency_s":    round(judge_latency_s, 3),
                "gpt_cost_estimated": gpt_cost_info["cost"],
                "gpt_cost_framework": total_cost if total_cost is not None else 0.0,
            })

        except Exception as e:
            # Catches any unhandled error in the entire question block —
            # logs it and moves on to the next question.
            print(f"   💥 Unexpected error on {run_name}, skipping: {type(e).__name__}: {e}")
            skipped.append(run_name)
            continue

    # ── 6. Parent summary run ─────────────────────────────────────────────────
    if results:
        df = pd.DataFrame(results)
        total = len(df)

        final_win_val_A = df["arena_score"].sum() / total
        final_win_val_B = -df["arena_score"].sum() / total
        final_win_val   = final_win_val_A - final_win_val_B

        exp1_wins = len(df[df["winner"] == EXP_1_NAME])
        exp2_wins = len(df[df["winner"] == EXP_2_NAME])
        ties      = len(df[df["winner"] == "Tie"])

        with mlflow.start_run(run_name=f"SUMMARY_{EXP_1_NAME}_vs_{EXP_2_NAME}"):
            mlflow.log_params({
                "model_A":           EXP_1_NAME,
                "model_B":           EXP_2_NAME,
                "judge_model":       "google/gemini-2.5-flash",
                "total_questions":   total,
                "total_skipped":     len(skipped),
                "exp_1_id":          exp_1_id,
                "exp_2_id":          exp_2_id,
                "comparison_exp_id": comparison_exp_id,
            })

            mlflow.log_metrics({
                "win_val":   round(final_win_val, 4),
                "win_val_A": round(final_win_val_A, 4),
                "win_val_B": round(final_win_val_B, 4),

                f"{EXP_1_NAME}_wins":     exp1_wins,
                f"{EXP_2_NAME}_wins":     exp2_wins,
                "ties":                   ties,
                f"{EXP_1_NAME}_win_rate": round(exp1_wins / total * 100, 2),
                f"{EXP_2_NAME}_win_rate": round(exp2_wins / total * 100, 2),
                "tie_rate":               round(ties / total * 100, 2),

                "arena_score_sum":  round(df["arena_score"].sum(), 4),
                "arena_score_mean": round(df["arena_score"].mean(), 4),
                "arena_score_min":  float(df["arena_score"].min()),
                "arena_score_max":  float(df["arena_score"].max()),

                "avg_answer_length_A":  round(df["answer_length_A"].mean(), 1),
                "avg_answer_length_B":  round(df["answer_length_B"].mean(), 1),
                "avg_reasoning_length": round(df["reasoning_length"].mean(), 1),

                "total_judge_latency_s": round(df["judge_latency_s"].sum(), 2),
                "avg_judge_latency_s":   round(df["judge_latency_s"].mean(), 2),

                "total_gpt_cost_estimated": round(df["gpt_cost_estimated"].sum(), 6),
                "total_gpt_cost_framework": round(df["gpt_cost_framework"].sum(), 6),
                "avg_gpt_cost_estimated":   round(df["gpt_cost_estimated"].mean(), 6),
                "avg_gpt_cost_framework":   round(df["gpt_cost_framework"].mean(), 6),

                "n_skipped": len(skipped),
            })

            for v in ARENA_SCORE_MAP:
                mlflow.log_metric(f"verdict_{v}", int((df["verdict"] == v).sum()))

            # Log skipped question IDs as an artifact for debugging
            if skipped:
                skipped_txt = "\n".join(skipped)
                mlflow.log_text(skipped_txt, "skipped_questions.txt")

            df.to_csv("benchmark_results.csv", index=False)
            mlflow.log_artifact("benchmark_results.csv")
            os.remove("benchmark_results.csv")

        # ── Console summary ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"📊 BENCHMARK SUMMARY: {EXP_1_NAME} vs {EXP_2_NAME}")
        print("=" * 60)
        print(f"   Questions judged      : {total}")
        print(f"   Questions skipped     : {len(skipped)}")
        print(f"   win_val_A ({EXP_1_NAME}): {final_win_val_A:+.4f}")
        print(f"   win_val_B ({EXP_2_NAME}): {final_win_val_B:+.4f}")
        print(f"   win_val (A − B)       : {final_win_val:+.4f}")
        print(f"   {EXP_1_NAME:>12} wins : {exp1_wins:>3}  ({exp1_wins / total * 100:.1f}%)")
        print(f"   {'Ties':>12}      : {ties:>3}  ({ties / total * 100:.1f}%)")
        print(f"   {EXP_2_NAME:>12} wins : {exp2_wins:>3}  ({exp2_wins / total * 100:.1f}%)")
        print(f"   Total cost (estimated): ${df['gpt_cost_estimated'].sum():.4f}")
        print(f"   Total cost (framework): ${df['gpt_cost_framework'].sum():.4f}")
        print(f"   Avg cost/q (estimated): ${df['gpt_cost_estimated'].mean():.6f}")
        print(f"   Avg cost/q (framework): ${df['gpt_cost_framework'].mean():.6f}")
        if skipped:
            print(f"   ⚠️  Skipped IDs logged to skipped_questions.txt in the summary run artifacts")
        print(f"\n🔗 View results: {MLFLOW_TRACKING_URI}/#/experiments")
    else:
        print("\n⚠️  No results produced.")
        if skipped:
            print(f"   Skipped questions ({len(skipped)}): {', '.join(skipped)}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())