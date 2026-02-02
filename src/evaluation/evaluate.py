import mlflow
import asyncio
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import AssessmentSource, AssessmentSourceType
from typing import Dict, Any, List
import os
import time

from .arena_judger import ArenaValidatorAgent

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set.")

EXP_1_NAME = "GPT4o_v2"
EXP_2_NAME = "GPT4o_v4"
COMPARISON_EXP_NAME = f"{EXP_1_NAME}_vs_{EXP_2_NAME}_Benchmark"

# Arena Hard weighted scoring (matches the leaderboard methodology)
ARENA_SCORE_MAP = {
    "A>>B": 1.0,
    "A>B": 0.5,
    "A=B": 0.0,
    "B>A": -0.5,
    "B>>A": -1.0,
}


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


# ==============================================================================
# TRACING WRAPPER (ADDED)
# ==============================================================================
@mlflow.trace(name="arena_judge_evaluation", span_type="LLM")
async def execute_traced_judge(validator, question, answer_a, answer_b, question_id):
    """
    Wraps the validation step in an MLflow trace.
    This ensures inputs (answers) and outputs (verdict) are visible in the Traces tab.
    """
    # Add context attributes to the trace for easy filtering later
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "question_id": question_id,
            "model_A": EXP_1_NAME,
            "model_B": EXP_2_NAME
        })

    # Execute the actual judge
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
        model_name="tngtech/deepseek-r1t2-chimera:free",
        api_key=OPENROUTER_API_KEY
    )

    # ── 5. Per-question judging loop ──────────────────────────────────────────
    results: List[Dict[str, Any]] = []

    # Running accumulators — A's score is the raw arena_score,
    # B's score is its mirror (B wins when A loses and vice-versa).
    cumulative_score_A = 0.0
    cumulative_score_B = 0.0

    per_question_run_ids: List[str] = []

    for idx, run_name in enumerate(common_names, 1):
        print(f"\n[{idx}/{len(common_names)}] Judging: {run_name}")

        run_a = runs_1[run_name]
        run_b = runs_2[run_name]

        run_id_a = run_a.info.run_id
        run_id_b = run_b.info.run_id

        # ── Pull traces (contains request/response) ──────────────────────────
        try:
            traces_a = extract_traces(client, run_id_a, exp_1_id)
            traces_b = extract_traces(client, run_id_b, exp_2_id)

            question_a = traces_a.iloc[0]["request"]["q"]
            answer_a = traces_a.iloc[0]["response"]["content"]
            trace_id_a = traces_a.iloc[0]["trace_id"]

            question_b = traces_b.iloc[0]["request"]["q"]
            answer_b = traces_b.iloc[0]["response"]["content"]
            trace_id_b = traces_b.iloc[0]["trace_id"]
        except Exception as e:
            print(f"   ❌ Failed to extract traces for {run_name}: {e}")
            continue

        question = question_a or question_b
        if not question:
            print(f"   ⚠️  Skipping: Could not extract question.")
            continue
        if not answer_a or not answer_b:
            print(f"   ⚠️  Skipping: Missing answer (A={bool(answer_a)}, B={bool(answer_b)})")
            continue


        # ── Log everything into a per-question run ───────────────────────────
        with mlflow.start_run(run_name=run_name) as comparison_run:
            comparison_run_id = comparison_run.info.run_id
            per_question_run_ids.append(comparison_run_id)

            # ── Judge ─────────────────────────────────────────────────────────────
            judge_start = time.time()

            # [CHANGED] Use the wrapper function to trigger tracing
            validation = await execute_traced_judge(
                validator,
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
                question_id=run_name
            )

            # [CHANGED] Capture the trace ID of the judge's execution
            judge_trace_id = mlflow.get_last_active_trace_id()

            judge_latency_s = time.time() - judge_start

            verdict = validation.get("verdict", "A=B")
            reasoning = validation.get("reasoning", "")
            winner = verdict_to_winner(verdict)
            arena_score = ARENA_SCORE_MAP.get(verdict, 0.0)

            # Per-response scores: A gets the raw arena_score, B gets its mirror.
            # e.g. verdict A>>B → A gets +1.0, B gets -1.0
            score_A = arena_score
            score_B = -arena_score

            cumulative_score_A += score_A
            cumulative_score_B += score_B
            n_judged = idx  # idx is 1-based

            running_win_val_A = cumulative_score_A / n_judged
            running_win_val_B = cumulative_score_B / n_judged

            print(
                f"   📊 Verdict: {verdict} → 🏆 {winner}  (arena={arena_score:+.1f} | win_val_A={running_win_val_A:+.3f} win_val_B={running_win_val_B:+.3f})")

            # --- Params (categorical / string metadata) -------------------------
            mlflow.log_params({
                "question_id": run_name,
                "model_A": EXP_1_NAME,
                "model_B": EXP_2_NAME,
                "verdict": verdict,
                "winner": winner,
                "source_run_id_A": run_id_a,
                "source_run_id_B": run_id_b,
                "source_trace_id_A": str(trace_id_a),
                "source_trace_id_B": str(trace_id_b),
                "judge_model": "deepseek/deepseek-r1",
                # [CHANGED] Log the trace ID here
                "judge_trace_id": str(judge_trace_id) if judge_trace_id else "N/A"
            })

            # --- Metrics (numeric, searchable, plottable) ----------------------
            mlflow.log_metrics({
                # Per-question scores
                "arena_score": arena_score,  # -1 … +1 weighted
                "win_binary": 1 if winner == EXP_1_NAME else (-1 if winner == EXP_2_NAME else 0),

                # Answer-length signals
                "answer_length_A": len(answer_a),
                "answer_length_B": len(answer_b),
                "answer_length_ratio": len(answer_a) / max(len(answer_b), 1),

                # Reasoning quality signal
                "reasoning_length": len(reasoning),

                # Latency
                "judge_latency_s": round(judge_latency_s, 3),

                # ★ Running win_val per response: sum(score_X) / n_judged
                #   win_val_A tracks EXP_1, win_val_B tracks EXP_2.
                #   win_val is their difference (A - B) — the head-to-head gap.
                "win_val_A": round(running_win_val_A, 4),
                "win_val_B": round(running_win_val_B, 4),
                "win_val": round(running_win_val_A - running_win_val_B, 4),
            })

            # --- Artifacts (full text blobs) -----------------------------------
            mlflow.log_text(question, "question.txt")
            mlflow.log_text(answer_a, "response_A.txt")
            mlflow.log_text(answer_b, "response_B.txt")
            mlflow.log_text(reasoning, "reasoning.txt")

            # --- Feedback on the SOURCE traces (links judge verdict back) ------
            # This makes the verdict visible directly on the original experiment
            # traces in the MLflow UI.
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
                            "verdict": verdict,
                            "winner": winner,
                            "arena_score": arena_score,
                            "opponent": EXP_2_NAME if label == "A" else EXP_1_NAME,
                            "comparison_run_id": comparison_run_id,
                            "question_id": run_name,
                            # [CHANGED] Link back to the judge trace
                            "judge_trace_id": str(judge_trace_id)
                        }
                    )
                except Exception as e:
                    print(f"   ⚠️  Could not log feedback on trace {trace_id} ({label}): {e}")

        # Accumulate for summary
        results.append({
            "question_id": run_name,
            "verdict": verdict,
            "winner": winner,
            "arena_score": arena_score,
            "answer_length_A": len(answer_a),
            "answer_length_B": len(answer_b),
            "reasoning_length": len(reasoning),
            "judge_latency_s": round(judge_latency_s, 3),
        })

    # ── 6. Parent summary run (one run that holds the final aggregates) ──────
    if results:
        df = pd.DataFrame(results)
        total = len(df)

        # Final per-response win_vals: A's scores sum to +X, B's to -X (mirrors)
        final_win_val_A = df["arena_score"].sum() / total
        final_win_val_B = -df["arena_score"].sum() / total
        final_win_val = final_win_val_A - final_win_val_B  # head-to-head gap

        exp1_wins = len(df[df["winner"] == EXP_1_NAME])
        exp2_wins = len(df[df["winner"] == EXP_2_NAME])
        ties = len(df[df["winner"] == "Tie"])

        with mlflow.start_run(run_name=f"SUMMARY_{EXP_1_NAME}_vs_{EXP_2_NAME}"):
            # --- Params ----------------------------------------------------------
            mlflow.log_params({
                "model_A": EXP_1_NAME,
                "model_B": EXP_2_NAME,
                "judge_model": "deepseek/deepseek-r1",
                "total_questions": total,
                "exp_1_id": exp_1_id,
                "exp_2_id": exp_2_id,
                "comparison_exp_id": comparison_exp_id,
            })

            # --- Metrics ---------------------------------------------------------
            mlflow.log_metrics({
                # ★ Final scores
                "win_val": round(final_win_val, 4),
                "win_val_A": round(final_win_val_A, 4),
                "win_val_B": round(final_win_val_B, 4),

                # Win counts & rates
                f"{EXP_1_NAME}_wins": exp1_wins,
                f"{EXP_2_NAME}_wins": exp2_wins,
                "ties": ties,
                f"{EXP_1_NAME}_win_rate": round(exp1_wins / total * 100, 2),
                f"{EXP_2_NAME}_win_rate": round(exp2_wins / total * 100, 2),
                "tie_rate": round(ties / total * 100, 2),

                # Arena-score stats
                "arena_score_sum": round(df["arena_score"].sum(), 4),
                "arena_score_mean": round(df["arena_score"].mean(), 4),
                "arena_score_min": float(df["arena_score"].min()),
                "arena_score_max": float(df["arena_score"].max()),

                # Answer-length stats
                "avg_answer_length_A": round(df["answer_length_A"].mean(), 1),
                "avg_answer_length_B": round(df["answer_length_B"].mean(), 1),
                "avg_reasoning_length": round(df["reasoning_length"].mean(), 1),

                # Latency
                "total_judge_latency_s": round(df["judge_latency_s"].sum(), 2),
                "avg_judge_latency_s": round(df["judge_latency_s"].mean(), 2),
            })

            # --- Verdict distribution as individual metrics (shows as bar in UI) -
            for v in ARENA_SCORE_MAP:
                mlflow.log_metric(f"verdict_{v}", int((df["verdict"] == v).sum()))

            # --- Artifacts -------------------------------------------------------
            # Full results CSV
            df.to_csv("benchmark_results.csv", index=False)
            mlflow.log_artifact("benchmark_results.csv")
            os.remove("benchmark_results.csv")

        # ── Console summary ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"📊 BENCHMARK SUMMARY: {EXP_1_NAME} vs {EXP_2_NAME}")
        print("=" * 60)
        print(f"   Questions judged  : {total}")
        print(f"   win_val_A ({EXP_1_NAME}): {final_win_val_A:+.4f}")
        print(f"   win_val_B ({EXP_2_NAME}): {final_win_val_B:+.4f}")
        print(f"   win_val (A − B)   : {final_win_val:+.4f}")
        print(f"   {EXP_1_NAME:>12} wins : {exp1_wins:>3}  ({exp1_wins / total * 100:.1f}%)")
        print(f"   {'Ties':>12}      : {ties:>3}  ({ties / total * 100:.1f}%)")
        print(f"   {EXP_2_NAME:>12} wins : {exp2_wins:>3}  ({exp2_wins / total * 100:.1f}%)")
        print(f"\n🔗 View results: {MLFLOW_TRACKING_URI}/#/experiments")
    else:
        print("\n⚠️  No results produced.")


if __name__ == "__main__":
    asyncio.run(run_benchmark())