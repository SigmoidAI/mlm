import asyncio
import mlflow
from mlflow.genai.datasets import get_dataset

import mlflow.pydantic_ai

import os
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..models.schemas import *

MLFLOW_URI = "http://127.0.0.1:5000"
MODEL_NAME = "GPT4o"

mlflow.set_tracking_uri(MLFLOW_URI)

try:
    mlflow.pydantic_ai.autolog()
    print("✅ PydanticAI Autologging enabled")
except Exception as e:
    print(f"⚠️  Could not enable autologging: {e}")


# ---------- Experiment versioning ----------
def get_or_create_versioned_experiment(model_name: str) -> tuple[str, str, int]:
    """
    Find latest version of experiment for given model, increment it.
    Pattern: {model_name}_v{version}
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    all_experiments = client.search_experiments(
        filter_string=f"name LIKE '{model_name}_v%'",
        order_by=["creation_time DESC"]
    )

    if all_experiments:
        versions = []
        for exp in all_experiments:
            try:
                version_str = exp.name.split("_v")[-1]
                versions.append(int(version_str))
            except (ValueError, IndexError):
                continue
        next_version = max(versions) + 1 if versions else 1
    else:
        next_version = 1

    experiment_name = f"{model_name}_v{next_version}"
    experiment_id = mlflow.create_experiment(experiment_name)

    print(f"✅ Created experiment: {experiment_name} (ID: {experiment_id})")
    return experiment_id, experiment_name, next_version


# ---------- Dataset -----------
##TODO past your dataset id here
dataset = get_dataset(dataset_id="d-751bd16f3e8748f3a71f361fcc6d59f5")
records = dataset.to_dict()
print(f"📂 Loaded {len(records['records'])} records from dataset")

# ---------- Azure Setup ----------
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "GPT4o")

azure_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

model = OpenAIChatModel(
    AZURE_DEPLOYMENT_NAME,
    provider=OpenAIProvider(openai_client=azure_client)
)

agent = Agent(
    model,
    output_type=AgentResponse,
    system_prompt="You are GPT-4, a helpful AI assistant. Provide detailed, accurate, and comprehensive answers to user questions."
)


async def generate_answer(agent, question: str) -> str:
    result = await agent.run(question)
    if hasattr(result, 'data'):
        return result.data
    elif hasattr(result, 'output'):
        return result.output
    return str(result)


def run_evaluation():
    experiment_id, experiment_name, version = get_or_create_versioned_experiment(MODEL_NAME)
    mlflow.set_experiment(experiment_name)

    print(f"\n🚀 Starting evaluation")
    print(f"   Experiment: {experiment_name}")
    print(f"   Model: {AZURE_DEPLOYMENT_NAME}")
    print(f"   Questions: {len(records['records'][:10])}\n")

    successful = 0
    failed = 0

    for idx, record in enumerate(records["records"][:10]):
        question = record["inputs"]["question"]
        question_id = record["inputs"]["question_id"]
        category = record["inputs"].get("category", "unknown")

        print(f"[{idx + 1}/10] {question_id} ({category})")

        # Each question is its own run, named by question_id
        with mlflow.start_run(run_name=question_id) as run:
            mlflow.log_params({
                "question_id": question_id,
                "category": category,
                "model_name": MODEL_NAME,
                "model_version": version,
                "deployment_name": AZURE_DEPLOYMENT_NAME,
            })

            try:
                @mlflow.trace(name="baseline_agent_answer", attributes={
                    "question_id": question_id,
                    "category": category
                })
                def generate_baseline_answer(q):
                    return asyncio.run(generate_answer(agent, q))

                answer = generate_baseline_answer(question)
                trace_id = mlflow.get_last_active_trace_id()

                answer_text = answer.content if hasattr(answer, 'content') else str(answer)

                mlflow.log_feedback(
                    trace_id=trace_id,
                    name="baseline_response",
                    value=True,
                    rationale=f"Baseline response for {question_id}",
                    source=mlflow.entities.AssessmentSource(
                        source_type=mlflow.entities.AssessmentSourceType.CODE,
                        source_id=f"baseline_{MODEL_NAME}_v{version}"
                    ),
                    metadata={
                        "question_id": question_id,
                        "category": category,
                        "model": AZURE_DEPLOYMENT_NAME,
                        "version": version,
                        "answer_length": len(answer_text)
                    }
                )

                mlflow.log_metrics({
                    "answer_length": len(answer_text),
                    "success": 1
                })

                successful += 1
                print(f"   ✅ {answer_text[:100]}...")

            except Exception as e:
                failed += 1
                print(f"   ❌ {e}")

                mlflow.log_metrics({"success": 0})

                if 'trace_id' in locals() and trace_id:
                    mlflow.log_feedback(
                        trace_id=trace_id,
                        name="baseline_response",
                        value=False,
                        rationale=f"Error: {str(e)}",
                        source=mlflow.entities.AssessmentSource(
                            source_type=mlflow.entities.AssessmentSourceType.CODE,
                            source_id=f"baseline_{MODEL_NAME}_v{version}"
                        ),
                        metadata={
                            "question_id": question_id,
                            "error": str(e)
                        }
                    )

    total = successful + failed
    print(f"\n{'=' * 80}")
    print(f"📊 {experiment_name} — {successful}/{total} successful ({successful / total * 100:.1f}%)")
    print(f"🔗 {MLFLOW_URI}/#/experiments/{experiment_id}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_evaluation()