# import os
# import pandas as pd
# import mlflow
# import asyncio
# import warnings
# from typing import Literal
# from pydantic import BaseModel, Field
# from pydantic_ai import Agent
# from pydantic_ai.models.openai import OpenAIChatModel
# from pydantic_ai.providers.openai import OpenAIProvider
# from openai import AsyncAzureOpenAI
# from mlflow.entities import AssessmentSource, AssessmentSourceType
# import nest_asyncio
#
#
#
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", module="openai")
#
# nest_asyncio.apply()
#
# # Azure Configuration
# # AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
# # AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
# # AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
# # AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "GPT4o")
# MLFLOW_URI = "http://127.0.0.1:5000"
#
# # os.environ["AZURE_OPENAI_API_KEY"] = AZURE_API_KEY
# # os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT
# # os.environ["OPENAI_API_VERSION"] = AZURE_API_VERSION
#
# mlflow.set_tracking_uri(MLFLOW_URI)
# mlflow.set_experiment("Agent_Judge_Experiment_v4")
#
#
# # Define PydanticAI Judge
# class JudgeVerdict(BaseModel):
#     score: int = Field(description="Score between 1 and 5")
#     reasoning: str = Field(description="Detailed justification for the score")
#
#
# # azure_client = AsyncAzureOpenAI(
# #     azure_endpoint=AZURE_ENDPOINT,
# #     api_key=AZURE_API_KEY,
# #     api_version=AZURE_API_VERSION,
# # )
# # judge_model = OpenAIChatModel(AZURE_DEPLOYMENT_NAME, provider=OpenAIProvider(openai_client=azure_client))
#
# judge_agent = Agent(
#     judge_model,
#     output_type=JudgeVerdict,
#     system_prompt=(
#         "You are a Code Quality Reviewer.\n\n"
#         "Scoring criteria:\n"
#         "5 = Perfect - Complete, correct, well-explained\n"
#         "4 = Good - Minor improvements possible\n"
#         "3 = Adequate - Correct but lacks depth\n"
#         "2 = Poor - Incomplete or partially incorrect\n"
#         "1 = Very Poor - Incorrect or misleading\n\n"
#         "Be strict but fair in your evaluation."
#     )
# )
#
# # Import or define your agent
# try:
#     from main import run_baseline_test
# except ImportError:
#     print("⚠️ WARNING: Could not import 'run_baseline_test'. Using a dummy function for testing.")
#
#
#     async def run_baseline_test(query):
#         from collections import namedtuple
#         Response = namedtuple('Response', ['content'])
#         return Response(content=f"To use async, define functions with 'async def'.")
#
#
# def evaluate_agent():
#     # Prepare evaluation data
#     eval_data = [
#         {
#             "inputs": {"query": "How do I write async python?"},
#             "expectations": {"expected_response": "Use async def to define coroutines."}
#         },
#         {
#             "inputs": {"query": "What is the difference between async and await?"},
#             "expectations": {
#                 "expected_response": "Async defines a coroutine, while await pauses execution until the awaitable completes."}
#         }
#     ]
#
#     # Define predict function that returns trace
#     @mlflow.trace
#     def model_fn(query):
#         res = asyncio.run(run_baseline_test(query))
#         return res.content
#
#     # Run evaluation
#     with mlflow.start_run(run_name="pydantic_ai_judge_evaluation"):
#         # Execute predictions and get traces
#         for item in eval_data:
#             query = item["inputs"]["query"]
#             response = model_fn(query)
#             trace_id = mlflow.get_last_active_trace_id()
#
#             # Evaluate with PydanticAI judge
#             prompt = f"User Question: {query}\n\nModel Answer: {response}"
#
#             try:
#                 result = asyncio.run(judge_agent.run(prompt))
#
#                 if hasattr(result, 'data'):
#                     score_obj = result.data
#                 elif hasattr(result, 'output'):
#                     score_obj = result.output
#                 else:
#                     raise AttributeError("Could not find .data or .output in result object")
#
#                 # Log feedback to MLflow
#                 mlflow.log_feedback(
#                     trace_id=trace_id,
#                     name="CodeQuality",
#                     value=score_obj.score >= 4,  # Convert to pass/fail
#                     rationale=score_obj.reasoning,
#                     source=AssessmentSource(
#                         source_type=AssessmentSourceType.LLM_JUDGE,
#                         source_id="pydantic_ai_judge_v1"
#                     ),
#                     metadata={"numeric_score": score_obj.score}
#                 )
#
#                 print(f"✅ Evaluated trace {trace_id}: Score {score_obj.score}/5")
#
#             except Exception as e:
#                 print(f"❌ Error evaluating trace {trace_id}: {e}")
#                 mlflow.log_feedback(
#                     trace_id=trace_id,
#                     name="CodeQuality",
#                     value=False,
#                     rationale=f"Error during evaluation: {str(e)}",
#                     source=AssessmentSource(
#                         source_type=AssessmentSourceType.LLM_JUDGE,
#                         source_id="pydantic_ai_judge_v1"
#                     ),
#                     metadata={"error": str(e)}
#                 )
#
#
# if __name__ == "__main__":
#     print("🚀 Starting Evaluation with PydanticAI Judge...")
#     print(f"   MLflow URI: {MLFLOW_URI}")
#
#     try:
#         evaluate_agent()
#         print("\n✅ Evaluation Complete! Check MLflow UI for results.")
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#
#         traceback.print_exc()