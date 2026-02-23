import os
import json
import mlflow
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import re
from ..config.prompts import ARENA_HARD_JUDGE_PROMPT
from ..config.make_config import make_config

CASCADE_MODELS_CONFIG: dict[str, str] = make_config()

# ==============================================================================
# CONFIG
# ==============================================================================


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JUDGE_MODEL = "google/gemini-2.5-flash"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

model = OpenAIChatModel(JUDGE_MODEL, provider=OpenAIProvider(openai_client=client))


# ==============================================================================
# SCHEMAS
# ==============================================================================
class ValidationResult(BaseModel):
    verdict: str = Field(description="One of: A>>B, A>B, A=B, B>A, B>>A")
    reasoning: str = Field(description="Detailed explanation for the verdict")


# ==============================================================================
# VALIDATOR
# ==============================================================================
class ArenaValidatorAgent:
    def __init__(
            self,
            model_name: str = "google/gemini-2.5-flash",  # Default model
            api_key: Optional[str] = None
    ):
        """
        Initializes the ValidatorAgent with a specific OpenRouter model.

        Args:
            model_name (str): The OpenRouter model ID (e.g., 'anthropic/claude-3-opus').
            api_key (str, optional): OpenRouter API Key. Defaults to env var 'OPENROUTER_API_KEY'.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing. Please set it in env or pass it to __init__.")

        # 1. Initialize the Async Client for OpenRouter
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # 2. Initialize the PydanticAI Model
        # OpenAIProvider because OpenRouter is OpenAI-compatible
        self.model = OpenAIChatModel(
            self.model_name,
            provider=OpenAIProvider(openai_client=self.client)
        )

        # 3. Initialize the Agent
        # Append the JSON instruction to the system prompt to ensure parsable output
        system_instruction = (
            f"{ARENA_HARD_JUDGE_PROMPT}\n\n"
            "After your verdict, output a JSON block exactly like this:\n"
            "```json\n{\"verdict\": \"[[A>B]]\", \"reasoning\": \"your explanation\"}\n```"
        )

        self.judge_agent = Agent(
            self.model,
            system_prompt=system_instruction
        )

        self.memory: List[Dict[str, Any]] = []

    def _parse_verdict(self, verdict_str: str) -> str:
        """Standardizes the verdict string."""
        match = re.search(r'\[\[([^\]]+)\]\]', verdict_str)
        if match:
            return match.group(1)
        if verdict_str in ['A>>B', 'A>B', 'A=B', 'B>A', 'B>>A']:
            return verdict_str
        return 'A=B'

    def _parse_response(self, raw_text: str) -> ValidationResult:
        """Extract ValidationResult from raw model text output."""
        # Try to find JSON block in response
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    data = data[0] if data and isinstance(data[0], dict) else {}
                return ValidationResult(
                    verdict=data.get("verdict", "[[A=B]]"),
                    reasoning=data.get("reasoning", raw_text)
                )
            except json.JSONDecodeError:
                pass

        # Fallback: extract verdict from anywhere in text
        verdict_match = re.search(r'\[\[([^\]]+)\]\]', raw_text)
        return ValidationResult(
            verdict=verdict_match.group(0) if verdict_match else "[[A=B]]",
            reasoning=raw_text
        )

    async def validate(
            self,
            question: str,
            answer_a: str,
            answer_b: str,
            question_id: str = "",
            category: str = "general"
    ) -> Dict[str, Any]:

        judge_prompt = f"""[User Prompt]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

        print(f"⚖️  Judging {question_id} using model: {self.model_name}...")

        # Run the agent
        judge_result = await self.judge_agent.run(judge_prompt)

        # Handle output (Standard PydanticAI response handling)
        raw_text = judge_result.data if hasattr(judge_result, 'data') else str(judge_result.output)

        # Parse logic
        validation = self._parse_response(raw_text)
        verdict = self._parse_verdict(validation.verdict)

        print(f"📊 Verdict: {verdict}")

        result = {
            "question_id": question_id,
            "category": category,
            "model_used": self.model_name,
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "verdict": verdict,
            "reasoning": validation.reasoning,
        }

        self.memory.append(result)
        return result

# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    DUMMY_QUESTION = (
        "Is there an early stop out method (to control for multiple testing problem "
        "in hypothesis tests) for a dataset with initial probabilities of passing?"
    )

    DUMMY_ANSWER_A = (
        "Yes. You can use the Bonferroni correction or the Benjamini-Hochberg procedure "
        "to control for multiple testing. The Bonferroni method divides your significance "
        "level by the number of tests, which is very conservative. Benjamini-Hochberg "
        "controls the false discovery rate instead, which is less conservative and more "
        "suitable when you have many strategies. For early stopping specifically, you can "
        "use sequential testing frameworks like alpha-spending functions (e.g., O'Brien-Fleming), "
        "which allow you to stop testing early if results are clearly significant or clearly not."
    )

    DUMMY_ANSWER_B = (
        "You can just run all your tests and see which ones pass. "
        "If too many fail, maybe try fewer strategies. "
        "There's no special method needed for this."
    )

    async def main():
        validator = ArenaValidatorAgent(model_name="deepseek/deepseek-r1",
                                   api_key=OPENROUTER_API_KEY)

        result = await validator.validate(
            question=DUMMY_QUESTION,
            answer_a=DUMMY_ANSWER_A,
            answer_b=DUMMY_ANSWER_B,
            question_id="arena_hard_44",
            category="math"
        )

        print(f"\n{'='*60}")
        print(f"Question ID : {result['question_id']}")
        print(f"Verdict     : {result['verdict']}")
        print(f"Reasoning   : {result['reasoning']}")
        print(f"{'='*60}")

    asyncio.run(main())