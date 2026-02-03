import uuid
import random
import os
import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .base import PydanticAIAgent
from ..models.schemas import (
    Prompt,
    AgentResponse,
    Argument,
    ValidationResult
)
from ..config.prompts import VALIDATOR_SYSTEM_PROMPT


def get_openrouter_api_key() -> str:
    """Get OPENROUTER_API_KEY from environment, raise if not set."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    return key


# ==============================================================================
# WORKING AGENT
# ==============================================================================
class WorkingAgent(PydanticAIAgent):
    """
    Implementation of <<custom>> WorkingAgent.
    Responsible for generating solutions and critiquing peers.
    Uses pydantic_ai Agent internally for actual LLM calls.
    """

    def __init__(
            self,
            model_id: str,
            role_name: str,
            system_instruction: str,
            config: Dict[str, Any],
            cascade_tier: str = "primary",
            api_key: Optional[str] = None
    ):
        super().__init__(model_id, role_name, system_instruction, config)
        self.cascade_tier = cascade_tier
        self.memory: List[AgentResponse] = []
        self.api_key = api_key or get_openrouter_api_key()
        
        # Extract endpoint config
        endpoint_config = config.get('endpoint', {})
        base_url = endpoint_config.get('api_base_url', 'https://openrouter.ai/api/v1')
        
        # Create OpenAI client for OpenRouter
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "MLM Cascade Evaluation",
            }
        )
        
        # Create pydantic_ai model and agent
        self.model = OpenAIChatModel(
            model_id,
            provider=OpenAIProvider(openai_client=self.client)
        )
        
        self.agent = Agent(
            self.model,
            output_type=str,
            system_prompt=system_instruction
        )

    async def generate(self, context: Prompt) -> AgentResponse:
        """Generate a response using the LLM."""
        return await self.generate_initial_solution(context)

    async def generate_initial_solution(self, user_input: Prompt) -> AgentResponse:
        """Generate initial solution by calling the actual LLM."""
        result = await self.agent.run(user_input.content)
        
        # Extract answer from result
        if hasattr(result, 'output'):
            answer = result.output
        elif hasattr(result, 'data'):
            answer = result.data
        else:
            answer = str(result)
        
        # If answer is an object with .content, extract it
        if hasattr(answer, 'content'):
            answer = answer.content
        
        response = AgentResponse(
            author_id=self.role_name,
            content=answer,
            confidence=0.85,
            arguments=[
                Argument(
                    claim="Response generated successfully.",
                    reasoning="LLM provided a coherent answer.",
                    verdict="Valid"
                )
            ],
            metadata={"model": self.model_id, "tier": self.cascade_tier}
        )
        self.memory.append(response)
        return response

    def run_sync(self, prompt: str) -> AgentResponse:
        """Synchronous wrapper for generate() - compatible with cascade script."""
        import asyncio
        user_input = Prompt(content=prompt)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            return asyncio.run(self.generate(user_input))
        else:
            # There's a running loop, use nest_asyncio or create task
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.generate(user_input))

    async def generate_critique(self, peer_responses: List[AgentResponse]) -> AgentResponse:
        """Generate critique of peer responses."""
        critique_prompt = f"Please critique the following {len(peer_responses)} responses:\n\n"
        for i, resp in enumerate(peer_responses, 1):
            critique_prompt += f"Response {i}:\n{resp.content}\n\n"
        
        result = await self.agent.run(critique_prompt)
        
        if hasattr(result, 'output'):
            critique_content = result.output
        elif hasattr(result, 'data'):
            critique_content = result.data
        else:
            critique_content = str(result)
        
        return AgentResponse(
            author_id=self.role_name,
            content=critique_content,
            confidence=0.9,
            arguments=[
                Argument(
                    claim="Critique completed.",
                    reasoning="Analyzed peer responses.",
                    verdict="Valid"
                )
            ],
            metadata={"type": "critique"}
        )

    def update_config(self, new_config: Dict[str, Any]) -> None:
        self.config.update(new_config)


# ==============================================================================
# VALIDATOR AGENT
# ==============================================================================
class ValidatorAgent:
    def __init__(
            self,
            model_name: str = "tngtech/deepseek-r1t2-chimera:free",
            api_key: Optional[str] = None,
            threshold: float = 0.8
    ):
        self.model_name = model_name
        self.api_key = api_key or get_openrouter_api_key()
        self.threshold = threshold
        self.config: Dict[str, Any] = {}
        self.memory: List[Dict[str, Any]] = []

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing.")

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        self.model = OpenAIChatModel(
            self.model_name,
            provider=OpenAIProvider(openai_client=self.client)
        )

        self.judge_agent = Agent(
            self.model,
            system_prompt=VALIDATOR_SYSTEM_PROMPT
        )

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Extract JSON from raw response — handles both bare JSON and code blocks."""
        # Try code block first
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try bare JSON (find first { to last })
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(raw_text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return {"reasoning": raw_text, "verdict": "Unknown", "score": 0.0}

    async def evaluate_single(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate a single response."""
        prompt = (
            f"[User Prompt]\n{question}\n\n"
            f"[Model Response]\n{answer}"
        )

        print(f"🔍 Evaluating single response...")
        result = await self.judge_agent.run(prompt)
        raw = result.data if hasattr(result, 'data') else str(result.output)
        parsed = self._parse_json(raw)

        print(f"📊 Verdict: {parsed.get('verdict', 'Unknown')}")
        print(f"📈 Score: {parsed.get('score', 'N/A')}")

        self.memory.append({"type": "single", "question": question, **parsed})
        return parsed

    async def evaluate_comparison(self, question: str, answer_a: str, answer_b: str) -> Dict[str, Any]:
        """Compare two responses head-to-head."""
        prompt = (
            f"[User Prompt]\n{question}\n\n"
            f"[The Start of Assistant A's Answer]\n{answer_a}\n"
            f"[The End of Assistant A's Answer]\n\n"
            f"[The Start of Assistant B's Answer]\n{answer_b}\n"
            f"[The End of Assistant B's Answer]"
        )

        print(f"⚖️  Comparing two responses...")
        result = await self.judge_agent.run(prompt)
        raw = result.data if hasattr(result, 'data') else str(result.output)
        parsed = self._parse_json(raw)

        print(f"📊 Verdict: {parsed.get('verdict', 'Unknown')}")
        print(f"🏆 Winner: {parsed.get('winner', 'Unknown')}")

        self.memory.append({"type": "comparison", "question": question, **parsed})
        return parsed

    async def validate(self, responses: List[AgentResponse]) -> ValidationResult:
        avg_conf = sum(r.confidence for r in responses) / len(responses) if responses else 0.0
        is_valid = avg_conf >= self.threshold

        return ValidationResult(
            is_valid=is_valid,
            score=avg_conf,
            feedback=["Confidence is sufficient"] if is_valid else ["Confidence too low"],
            refined_response=responses[0] if responses else None
        )

    async def synthesize_final(self, responses: List[AgentResponse]) -> AgentResponse:
        best = max(responses, key=lambda x: x.confidence)
        return AgentResponse(
            author_id="Final_Validator",
            content=f"Synthesized Answer based on {best.author_id}",
            confidence=best.confidence,
            arguments=best.arguments,
            metadata={"source_count": len(responses)}
        )

    def should_escalate(self, response: AgentResponse) -> bool:
        return response.confidence < self.threshold

    def update_config(self, new_config: Dict[str, Any]) -> None:
        self.config.update(new_config)
        if "threshold" in new_config:
            self.threshold = new_config["threshold"]


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
        validator = ValidatorAgent(
            model_name="deepseek/deepseek-r1",
            api_key=get_openrouter_api_key()
        )

        print("=" * 60)
        print("TEST 1: Single evaluation (good answer)")
        print("=" * 60)
        result_single = await validator.evaluate_single(
            question=DUMMY_QUESTION,
            answer=DUMMY_ANSWER_A
        )
        print(f"Reasoning : {result_single.get('reasoning', '')[:200]}...")
        print(f"Feedback  : {result_single.get('feedback', 'N/A')}")

        print("\n" + "=" * 60)
        print("TEST 2: Single evaluation (bad answer)")
        print("=" * 60)
        result_bad = await validator.evaluate_single(
            question=DUMMY_QUESTION,
            answer=DUMMY_ANSWER_B
        )
        print(f"Reasoning : {result_bad.get('reasoning', '')[:200]}...")
        print(f"Feedback  : {result_bad.get('feedback', 'N/A')}")

        print("\n" + "=" * 60)
        print("TEST 3: Head-to-head comparison (A vs B)")
        print("=" * 60)
        result_compare = await validator.evaluate_comparison(
            question=DUMMY_QUESTION,
            answer_a=DUMMY_ANSWER_A,
            answer_b=DUMMY_ANSWER_B
        )
        print(f"Reasoning : {result_compare.get('reasoning', '')[:200]}...")
        print(f"Winner    : {result_compare.get('winner', 'N/A')}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Single (good) — verdict: {result_single.get('verdict')}, score: {result_single.get('score')}")
        print(f"Single (bad)  — verdict: {result_bad.get('verdict')},  score: {result_bad.get('score')}")
        print(f"Comparison    — verdict: {result_compare.get('verdict')}, winner: {result_compare.get('winner')}")
        print(f"Total calls in memory: {len(validator.memory)}")

    asyncio.run(main())