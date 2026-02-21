import asyncio
import json
import os
import random
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import json_repair
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import sys
import os
AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(AGENTS_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(SRC_DIR, '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except ImportError:
    pass 
from config.prompts import VALIDATOR_SYSTEM_PROMPT
from models.schemas import AgentResponse, Argument, Prompt, ValidationResult
from agents.base import PydanticAIAgent


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
            cascade_tier: str = "primary",  # TODO: convert to int (Cascade levels in config are integers = [1..5])
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
        
        # TODO: Hardcoded confidence and arguments should be replaced with actual values.
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
        user_input = Prompt(content=prompt)  # TODO: WorkingAgent object may be an Agent in either Simple or Complex workflow => Prompt(..., model_tier=["simple", complex])
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
        
        # TODO: Hardcoded confidence should be replaced with actual confidence.
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
                return json_repair.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try bare JSON (find first { to last })
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json_repair.loads(raw_text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return {"reasoning": raw_text, "verdict": "Unknown", "score": 0.0}

    async def evaluate_single(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate a single response."""
        # TODO: ADD PROMPT CUSTOM
        prompt = (
            f"[User Prompt]\n{question}\n\n"
            f"[Model Response]\n{answer}"
        )

        print(f"🔍 Evaluating single response...")
        result = await self.judge_agent.run(prompt)
        raw = result.data if hasattr(result, 'data') else str(result.output)
        parsed = self._parse_json(raw)
        
        if not isinstance(parsed, dict):
            print(f"Warning: parsed result is not a dict, got {type(parsed)}")
            parsed = {"reasoning": str(parsed), "verdict": "Unknown", "score": 0.0}
    
        print(f"📊 Verdict: {parsed.get('verdict', 'Unknown')}")
        print(f"📈 Score: {parsed.get('score', 'N/A')}")

        self.memory.append({"type": "single_evaluation", "question": question, **parsed})
        return parsed

    # TODO: TEMPORARY NEW METHOD, UNDER DISCUSSION
    async def evaluate_multiple(self, prompt: Prompt, question: str, answers: dict[str, AgentResponse]) -> dict[str, Any]:
        """Evaluate multiple answers from worker agents."""
        print(f"Evaluating {len(answers.keys())} responses")
        
        result = await self.judge_agent.run(prompt)
        
        raw = result.data if hasattr(result, 'data') else str(result.output)
        parsed = self._parse_json(raw)
        
        if not isinstance(parsed, dict):
            print(f"Warning: parsed result is not a dict, got {type(parsed)}")
            parsed = {"reasoning": str(parsed), "verdict": "Unknown", "score": 0.0}
        
        self.memory.append({"type": "multiple_evaluation", "question": question, **parsed})
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

        self.memory.append({"type": "comparison_evaluation", "question": question, **parsed})
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
# INTERN VALIDATOR AGENT
# ============================================================================== 
class InternValidatorAgent(ValidatorAgent):
    """
    InternValidatorAgent: Provides more detailed feedback for final answers and uses separate prompts for simple and creative tasks.
    """
    SIMPLE_PROMPT = (
        "[User Prompt]\n{question}\n\n"
        "[Model Response]\n{answer}\n\n"
        "You are an internal judge. Please provide a detailed evaluation focusing on factual accuracy, clarity, and completeness. "
        "Give specific, actionable feedback for improvement if needed."
    )
    CREATIVE_PROMPT = (
        "[User Prompt]\n{question}\n\n"
        "[Model Response]\n{answer}\n\n"
        "You are an internal judge. Please provide a detailed evaluation focusing on originality, depth, and creativity. "
        "Give specific, actionable feedback for improvement, highlighting strengths and weaknesses."
    )

    async def evaluate_simple(self, question: str, answer: str) -> Dict[str, Any]:
        prompt = self.SIMPLE_PROMPT.format(question=question, answer=answer)
        result = await self.judge_agent.run(prompt)
        raw = result.data if hasattr(result, 'data') else str(result.output)
        parsed = self._parse_json(raw)
        feedback = self._generate_simple_feedback(parsed, question, answer)
        parsed['feedback'] = feedback
        self.memory.append({"type": "simple_evaluation", "question": question, **parsed})
        return parsed

    async def evaluate_creative(self, question: str, answer: str) -> Dict[str, Any]:
        prompt = self.CREATIVE_PROMPT.format(question=question, answer=answer)
        result = await self.judge_agent.run(prompt)
        raw = result.data if hasattr(result, 'data') else str(result.output)
        parsed = self._parse_json(raw)
        feedback = self._generate_creative_feedback(parsed, question, answer)
        parsed['feedback'] = feedback
        self.memory.append({"type": "creative_evaluation", "question": question, **parsed})
        return parsed

    def _generate_simple_feedback(self, parsed: Dict[str, Any], question: str, answer: str) -> list:
        feedback = []
        if parsed.get('verdict') == 'Valid':
            feedback.append(
                "Excellent factual accuracy and clarity: The answer directly addresses the question, is precise, and covers all required details. Well-structured and easy to follow."
            )
        else:
            feedback.append(
                "Improvements needed: The answer is missing key facts, lacks clarity, or does not fully address the question. Please ensure all relevant information is included and presented in a logical, concise manner."
            )
        if 'reasoning' in parsed:
            feedback.append(f"Evaluation reasoning: {parsed['reasoning'][:200]}")
        if 'score' in parsed:
            feedback.append(f"Evaluation score: {parsed['score']}")
        feedback.append(
            "Actionable tip: Use concrete examples, cite authoritative sources if possible, and double-check for completeness and accuracy."
        )
        return feedback

    def _generate_creative_feedback(self, parsed: Dict[str, Any], question: str, answer: str) -> list:
        feedback = []
        if parsed.get('verdict') == 'Valid':
            feedback.append(
                "Outstanding creativity and depth: The answer is imaginative, offers fresh perspectives, and explores the topic in a thoughtful, engaging way. Strong use of vivid details or storytelling."
            )
        else:
            feedback.append(
                "Needs more creative development: The answer is too generic or lacks originality. Try to approach the topic from a new angle, add richer details, or use analogies and narrative elements to make it more compelling."
            )
        if 'reasoning' in parsed:
            feedback.append(f"Evaluation reasoning: {parsed['reasoning'][:200]}")
        if 'score' in parsed:
            feedback.append(f"Evaluation score: {parsed['score']}")
        feedback.append(
            "Actionable tip: Experiment with unique ideas, draw connections to broader themes, and use descriptive language or examples to enhance engagement."
        )
        return feedback

    async def evaluate(self, question: str, answer: str, creative: bool = False) -> Dict[str, Any]:
        if creative:
            return await self.evaluate_creative(question, answer)
        else:
            return await self.evaluate_simple(question, answer)

    async def validate(self, responses: List[AgentResponse], creative: bool = False) -> ValidationResult:
        avg_conf = sum(r.confidence for r in responses) / len(responses) if responses else 0.0
        is_valid = avg_conf >= self.threshold
        feedback = []
        if creative:
            if is_valid:
                feedback.append(
                    "Creative validation: The responses provided are highly original, demonstrate significant depth of thought, and show strong creativity. The answers go beyond the obvious, offering unique perspectives and well-developed ideas."
                )
            else:
                feedback.append(
                    "Creative validation: The responses lack sufficient originality or depth. To improve, encourage more imaginative thinking, provide richer details, and explore novel or unconventional ideas. Consider using analogies, storytelling, or vivid examples to make the answers stand out."
                )
        else:
            if is_valid:
                feedback.append(
                    "Simple validation: The responses are factually accurate, clear, and address all aspects of the question. The answers are concise, well-structured, and easy to understand. Good use of relevant facts and logical reasoning."
                )
            else:
                feedback.append(
                    "Simple validation: The responses need improvement in factual accuracy, clarity, or completeness. To enhance the answers, ensure all key facts are correct, address every part of the question, and organize the information logically. Use specific examples or references where possible."
                )
        # Add detailed feedback for final answers
        if responses:
            best = max(responses, key=lambda x: x.confidence)
            feedback.append(f"Best answer by: {best.author_id} | Confidence: {best.confidence}")
            feedback.append(f"Best answer content: {best.content[:200]}")
        return ValidationResult(
            is_valid=is_valid,
            score=avg_conf,
            feedback=feedback,
            refined_response=responses[0] if responses else None
        )


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

    SIMPLE_QUESTION = "What is the capital of France?"
    SIMPLE_ANSWER_GOOD = "The capital of France is Paris."
    SIMPLE_ANSWER_BAD = "France is a country in Europe."

    CREATIVE_QUESTION = "Invent a new holiday and describe how people celebrate it."
    CREATIVE_ANSWER_GOOD = (
        "Dreamer's Day: On this holiday, people write down their wildest dreams and share them with friends. "
        "Communities organize parades where everyone wears costumes representing their dreams. "
        "At night, lanterns are released into the sky, symbolizing hopes taking flight."
    )
    CREATIVE_ANSWER_BAD = "People just stay home and do nothing."

    # async def main():
    #     validator = ValidatorAgent(
    #         model_name="deepseek/deepseek-r1",
    #         api_key=get_openrouter_api_key()
    #     )
    #
    #     print("=" * 60)
    #     print("TEST 1: Single evaluation (good answer)")
    #     print("=" * 60)
    #     result_single = await validator.evaluate_single(
    #         question=DUMMY_QUESTION,
    #         answer=DUMMY_ANSWER_A
    #     )
    #     print(f"Reasoning : {result_single.get('reasoning', '')[:200]}...")
    #     print(f"Feedback  : {result_single.get('feedback', 'N/A')}")
    #
    #     print("\n" + "=" * 60)
    #     print("TEST 2: Single evaluation (bad answer)")
    #     print("=" * 60)
    #     result_bad = await validator.evaluate_single(
    #         question=DUMMY_QUESTION,
    #         answer=DUMMY_ANSWER_B
    #     )
    #     print(f"Reasoning : {result_bad.get('reasoning', '')[:200]}...")
    #     print(f"Feedback  : {result_bad.get('feedback', 'N/A')}")
    #
    #     print("\n" + "=" * 60)
    #     print("TEST 3: Head-to-head comparison (A vs B)")
    #     print("=" * 60)
    #     result_compare = await validator.evaluate_comparison(
    #         question=DUMMY_QUESTION,
    #         answer_a=DUMMY_ANSWER_A,
    #         answer_b=DUMMY_ANSWER_B
    #     )
    #     print(f"Reasoning : {result_compare.get('reasoning', '')[:200]}...")
    #     print(f"Winner    : {result_compare.get('winner', 'N/A')}")
    #
    #     print("\n" + "=" * 60)
    #     print("SUMMARY")
    #     print("=" * 60)
    #     print(f"Single (good) — verdict: {result_single.get('verdict')}, score: {result_single.get('score')}")
    #     print(f"Single (bad)  — verdict: {result_bad.get('verdict')},  score: {result_bad.get('score')}")
    #     print(f"Comparison    — verdict: {result_compare.get('verdict')}, winner: {result_compare.get('winner')}")
    #     print(f"Total calls in memory: {len(validator.memory)}")
    #
    # asyncio.run(main())

    async def intern_main():
        intern_judge = InternValidatorAgent(
            model_name="deepseek/deepseek-r1",
            api_key=get_openrouter_api_key()
        )

        print("=" * 60)
        print("INTERN TEST 1: Simple evaluation (good answer)")
        print("=" * 60)
        result_simple = await intern_judge.evaluate(
            question=SIMPLE_QUESTION,
            answer=SIMPLE_ANSWER_GOOD,
            creative=False
        )
        print(f"Reasoning : {result_simple.get('reasoning', '')[:200]}...")
        print(f"Feedback  : {result_simple.get('feedback', 'N/A')}")

        print("\n" + "=" * 60)
        print("INTERN TEST 2: Creative evaluation (good answer)")
        print("=" * 60)
        result_creative = await intern_judge.evaluate(
            question=CREATIVE_QUESTION,
            answer=CREATIVE_ANSWER_GOOD,
            creative=True
        )
        print(f"Reasoning : {result_creative.get('reasoning', '')[:200]}...")
        print(f"Feedback  : {result_creative.get('feedback', 'N/A')}")

        print("\n" + "=" * 60)
        print("INTERN TEST 3: Simple evaluation (bad answer)")
        print("=" * 60)
        result_simple_bad = await intern_judge.evaluate(
            question=SIMPLE_QUESTION,
            answer=SIMPLE_ANSWER_BAD,
            creative=False
        )
        print(f"Reasoning : {result_simple_bad.get('reasoning', '')[:200]}...")
        print(f"Feedback  : {result_simple_bad.get('feedback', 'N/A')}")

        print("\n" + "=" * 60)
        print("INTERN TEST 4: Creative evaluation (bad answer)")
        print("=" * 60)
        result_creative_bad = await intern_judge.evaluate(
            question=CREATIVE_QUESTION,
            answer=CREATIVE_ANSWER_BAD,
            creative=True
        )
        print(f"Reasoning : {result_creative_bad.get('reasoning', '')[:200]}...")
        print(f"Feedback  : {result_creative_bad.get('feedback', 'N/A')}")

        # Test validate() method with synthetic AgentResponse objects
        print("\n" + "=" * 60)
        print("INTERN TEST 5: validate() method (simple)")
        print("=" * 60)
        from models.schemas import AgentResponse
        responses_simple = [
            AgentResponse(author_id="A", content="Paris", confidence=0.9, arguments=[], metadata={}),
            AgentResponse(author_id="B", content="Lyon", confidence=0.7, arguments=[], metadata={}),
        ]
        val_result_simple = await intern_judge.validate(responses_simple, creative=False)
        print(f"is_valid: {val_result_simple.is_valid}")
        print(f"score: {val_result_simple.score}")
        print(f"feedback: {val_result_simple.feedback}")

        print("\n" + "=" * 60)
        print("INTERN TEST 6: validate() method (creative)")
        print("=" * 60)
        responses_creative = [
            AgentResponse(author_id="A", content="Dreamer's Day", confidence=0.95, arguments=[], metadata={}),
            AgentResponse(author_id="B", content="Stay Home Day", confidence=0.6, arguments=[], metadata={}),
        ]
        val_result_creative = await intern_judge.validate(responses_creative, creative=True)
        print(f"is_valid: {val_result_creative.is_valid}")
        print(f"score: {val_result_creative.score}")
        print(f"feedback: {val_result_creative.feedback}")

        print("\n" + "=" * 60)
        print("INTERN SUMMARY")
        print("=" * 60)
        print(f"Simple (good)    — verdict: {result_simple.get('verdict')}, score: {result_simple.get('score')}")
        print(f"Creative (good)  — verdict: {result_creative.get('verdict')}, score: {result_creative.get('score')}")
        print(f"Simple (bad)     — verdict: {result_simple_bad.get('verdict')}, score: {result_simple_bad.get('score')}")
        print(f"Creative (bad)   — verdict: {result_creative_bad.get('verdict')}, score: {result_creative_bad.get('score')}")
        print(f"Total calls in memory: {len(intern_judge.memory)}")

    asyncio.run(intern_main())