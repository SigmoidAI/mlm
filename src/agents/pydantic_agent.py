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

from ..config.prompts import VALIDATOR_SYSTEM_PROMPT
from ..models.schemas import AgentResponse, Argument, Prompt, ValidationResult
from .base import PydanticAIAgent


def get_openrouter_api_key() -> str:
    """Get OPENROUTER_API_KEY from environment, raise if not set."""
    # Ensure dotenv is loaded before reading the key
    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
    except ImportError:
        pass
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
# INTERNAL VALIDATOR 
# ============================================================================== 
class InternalValidator:
    """
    InternalValidator: Selects the best response, determines type (creative/coding),
    generates LLM feedback with 10 parameters (5 specific, 5 common), and returns best answer and structured feedback.
    """
    def __init__(self, judge_agent):
        self.judge_agent = judge_agent

    async def validate(self, responses: List['AgentResponse'], category: str = "creative") -> Dict[str, Any]:
        """
        Validate a list of AgentResponse objects (as produced by ValidatorAgent).
        Args:
            responses: List[AgentResponse] - must be AgentResponse objects (not dicts).
            category: 'creative' or 'coding' - determines parameter set.
        Returns:
            Dict with keys: is_valid, score, feedback, refined_response.
        """
        # Input validation
        if not isinstance(responses, list) or not all(hasattr(r, 'confidence') and hasattr(r, 'content') for r in responses):
            return {
                "is_valid": False,
                "score": 0.0,
                "feedback": ["Input must be a list of AgentResponse objects from ValidatorAgent."],
                "refined_response": None
            }
        if not responses:
            return {
                "is_valid": False,
                "score": 0.0,
                "feedback": ["No responses provided."],
                "refined_response": None
            }
        best = max(responses, key=lambda x: x.confidence)
        is_valid = best.confidence >= 0.9

        common_parameters = [
            "clarity", "structure", "relevance", "engagement", "use_of_examples"
        ]
        if category == "coding":
            specific_parameters = [
                "correctness", "efficiency", "readability", "use_of_libraries", "library_freshness",
                "error_handling", "scalability", "documentation", "test_coverage", "code_style"
            ]
        else:
            specific_parameters = [
                "originality", "imagination", "emotional_impact", "storytelling", "vividness",
                "creativity", "depth", "novelty", "expressiveness", "risk_taking"
            ]
        parameters = specific_parameters + common_parameters

        analysis_prompt = (
            f"You are an internal judge. Analyze the following answer for these parameters: {', '.join(parameters)}.\n"
            f"For each parameter, provide a score from 1 to 10 (10 is best) and a brief comment.\n"
            f"Then provide an in-depth overall feedback (at least 3 sentences).\n"
            f"Return a JSON object with the following structure:\n"
            f"{{\n  'parameter_scores': {{'param': {{'score': int, 'comment': str}}, ...}},\n  'overall_feedback': str\n}}\n"
            f"Answer: {best.content}"
        )

        result = await self.judge_agent.run(analysis_prompt)
        raw = result.data if hasattr(result, 'data') else str(result.output)
        try:
            parsed = self._parse_json(raw)
        except Exception:
            parsed = {"parameter_scores": {}, "overall_feedback": raw, "improvement_suggestions": []}
        # Ensure all keys exist
        if "parameter_scores" not in parsed:
            parsed["parameter_scores"] = {}
        if "overall_feedback" not in parsed:
            parsed["overall_feedback"] = ""

        feedback = [
            f"Best answer by: {best.author_id} | Confidence: {best.confidence}",
            f"Best answer content: {best.content[:200]}",
            f"Parameter scores: {parsed['parameter_scores']}",
            f"Overall feedback: {parsed['overall_feedback']}"
        ]

        return {
            "is_valid": is_valid,
            "score": best.confidence,
            "feedback": feedback,
            "refined_response": best
        }

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        import re, json_repair
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
        if match:
            try:
                return json_repair.loads(match.group(1))
            except Exception:
                pass
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json_repair.loads(raw_text[start:end + 1])
            except Exception:
                pass
        return {"reasoning": raw_text, "verdict": "Unknown", "score": 0.0}

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

    # asyncio.run(main())

    print("\n" + "=" * 60)
    print("INTERNAL VALIDATOR TESTS")
    print("=" * 60)

    async def internal_validator_tests():
        judge_agent = Agent(
            OpenAIChatModel("deepseek/deepseek-r1", provider=OpenAIProvider(openai_client=AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=get_openrouter_api_key()))),
            system_prompt=VALIDATOR_SYSTEM_PROMPT
        )
        internal_validator = InternalValidator(judge_agent)

        # Coding responses
        coding_responses = [
            AgentResponse(author_id="CoderA", content="import numpy as np\narr = np.array([1,2,3])\nprint(arr)", confidence=0.92, arguments=[], metadata={}),
            AgentResponse(author_id="CoderB", content="import numpy\nprint(numpy.array([1,2,3]))", confidence=0.75, arguments=[], metadata={}),
        ]
        print("\nINTERNAL TEST 1: Coding (good answer)")
        result_coding = await internal_validator.validate(coding_responses, category="coding")
        print(f"is_valid: {result_coding['is_valid']}")
        print(f"score: {result_coding['score']}")
        print(f"feedback: {result_coding['feedback']}")

        # Creative responses
        creative_responses = [
            AgentResponse(author_id="CreativeA", content="A festival where everyone paints their dreams on giant canvases in the city square.", confidence=0.88, arguments=[], metadata={}),
            AgentResponse(author_id="CreativeB", content="People just eat cake.", confidence=0.6, arguments=[], metadata={}),
        ]
        print("\nINTERNAL TEST 2: Creative (good answer)")
        result_creative = await internal_validator.validate(creative_responses, category="creative")
        print(f"is_valid: {result_creative['is_valid']}")
        print(f"score: {result_creative['score']}")
        print(f"feedback: {result_creative['feedback']}")

        # Edge case: No responses
        print("\nINTERNAL TEST 3: No responses")
        result_none = await internal_validator.validate([], category="coding")
        print(f"is_valid: {result_none['is_valid']}")
        print(f"score: {result_none['score']}")
        print(f"feedback: {result_none['feedback']}")

        # Edge case: All responses invalid (low confidence)
        low_conf_coding = [
            AgentResponse(author_id="CoderA", content="import numpy as np\narr = np.array([1,2,3])\nprint(arr)", confidence=0.5, arguments=[], metadata={}),
            AgentResponse(author_id="CoderB", content="import numpy\nprint(numpy.array([1,2,3]))", confidence=0.4, arguments=[], metadata={}),
        ]
        print("\nINTERNAL TEST 4: Coding (all invalid)")
        result_low_coding = await internal_validator.validate(low_conf_coding, category="coding")
        print(f"is_valid: {result_low_coding['is_valid']}")
        print(f"score: {result_low_coding['score']}")
        print(f"feedback: {result_low_coding['feedback']}")

        low_conf_creative = [
            AgentResponse(author_id="CreativeA", content="A festival where everyone paints their dreams on giant canvases in the city square.", confidence=0.3, arguments=[], metadata={}),
            AgentResponse(author_id="CreativeB", content="People just eat cake.", confidence=0.2, arguments=[], metadata={}),
        ]
        print("\nINTERNAL TEST 5: Creative (all invalid)")
        result_low_creative = await internal_validator.validate(low_conf_creative, category="creative")
        print(f"is_valid: {result_low_creative['is_valid']}")
        print(f"score: {result_low_creative['score']}")
        print(f"feedback: {result_low_creative['feedback']}")

    asyncio.run(internal_validator_tests())
