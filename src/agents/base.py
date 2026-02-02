from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.schemas import Prompt, AgentResponse


class PydanticAIAgent(ABC):
    """
    Abstract Base Class corresponding to the <<pydantic Interface>>
    Pydantic_AI_Agent in the UML diagram.
    """

    def __init__(
            self,
            model_id: str,
            role_name: str,
            system_instruction: Optional[str] = None,
            config: Dict[str, Any] = None
    ):
        self.model_id = model_id
        self.role_name = role_name
        self.system_instruction = system_instruction or "You are a helpful AI assistant."
        self.config = config or {}

    @abstractmethod
    async def generate(self, context: Prompt) -> AgentResponse:
        """
        Primary entry point for the agent to process a prompt.

        Args:
            context (Prompt): The input prompt and model tier.

        Returns:
            AgentResponse: The structured response containing content and arguments.
        """
        pass

    @abstractmethod
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Runtime configuration updates (e.g. changing temperature).
        """
        pass