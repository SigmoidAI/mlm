"""
Data Models for Multi-Agent System
Based on UML Architecture Diagram.

This module defines the Data Transfer Objects (DTOs) used to pass messages
between the WorkingAgent, ValidatorAgent, and the orchestration layer.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# -------------------------------------------------------------------------
# Base Configuration
# -------------------------------------------------------------------------

class PydanticAIBaseModel(BaseModel):
    """
    Base class for all models in the system, corresponding to
    <<pydantic Interface>> PydanticAIBaseModel in the diagram.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="ignore"
    )

# -------------------------------------------------------------------------
# Core Data Structures
# -------------------------------------------------------------------------

class Prompt(PydanticAIBaseModel):
    """
    Input structure corresponding to the <<custom>> Prompt class.

    Attributes:
        content: The actual text prompt to be processed.
        model_tier: Strategy for model selection (e.g., 'standard', 'premium').
    """
    content: str = Field(..., description="The user input or system instruction")
    model_tier: Literal["simple", "complex"] = Field(
        default="simple",
        description="The tier of LLM to use for this prompt"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Argument(PydanticAIBaseModel):
    """
    Structured reasoning unit corresponding to the <<custom>> Argument class.

    This is used inside AgentResponse to break down chain-of-thought.
    """
    claim: str = Field(..., description="The specific point being argued")
    reasoning: str = Field(..., description="The logic or evidence supporting the claim")
    verdict: Literal["Valid", "Weak", "Fallacy"] = Field(
        ...,
        description="Assessment of the claim's strength"
    )


class AgentResponse(PydanticAIBaseModel):
    """
    Standardized output corresponding to <<custom>> AgentResponse.

    This represents the payload passed between agents in the graph.
    """
    author_id: str = Field(..., description="ID of the agent (WorkingAgent or Validator)")
    content: str = Field(..., description="The main textual response/solution")

    # Composition relationship as shown in diagram (Diamond shape)
    arguments: List[Argument] = Field(
        default_factory=list,
        description="Structured breakdown of reasoning steps"
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Self-evaluated confidence score (0.0 to 1.0)"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context (latency, token usage, model version)"
    )

# -------------------------------------------------------------------------
# Operational / Config Schemas
# -------------------------------------------------------------------------

class ValidationResult(PydanticAIBaseModel):
    """
    Return type for ValidatorAgent.validate() operations.
    """
    is_valid: bool
    score: float = Field(..., ge=0.0, le=1.0)
    feedback: List[str] = Field(default_factory=list)
    refined_response: Optional[AgentResponse] = None


class AgentConfig(PydanticAIBaseModel):
    """
    Configuration schema to type-check the 'config: dict' attribute
    shown in the Pydantic_AI_Agent interface.
    """
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 1000
    system_instruction: Optional[str] = None

    # Specific fields from your diagram's subclasses can be optional here
    cascade_tier: Optional[str] = None      # For WorkingAgent
    threshold: Optional[float] = 0.8        # For ValidatorAgent