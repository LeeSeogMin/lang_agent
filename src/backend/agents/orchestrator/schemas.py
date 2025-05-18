"""Core schemas for agent system."""

from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from pydantic import BaseModel, Field


# Agent State
class AgentState(AgentStatePydantic):
    """Base state for all agents."""

    # Common state
    summary: Optional[str] = None
    routing_decision: Optional[str] = None

    # Agent-specific state - only one will be populated based on routing
    knowledge_findings: Optional[Dict[str, Any]] = None
    summarizer_response: Optional[Union[Dict[str, Any], Any]] = None
    document_content: Optional[str] = None
    data_analysis_response: Optional[Any] = None
    uploaded_data_file: Optional[str] = None
    
    # 데이터 분석 관련 필드
    apply_rag_filtering: Optional[bool] = None
    filter_query: Optional[str] = None
    apply_code_generation: Optional[bool] = None
    code_generation_task: Optional[str] = None
    code_type: Optional[str] = None
    input_data_content: Optional[str] = None


# Node Return Types


class SummaryReturn(BaseModel):
    """Return type for summary node."""

    summary: str
    messages: List[BaseMessage] = Field(default_factory=list)


class AnswerReturn(BaseModel):
    """Return type for answer node."""

    messages: List[BaseMessage] = Field(default_factory=list)
