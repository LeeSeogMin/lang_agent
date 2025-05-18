"""Schemas for the data analysis agent."""

from typing import Annotated, Any, Dict, List, Optional

import operator

from pydantic import BaseModel, Field

from backend.agents.orchestrator.schemas import AgentState


class AnalysisApproachRecommendation(BaseModel):
    """Structured output for analysis approach recommendation."""

    analysis_methods: List[str] = Field(
        description="Recommended analytical methods to apply", min_items=1
    )
    visualization_types: List[str] = Field(
        description="Recommended visualization types", min_items=1
    )
    reasoning: str = Field(description="Brief explanation for the recommendations")


# State for processing individual data segments
class DataSegmentState(BaseModel):
    """State for processing individual data segments."""

    segment: str
    segment_id: int


# Main state for the data analysis agent
class DataAnalysisState(AgentState):
    """Main state for the data analysis agent."""

    input_data_content: Optional[str] = Field(
        default=None,
        description="Pre-processed content of the data to be analyzed",
    )
    data: str = Field(
        default="",
        description="Original data processed by the node (can be same as input_data_content or derived)",
    )
    segments: List[str] = Field(
        default_factory=list, description="Data split into segments for analysis"
    )
    analysis_results: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="Individual segment analysis results (uses add reducer)",
    )
    final_analysis: Optional[str] = Field(
        default=None, description="Final combined analysis"
    )
    uploaded_data_file: Optional[str] = Field(
        default=None, description="Path to uploaded data file if any"
    )
    
    # RAG 필터링 관련 필드
    apply_rag_filtering: bool = Field(
        default=False, 
        description="RAG 기반 데이터 필터링을 적용할지 여부"
    )
    filter_query: str = Field(
        default="중요한 정보 추출", 
        description="데이터 필터링을 위한 자연어 쿼리"
    )
    filtered_data_result: Optional[str] = Field(
        default=None, 
        description="RAG 필터링 결과"
    )
    filtered_data_success: bool = Field(
        default=False,
        description="RAG 필터링 성공 여부"
    )
    filter_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="필터링 관련 메타데이터"
    )
    
    # 코드 생성 관련 필드
    apply_code_generation: bool = Field(
        default=False,
        description="LLM 기반 코드 생성을 적용할지 여부"
    )
    code_generation_task: str = Field(
        default="데이터 분석 및 시각화",
        description="코드 생성을 위한 작업 설명"
    )
    code_type: str = Field(
        default="pandas",
        description="생성할 코드 유형(pandas, numpy, matplotlib)"
    )
    generated_code_result: Optional[str] = Field(
        default=None,
        description="코드 생성 결과"
    )
    generated_code_success: bool = Field(
        default=False,
        description="코드 생성 성공 여부"
    )
    generated_code: Optional[str] = Field(
        default=None,
        description="생성된 Python 코드"
    )


# Schema for the output of process_data_node (partial update for DataAnalysisState)
class ProcessDataNodeOutput(BaseModel):
    """Defines the structure of the dictionary returned by process_data_node."""

    data: Optional[str] = None
    segments: Optional[List[str]] = None
    analysis_results: Optional[List[str]] = None
    final_analysis: Optional[str] = None


# Response from the data analysis agent
class DataAnalysisResponse(BaseModel):
    """Response from the data analysis agent."""

    segment_analyses: List[str] = Field(
        default_factory=list
    )  # List of individual analyses
    formatted_segment_analyses: str = Field(
        default="",
        description="Individual segment analyses formatted as a single string",
    )
    num_segments: int = Field(default=0)
    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )  # Additional info like processing time, data specifics, etc.
    
    class Config:
        """Configuration for the DataAnalysisResponse model."""
        arbitrary_types_allowed = True


# Output state for the data analysis agent
class DataAnalysisOutput(BaseModel):
    """Output state for the data analysis agent."""

    data_analysis_response: DataAnalysisResponse

    def dict(self, *args, **kwargs):
        """Convert the model to a dictionary."""
        result = super().dict(*args, **kwargs)
        # 직접 data_analysis_response 속성을 가진다면 추가
        if hasattr(self, "data_analysis_response"):
            result["data_analysis_response"] = self.data_analysis_response
        return result
    
    def model_dump(self, *args, **kwargs):
        """Convert the model to a dictionary (newer pydantic version)."""
        if hasattr(super(), "model_dump"):
            result = super().model_dump(*args, **kwargs)
        else:
            result = self.dict(*args, **kwargs)
        
        # 직접 data_analysis_response 속성을 가진다면 추가
        if hasattr(self, "data_analysis_response"):
            result["data_analysis_response"] = self.data_analysis_response
        return result
    
    def get(self, key, default=None):
        """Dictionary-like get method for compatibility."""
        if key == "data_analysis_response" and hasattr(self, "data_analysis_response"):
            return self.data_analysis_response
        return getattr(self, key, default) 