"""Data analysis agent graph definition."""

from typing import Any, Dict, List, Optional
import os

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.agents.data_analysis.schemas import (
    AnalysisApproachRecommendation,
    DataSegmentState,
    ProcessDataNodeOutput,
    DataAnalysisOutput,
    DataAnalysisState,
)
from backend.agents.data_analysis.prompts import (
    DATA_APPROACH_PROMPT,
    DATA_ANALYSIS_PROMPT,
)
from backend.agents.data_analysis.tools import (
    segment_data,
    validate_data,
    analyze_data_statistics,
    format_results,
    filter_data_rag,
    generate_analysis_code,
)
from backend.config import settings
from backend.utils.file_utils import count_tokens

# Initialize model using shared configuration
model = settings.get_model()


# Node functions
async def analyze_data_approach(data: str) -> Dict[str, Any]:
    """Analyze data structure and recommend analytical approach."""
    # Get data preview and stats
    preview = data[:500]
    total_tokens = await count_tokens(data)

    # Get recommendation from LLM using function calling
    recommendation = await model.with_structured_output(
        AnalysisApproachRecommendation, method="function_calling"
    ).ainvoke(
        [
            SystemMessage(
                content=DATA_APPROACH_PROMPT.format(
                    data_preview=preview,
                    metadata={
                        "total_tokens": total_tokens,
                        "total_length": len(data),
                        "has_numeric": any(c.isdigit() for c in data),
                        "has_tabular_indicators": "\t" in data or "," in data,
                    },
                )
            )
        ]
    )

    return {
        "analysis_methods": recommendation.analysis_methods,
        "visualization_types": recommendation.visualization_types,
        "segment_size": 500,  # Default segment size
        "segment_overlap": 50,  # Default segment overlap
    }


async def process_data_node(state: DataAnalysisState) -> Dict[str, Any]:
    """
    Node to process and segment input data.
    It expects the data text to be already populated in the state.
    """
    # 데이터 소스 확인 (여러 소스에서 데이터를 받을 수 있음)
    data_content_from_orchestrator = getattr(state, "document_content", None)
    data_content_from_input = getattr(state, "input_data_content", None)
    data_content_from_file = None
    
    # 파일 경로가 있으면 파일에서 데이터 읽기
    uploaded_file_path = getattr(state, "uploaded_data_file", None)
    print(f"[DEBUG] process_data_node: uploaded_data_file = {uploaded_file_path}")
    
    if uploaded_file_path and os.path.exists(uploaded_file_path):
        try:
            with open(uploaded_file_path, "r", encoding="utf-8") as f:
                data_content_from_file = f.read()
            print(f"[DEBUG] Data loaded from file: {uploaded_file_path}, length: {len(data_content_from_file)} chars")
        except Exception as e:
            print(f"[DEBUG] Error reading file {uploaded_file_path}: {str(e)}")
    elif uploaded_file_path:
        print(f"[DEBUG] Uploaded file path exists in state but file not found: {uploaded_file_path}")
    
    data_to_process: Optional[str] = None
    error_source_field = ""

    # 여러 소스 중 우선순위에 따라 데이터 선택
    if data_content_from_orchestrator and data_content_from_orchestrator.strip():
        data_to_process = data_content_from_orchestrator
        error_source_field = "document_content (from orchestrator)"
        print(f"[DEBUG] Using data from orchestrator: {len(data_to_process)} chars")
    elif data_content_from_input and data_content_from_input.strip():
        data_to_process = data_content_from_input
        error_source_field = "input_data_content (direct input)"
        print(f"[DEBUG] Using data from direct input: {len(data_to_process)} chars")
    elif data_content_from_file and data_content_from_file.strip():
        data_to_process = data_content_from_file
        error_source_field = f"uploaded_data_file ({uploaded_file_path})"
        print(f"[DEBUG] Using data from file: {len(data_to_process)} chars")

    if not data_to_process:
        final_analysis_update = (
            "Error: No data content found. Checked state fields "
            "'document_content', 'input_data_content', and 'uploaded_data_file'. "
            "Content must be provided in one of these fields."
        )
        print(f"[DEBUG] No data to process found. State fields: document_content={bool(data_content_from_orchestrator)}, "
              f"input_data_content={bool(data_content_from_input)}, "
              f"data_content_from_file={bool(data_content_from_file)}")
        return ProcessDataNodeOutput(
            data="",
            segments=[],
            analysis_results=[],
            final_analysis=final_analysis_update,
        ).model_dump(exclude_none=True)

    try:
        # 데이터 유효성 검사
        print(f"[DEBUG] Validating data...")
        validation_result = await validate_data.ainvoke({"data": data_to_process})
        
        if not validation_result.get("is_valid", True):
            # 데이터 유효성 검사 실패
            error_message = validation_result.get("error", "Data validation failed without specific error.")
            print(f"[DEBUG] Data validation failed: {error_message}")
            return ProcessDataNodeOutput(
                data=data_to_process,
                segments=[],
                analysis_results=[],
                final_analysis=f"데이터 유효성 검사 오류 (소스: {error_source_field}): {error_message}",
            ).model_dump(exclude_none=True)
        
        # 데이터 통계 분석
        print(f"[DEBUG] Analyzing data statistics...")
        stats_result = await analyze_data_statistics.ainvoke({
            "data": data_to_process,
            "analysis_type": "general"
        })
        
        # 데이터 특성에 기반한 분석 접근 방식 결정
        print(f"[DEBUG] Determining analysis approach...")
        approach_settings = await analyze_data_approach(data_to_process)
        
        # 유효성 검사에서 감지된 형식을 접근 방식 설정에 추가
        if "detected_format" in validation_result:
            approach_settings["detected_format"] = validation_result["detected_format"]
            print(f"[DEBUG] Detected format: {validation_result['detected_format']}")
        
        # 세그먼트로 분할하여 상세 분석 
        print(f"[DEBUG] Segmenting data...")
        segment_result = await segment_data.ainvoke(
            {
                "data": data_to_process,
                "segment_size": approach_settings.get("segment_size", 500),
                "segment_overlap": approach_settings.get("segment_overlap", 50),
            }
        )
        
        if "error" in segment_result or not segment_result.get("segments"):
            error_message = segment_result.get(
                "error", "No segments produced or an unspecified error occurred."
            )
            print(f"[DEBUG] Segmentation error: {error_message}")
            return ProcessDataNodeOutput(
                data=data_to_process,
                segments=[],
                analysis_results=[],
                final_analysis=f"데이터 세그먼트 분할 오류 (소스: {error_source_field}): {error_message}",
            ).model_dump(exclude_none=True)
        
        # 통계적 분석 결과와 유효성 검사 결과를 analysis_results에 저장
        print(f"[DEBUG] Creating initial analysis...")
        analysis_results = [
            f"## 데이터 유효성 검사 결과\n" + 
            f"파일 형식: {validation_result.get('detected_format', '알 수 없음')}\n" +
            f"데이터 길이: {validation_result.get('data_length', 0)} 문자\n" +
            f"미리보기: {validation_result.get('data_preview', '')}\n",
            
            f"## 통계 분석\n" +
            f"총 길이: {stats_result.get('statistics', {}).get('length', 0)} 문자\n" +
            f"토큰 수: {stats_result.get('statistics', {}).get('tokens', 0)}\n" +
            f"라인 수: {stats_result.get('statistics', {}).get('lines', 0)}\n" +
            (f"단어 수: {stats_result.get('statistics', {}).get('word_count', 0)}\n" if 'word_count' in stats_result.get('statistics', {}) else "") +
            (f"고유 단어 수: {stats_result.get('statistics', {}).get('unique_words', 0)}\n" if 'unique_words' in stats_result.get('statistics', {}) else "")
        ]
        
        # 분석 방법 정보 추가
        if approach_settings.get("analysis_methods"):
            method_info = "## 분석 방법\n" + "\n".join([f"- {method}" for method in approach_settings["analysis_methods"]])
            analysis_results.append(method_info)
        
        # 시각화 추천 정보 추가
        if approach_settings.get("visualization_types"):
            vis_info = "## 추천 시각화\n" + "\n".join([f"- {vis}" for vis in approach_settings["visualization_types"]])
            analysis_results.append(vis_info)

        print(f"[DEBUG] Process data complete. Found {len(segment_result['segments'])} segments.")
        return ProcessDataNodeOutput(
            data=data_to_process,
            segments=segment_result["segments"],
            analysis_results=analysis_results,  # 초기 분석 포함
            final_analysis=None,
        ).model_dump(exclude_none=True)

    except Exception as e:
        print(f"[DEBUG] Error in data processing: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return ProcessDataNodeOutput(
            data=data_to_process,
            segments=[],
            analysis_results=[],
            final_analysis=f"데이터 분석 파이프라인 오류: {str(e)}",
        ).model_dump(exclude_none=True)


async def analyze_segment(state: DataSegmentState) -> Dict[str, List[str]]:
    """Node to analyze an individual data segment."""
    segment = state.segment
    segment_id = state.segment_id

    # Perform content analysis using LLM
    response = await model.ainvoke(
        [
            SystemMessage(
                content=DATA_ANALYSIS_PROMPT.format(
                    data=segment,
                    context=f"This is segment {segment_id} of the data. Focus on identifying patterns and insights within this segment.",
                )
            )
        ]
    )
    
    # For text segments, also get statistical insights
    try:
        stats_result = await analyze_data_statistics.ainvoke({
            "data": segment,
            "analysis_type": "text" if len(segment) > 0 and segment[0].isalpha() else "numerical"
        })
        
        # Add statistical insights to the analysis
        stats_summary = ""
        if "statistics" in stats_result:
            stats = stats_result["statistics"]
            stats_summary = "\n\n### Statistical Insights:\n"
            
            if "word_count" in stats:
                stats_summary += f"- Words: {stats.get('word_count', 0)}\n"
                stats_summary += f"- Unique Words: {stats.get('unique_words', 0)}\n"
                stats_summary += f"- Avg Word Length: {stats.get('avg_word_length', 0):.2f}\n"
            
            if "numeric_count" in stats:
                stats_summary += f"- Numeric Values: {stats.get('numeric_count', 0)}\n"
                if stats.get("numeric_count", 0) > 0:
                    stats_summary += f"- Range: {stats.get('min', 0)} to {stats.get('max', 0)}\n"
                    stats_summary += f"- Mean: {stats.get('mean', 0):.2f}\n"
    except Exception as e:
        stats_summary = f"\n\nError getting segment statistics: {str(e)}"
    
    # Combine LLM analysis with statistical insights
    full_analysis = f"[Segment {segment_id}] {response.content}{stats_summary}"
    
    return {"analysis_results": [full_analysis]}


async def filter_data_with_rag(state: DataAnalysisState) -> Dict[str, Any]:
    """RAG 기반 데이터 필터링을 처리하는 노드."""
    data = state.data if hasattr(state, "data") else ""
    query = getattr(state, "filter_query", "중요한 정보 추출")
    
    if not data:
        return {
            "filtered_data_result": "필터링할 데이터가 없습니다. 먼저 데이터를 처리해야 합니다.",
            "filtered_data_success": False
        }
    
    try:
        print(f"[DEBUG] Applying RAG-based filtering with query: {query}")
        # 데이터 형식 자동 감지
        validation_result = await validate_data.ainvoke({"data": data})
        detected_format = validation_result.get("detected_format", "text")
        
        # RAG 기반 필터링 실행
        filter_result = await filter_data_rag.ainvoke({
            "data": data,
            "query": query,
            "data_format": detected_format
        })
        
        if "error" in filter_result:
            print(f"[DEBUG] Filtering error: {filter_result['error']}")
            return {
                "filtered_data_result": f"필터링 중 오류 발생: {filter_result['error']}",
                "filtered_data_success": False
            }
        
        filtered_data = filter_result.get("filtered_result", "필터링 결과가 없습니다.")
        metadata = filter_result.get("metadata", {})
        
        # 결과 포맷팅
        formatted_result = f"""## RAG 기반 데이터 필터링 결과
        
### 검색 쿼리
{query}

### 필터링 결과
{filtered_data}

### 메타데이터
- 데이터 형식: {metadata.get('detected_format', detected_format)}
- 처리된 청크: {metadata.get('chunks_processed', 'N/A')}
- 원본 행: {metadata.get('original_rows', 'N/A')}
- 원본 열: {metadata.get('original_columns', 'N/A')}
"""
        
        return {
            "filtered_data_result": formatted_result,
            "filtered_data_success": True,
            "filter_metadata": metadata
        }
        
    except Exception as e:
        print(f"[DEBUG] Error in RAG filtering: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return {
            "filtered_data_result": f"RAG 필터링 중 오류 발생: {str(e)}",
            "filtered_data_success": False
        }


async def generate_analysis_code_node(state: DataAnalysisState) -> Dict[str, Any]:
    """LLM 기반 분석 코드 생성을 처리하는 노드."""
    data = state.data if hasattr(state, "data") else ""
    task = getattr(state, "code_generation_task", "데이터 분석 및 시각화")
    code_type = getattr(state, "code_type", "pandas")
    
    if not data:
        return {
            "generated_code_result": "코드를 생성할 데이터가 없습니다. 먼저 데이터를 처리해야 합니다.",
            "generated_code_success": False
        }
    
    try:
        print(f"[DEBUG] Generating analysis code for task: {task}")
        # 데이터 샘플 준비 (전체 데이터가 너무 클 수 있으므로)
        data_sample = data[:2000] if len(data) > 2000 else data
        
        # 코드 생성 실행
        code_result = await generate_analysis_code.ainvoke({
            "data": data_sample,
            "task": task,
            "code_type": code_type
        })
        
        if "error" in code_result:
            print(f"[DEBUG] Code generation error: {code_result['error']}")
            return {
                "generated_code_result": f"코드 생성 중 오류 발생: {code_result['error']}",
                "generated_code_success": False
            }
        
        generated_code = code_result.get("generated_code", "코드 생성에 실패했습니다.")
        
        # 결과 포맷팅
        formatted_result = f"""## 분석 코드 생성 결과
        
### 작업 설명
{task}

### 생성된 코드
```python
{generated_code}
```

### 실행 정보
- 코드 유형: {code_result.get('code_type', code_type)}
- 데이터 형식: {code_result.get('detected_format', 'text')}
- 상태: {code_result.get('execution_info', '코드가 생성되었습니다.')}
"""
        
        return {
            "generated_code_result": formatted_result,
            "generated_code_success": True,
            "generated_code": generated_code
        }
        
    except Exception as e:
        print(f"[DEBUG] Error in code generation: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return {
            "generated_code_result": f"코드 생성 중 오류 발생: {str(e)}",
            "generated_code_success": False
        }


async def combine_analyses(state: DataAnalysisState) -> DataAnalysisOutput:
    """Node to combine all segment analyses into a single formatted result."""
    analyses = state.analysis_results if hasattr(state, "analysis_results") else []

    # Format the individual segment analyses using the format_results tool
    try:
        formatted_result = await format_results.ainvoke({
            "results": analyses,
            "format_type": "text"  # Can be adjusted based on preferences
        })
        
        formatted_analyses_string = formatted_result.get("formatted_result", "")
        if not formatted_analyses_string:
            formatted_analyses_string = "\n\n".join(analyses) if analyses else "No analyses generated."
    except Exception as e:
        # Fallback to simple joining if formatting fails
        formatted_analyses_string = "\n\n".join(analyses) if analyses else f"No analyses generated. Error: {str(e)}"

    # Add a summary section synthesizing the key findings
    try:
        if analyses and len("\n".join(analyses)) > 0:
            summary_prompt = (
                "Based on the following analysis results, provide a concise summary "
                "of the key findings and insights:\n\n" + "\n\n".join(analyses[:5])  # Limit to first 5 for context length
            )
            
            summary_response = await model.ainvoke([SystemMessage(content=summary_prompt)])
            summary = f"\n\n## Summary of Key Findings\n\n{summary_response.content}"
            formatted_analyses_string += summary
    except Exception as e:
        # Skip summary if there's an error
        formatted_analyses_string += f"\n\n## Summary\nError generating summary: {str(e)}"
    
    # 결과 통합
    combined_results = [formatted_analyses_string]
    
    # 추가: 필터링 결과가 있으면 추가
    if hasattr(state, "filtered_data_result") and state.filtered_data_result:
        filtered_data_result = state.filtered_data_result
        combined_results.append(f"\n\n## RAG 기반 데이터 필터링 결과\n{filtered_data_result}")
    
    # 추가: 코드 생성 결과가 있으면 추가 
    if hasattr(state, "generated_code_result") and state.generated_code_result:
        generated_code_result = state.generated_code_result
        combined_results.append(f"\n\n## 생성된 분석 코드\n{generated_code_result}")
    
    # 모든 결과 결합
    final_formatted_result = "\n\n".join(combined_results)

    # Prepare the output dictionary matching the DataAnalysisResponse schema
    result_data = {
        "segment_analyses": analyses,
        "formatted_segment_analyses": final_formatted_result,
        "num_segments": len(analyses),
        "metadata": {
            "num_segments": len(analyses),
            "avg_analysis_length": sum(len(a) for a in analyses) / len(analyses)
            if analyses
            else 0,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
        },
    }

    # Return the dictionary. LangGraph will use this to update state.data_analysis_response
    return DataAnalysisOutput(data_analysis_response=result_data)


# Edge functions
def distribute_segments(state: DataAnalysisState) -> List[Send]:
    """Creates Send objects for each segment to be processed in parallel."""
    segments = state.segments if hasattr(state, "segments") else []
    if not segments:
        return []

    return [
        Send("analyze_segment", DataSegmentState(segment=segment, segment_id=i))
        for i, segment in enumerate(segments)
    ]


# 새 노드 적용을 위한 조건 함수
def should_apply_rag_filtering(state: DataAnalysisState) -> bool:
    """RAG 필터링이 필요한지 확인."""
    return (hasattr(state, "apply_rag_filtering") and 
            state.apply_rag_filtering and 
            hasattr(state, "data") and 
            state.data)

def should_generate_code(state: DataAnalysisState) -> bool:
    """코드 생성이 필요한지 확인."""
    return (hasattr(state, "apply_code_generation") and 
            state.apply_code_generation and 
            hasattr(state, "data") and 
            state.data)


# Create the graph
def create_data_analysis_graph() -> StateGraph:
    """Create the data analysis workflow graph."""
    # Initialize the graph with our state type
    workflow = StateGraph(DataAnalysisState, output=DataAnalysisOutput)

    # Add nodes
    workflow.add_node("process_data", process_data_node)
    workflow.add_node("analyze_segment", analyze_segment)
    workflow.add_node("filter_data_with_rag", filter_data_with_rag)
    workflow.add_node("generate_analysis_code", generate_analysis_code_node)
    workflow.add_node("combine_analyses", combine_analyses)

    # Add edges
    workflow.add_edge(START, "process_data")

    # Add conditional edges for parallel processing
    workflow.add_conditional_edges(
        "process_data", distribute_segments, ["analyze_segment"]
    )

    # Connect analyze_segment to combine_analyses
    # All parallel analyze_segment nodes will feed into combine_analyses
    workflow.add_edge("analyze_segment", "combine_analyses")
    
    # 새로운 조건부 노드 추가
    workflow.add_conditional_edges(
        "process_data",
        should_apply_rag_filtering,
        {
            True: "filter_data_with_rag",
            False: "combine_analyses"  # None 대신 combine_analyses로 이동
        }
    )
    
    workflow.add_conditional_edges(
        "process_data",
        should_generate_code,
        {
            True: "generate_analysis_code",
            False: "combine_analyses"  # None 대신 combine_analyses로 이동
        }
    )
    
    # 새 노드에서 combine_analyses로 연결
    workflow.add_edge("filter_data_with_rag", "combine_analyses")
    workflow.add_edge("generate_analysis_code", "combine_analyses")

    # After combining analyses, the subgraph finishes
    workflow.add_edge("combine_analyses", END)

    return workflow 