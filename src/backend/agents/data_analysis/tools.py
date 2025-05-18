"""Data analysis agent tools."""

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os
import tempfile
import pandas as pd
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from pathlib import Path

from backend.utils.file_utils import count_tokens, create_chunks
from backend.config import settings

# Define input schemas
class SegmentDataInput(BaseModel):
    """Input schema for data segmentation."""

    data: str = Field(description="The data content to split into segments")
    segment_size: int = Field(
        default=500,
        description="Target size for each segment in tokens",
        ge=100,
        le=4000,
    )
    segment_overlap: int = Field(
        default=50,
        description="Number of tokens to overlap between segments",
        ge=20,
        le=500,
    )


class ValidateDataInput(BaseModel):
    """Input schema for data validation."""
    
    data: str = Field(description="The data content to validate")
    expected_format: Optional[str] = Field(
        default=None, 
        description="Expected format of the data (e.g., 'csv', 'json', 'text')"
    )


class AnalyzeDataStatisticsInput(BaseModel):
    """Input schema for statistical analysis."""
    
    data: str = Field(description="The data content to analyze")
    analysis_type: str = Field(
        default="general",
        description="Type of analysis to perform ('general', 'text', 'numerical')"
    )


class FormatResultsInput(BaseModel):
    """Input schema for results formatting."""
    
    results: List[str] = Field(description="List of analysis results to format")
    format_type: str = Field(
        default="text",
        description="Output format type ('text', 'json', 'html')"
    )


class FilterDataRagInput(BaseModel):
    """Input schema for RAG-based data filtering."""
    
    data: str = Field(description="The data content to filter")
    query: str = Field(description="Natural language query for filtering the data")
    data_format: str = Field(
        default="auto",
        description="Format of the data ('csv', 'json', 'text', 'auto')"
    )


class GenerateCodeInput(BaseModel):
    """Input schema for code generation."""
    
    data: str = Field(description="The data or data sample to analyze")
    task: str = Field(description="Natural language description of the analysis task")
    code_type: str = Field(
        default="pandas",
        description="Type of code to generate ('pandas', 'numpy', 'matplotlib')"
    )


@tool("segment_data", args_schema=SegmentDataInput)
async def segment_data(
    data: str, segment_size: int = 500, segment_overlap: int = 50
) -> Dict[str, Any]:
    """Split data into segments for analysis while preserving context.

    Args:
        data: The data text to split
        segment_size: Target size for each segment in tokens
        segment_overlap: Number of tokens to overlap between segments

    Returns:
        Dictionary containing segments and metadata about the segmentation process
    """
    if not data:
        return {"segments": [], "error": "No data provided"}

    try:
        segments = await create_chunks(data, segment_size, segment_overlap)
        total_tokens = await count_tokens(data)
        segment_tokens = [await count_tokens(s) for s in segments]

        return {
            "segments": segments,
            "num_segments": len(segments),
            "metadata": {
                "avg_segment_size": sum(segment_tokens) / len(segments) if segments else 0,
                "num_segments": len(segments),
                "total_tokens": total_tokens,
            },
        }
    except Exception as e:
        return {"segments": [], "error": f"Error segmenting data: {str(e)}"}


@tool("validate_data", args_schema=ValidateDataInput)
async def validate_data(data: str, expected_format: Optional[str] = None) -> Dict[str, Any]:
    """Validate the data format and structure.
    
    Args:
        data: The data content to validate
        expected_format: Expected format of the data (e.g., 'csv', 'json', 'text')
        
    Returns:
        Dictionary with validation results and metadata
    """
    if not data:
        return {"is_valid": False, "error": "No data provided"}
    
    try:
        # Basic validation
        validation_results = {
            "is_valid": True,
            "data_length": len(data),
            "data_preview": data[:100] + "..." if len(data) > 100 else data,
            "validation_details": {}
        }
        
        # Format-specific validation
        if expected_format:
            expected_format = expected_format.lower()
            if expected_format == "csv":
                # Check for comma-separated values
                lines = data.strip().split("\n")
                if not lines:
                    validation_results["is_valid"] = False
                    validation_results["validation_details"]["csv_error"] = "Empty data"
                else:
                    header_fields = lines[0].count(",") + 1
                    validation_results["validation_details"]["csv"] = {
                        "rows": len(lines),
                        "header_fields": header_fields,
                        "consistent_fields": all(line.count(",") + 1 == header_fields for line in lines[1:]),
                    }
                    if not validation_results["validation_details"]["csv"]["consistent_fields"]:
                        validation_results["is_valid"] = False
                        validation_results["validation_details"]["csv_error"] = "Inconsistent number of fields"
                        
            elif expected_format == "json":
                # Check if it's valid JSON
                import json
                try:
                    json.loads(data)
                    validation_results["validation_details"]["json"] = {"valid_json": True}
                except json.JSONDecodeError as e:
                    validation_results["is_valid"] = False
                    validation_results["validation_details"]["json"] = {
                        "valid_json": False,
                        "error": str(e)
                    }
        
        # Detect data type if not specified
        if not expected_format:
            # Check for common patterns
            if data.strip().startswith("{") and data.strip().endswith("}"):
                import json
                try:
                    json.loads(data)
                    validation_results["detected_format"] = "json"
                except:
                    pass
                    
            elif "," in data and "\n" in data:
                # Likely CSV
                lines = data.strip().split("\n")
                if len(lines) > 1:
                    fields = lines[0].count(",") + 1
                    if all(line.count(",") + 1 == fields for line in lines[1:]):
                        validation_results["detected_format"] = "csv"
            
            # Default to text if no specific format detected
            if "detected_format" not in validation_results:
                validation_results["detected_format"] = "text"
        
        return validation_results
    
    except Exception as e:
        return {"is_valid": False, "error": f"Error validating data: {str(e)}"}


@tool("analyze_data_statistics", args_schema=AnalyzeDataStatisticsInput)
async def analyze_data_statistics(data: str, analysis_type: str = "general") -> Dict[str, Any]:
    """Perform statistical analysis on the data.
    
    Args:
        data: The data content to analyze
        analysis_type: Type of analysis to perform ('general', 'text', 'numerical')
        
    Returns:
        Dictionary with statistical analysis results
    """
    if not data:
        return {"error": "No data provided"}
    
    try:
        # Basic stats for all analysis types
        stats = {
            "length": len(data),
            "tokens": await count_tokens(data),
            "lines": data.count("\n") + 1,
        }
        
        if analysis_type == "text" or analysis_type == "general":
            # Text analysis
            import re
            words = re.findall(r'\b\w+\b', data)
            stats.update({
                "word_count": len(words),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "unique_words": len(set(words)),
                "character_count": len(data),
                "alphanumeric_count": sum(c.isalnum() for c in data),
                "non_alphanumeric_count": sum(not c.isalnum() for c in data),
            })
            
            # Most common words (for text analysis only)
            if analysis_type == "text":
                from collections import Counter
                common_words = Counter(words).most_common(10)
                stats["common_words"] = [{"word": word, "count": count} for word, count in common_words]
                
        if analysis_type == "numerical" or analysis_type == "general":
            # Try to extract numbers from the text
            import re
            numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', data)]
            
            if numbers:
                stats.update({
                    "numeric_count": len(numbers),
                    "min": min(numbers),
                    "max": max(numbers),
                    "mean": sum(numbers) / len(numbers),
                    "has_negative": any(n < 0 for n in numbers),
                })
                
                # Additional stats for numerical analysis
                if analysis_type == "numerical":
                    # Calculate median, variance, etc.
                    numbers.sort()
                    mid = len(numbers) // 2
                    median = numbers[mid] if len(numbers) % 2 == 1 else (numbers[mid-1] + numbers[mid]) / 2
                    variance = sum((x - stats["mean"]) ** 2 for x in numbers) / len(numbers)
                    
                    stats.update({
                        "median": median,
                        "variance": variance,
                        "std_dev": variance ** 0.5,
                    })
        
        return {
            "statistics": stats,
            "analysis_type": analysis_type,
        }
            
    except Exception as e:
        return {"error": f"Error analyzing data statistics: {str(e)}"}


@tool("format_results", args_schema=FormatResultsInput)
async def format_results(results: List[str], format_type: str = "text") -> Dict[str, Any]:
    """Format analysis results into a structured output.
    
    Args:
        results: List of analysis results to format
        format_type: Output format type ('text', 'json', 'html')
        
    Returns:
        Dictionary with formatted results
    """
    if not results:
        return {"formatted_result": "", "error": "No results provided"}
    
    try:
        if format_type == "text":
            # Simple text formatting
            formatted = "\n\n".join([f"Analysis {i+1}:\n{result}" for i, result in enumerate(results)])
            
        elif format_type == "json":
            # JSON formatting
            import json
            formatted = json.dumps({
                "analyses": [{"index": i, "content": result} for i, result in enumerate(results)],
                "total_analyses": len(results),
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }, indent=2)
            
        elif format_type == "html":
            # HTML formatting
            html_parts = ["<div class='analysis-results'>"]
            for i, result in enumerate(results):
                html_parts.append(f"<div class='analysis-item'><h3>Analysis {i+1}</h3><p>{result}</p></div>")
            html_parts.append("</div>")
            formatted = "\n".join(html_parts)
            
        else:
            # Default to text if format is not recognized
            formatted = "\n\n".join(results)
            
        return {
            "formatted_result": formatted,
            "format_type": format_type,
            "num_results": len(results)
        }
        
    except Exception as e:
        return {"formatted_result": "", "error": f"Error formatting results: {str(e)}"}


@tool("filter_data_rag", args_schema=FilterDataRagInput)
async def filter_data_rag(data: str, query: str, data_format: str = "auto") -> Dict[str, Any]:
    """RAG 기반 데이터 필터링을 수행합니다.
    
    Args:
        data: 필터링할 데이터 내용
        query: 필터링을 위한 자연어 쿼리
        data_format: Format of the data ('csv', 'json', 'text', 'auto')
        
    Returns:
        Dictionary with filtered results and metadata
    """
    if not data or not query:
        return {"error": "Data or query not provided"}
    
    try:
        # 데이터 형식 감지 (auto일 경우)
        detected_format = data_format
        if data_format == "auto":
            validation_result = await validate_data(data)
            detected_format = validation_result.get("detected_format", "text")
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{detected_format}", mode="w", encoding="utf-8") as temp_file:
            temp_file.write(data)
            temp_file_path = temp_file.name
        
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        # 데이터 형식에 따른 처리
        if detected_format == "csv":
            try:
                # CSV 데이터를 DataFrame으로 변환 - 오류 처리 추가
                try:
                    df = pd.read_csv(temp_file_path)
                except pd.errors.ParserError:
                    # 콤마로 나누어진 데이터를 직접 파싱
                    lines = data.strip().split('\n')
                    header = lines[0].split(',')
                    data_rows = [line.split(',') for line in lines[1:]]
                    df = pd.DataFrame(data_rows, columns=header)
                
                # 데이터프레임이 너무 크면 샘플링
                if len(df) > 1000:
                    df_sample = df.sample(1000)
                else:
                    df_sample = df
                
                # 기본 데이터 요약 생성
                data_summary = f"""
## CSV 데이터 요약
- 행 수: {len(df)}
- 열 수: {len(df.columns)}
- 열 이름: {', '.join(df.columns)}

### 처음 5개 행:
{df.head().to_string()}

### 기술 통계:
{df.describe().to_string()}
                """
                
                # DataFrame 에이전트 생성
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type="openai-tools",
                    handle_parsing_errors=True
                )
                
                # 쿼리에 기반한 필터링 수행
                system_message = f"""
                당신은 데이터 분석 전문가입니다. 사용자의 자연어 쿼리를 기반으로 데이터를 필터링하고 결과를 반환해야 합니다.
                사용자 쿼리: {query}
                
                이 쿼리를 pandas 코드로 변환하여 필터링하고, 결과 데이터프레임을 반환하세요.
                결과는 표 형식으로 깔끔하게 표시하고, 코드와 함께 분석 내용을 상세히 설명해주세요.
                """
                
                result = agent.invoke({"input": system_message})
                agent_response = result.get("output", "분석 결과를 찾을 수 없습니다.")
                
                # 결합된 결과 생성
                filtered_data = f"{data_summary}\n\n### 쿼리 결과:\n{agent_response}"
                
                # 원본 및 필터링 데이터 요약
                metadata = {
                    "original_rows": len(df),
                    "original_columns": len(df.columns),
                    "column_names": list(df.columns),
                    "detected_format": "csv",
                    "query": query
                }
                
                return {
                    "filtered_result": filtered_data,
                    "metadata": metadata
                }
                
            except Exception as e:
                print(f"CSV 처리 중 오류: {str(e)}")
                import traceback
                print(f"상세 오류: {traceback.format_exc()}")
                # CSV 처리 실패 시 텍스트 기반 RAG로 폴백
                detected_format = "text"
        
        # 텍스트 기반 RAG
        if detected_format in ["text", "json"]:
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            chunks = text_splitter.split_text(data)
            
            # 임베딩 및 벡터 저장소 설정
            embeddings = OpenAIEmbeddings()
            
            # 임시 디렉토리에 Chroma DB 생성
            with tempfile.TemporaryDirectory() as temp_dir:
                vectorstore = Chroma.from_texts(
                    chunks, 
                    embeddings,
                    persist_directory=temp_dir
                )
                
                # 데이터 검색
                docs = vectorstore.similarity_search(query, k=3)
                context = "\n".join([doc.page_content for doc in docs])
                
                # LLM을 사용하여 검색 결과 처리
                prompt_template = """
                다음은 데이터 세트에서 추출한 일부 내용입니다:
                
                {context}
                
                위 데이터에서 다음 쿼리에 대한 답변을 해주세요: {query}
                
                관련된 정보를 자세히 설명하고, 쿼리와 관련된 모든 정보를 포함해주세요.
                """
                
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "query"]
                )
                
                formatted_prompt = prompt.format(context=context, query=query)
                response = llm.invoke(formatted_prompt)
                
                return {
                    "filtered_result": response.content,
                    "metadata": {
                        "chunks_processed": len(chunks),
                        "detected_format": detected_format,
                        "query": query
                    }
                }
        
        return {
            "error": f"지원하지 않는 데이터 형식: {detected_format}"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"RAG 필터링 오류: {error_details}")
        return {"error": f"RAG 기반 필터링 중 오류 발생: {str(e)}"}
    finally:
        # 임시 파일 정리
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass


@tool("generate_analysis_code", args_schema=GenerateCodeInput)
async def generate_analysis_code(data: str, task: str, code_type: str = "pandas") -> Dict[str, Any]:
    """Generate code for data analysis based on natural language task description.
    
    Args:
        data: Sample data to analyze
        task: Natural language description of the analysis task
        code_type: Type of code to generate ('pandas', 'numpy', 'matplotlib')
        
    Returns:
        Dictionary with generated code and execution results
    """
    if not data or not task:
        return {"error": "Data sample or task description not provided"}
    
    try:
        # 데이터 형식 감지 및 샘플 준비
        validation_result = await validate_data(data)
        detected_format = validation_result.get("detected_format", "text")
        
        # LLM 모델 초기화
        client = OpenAI()
        
        # 시스템 프롬프트 구성
        system_prompt = f"""
        당신은 데이터 분석 전문가입니다. 사용자가 제공한 데이터와 작업 설명을 기반으로 Python 코드를 생성해야 합니다.
        
        생성할 코드 유형: {code_type}
        데이터 형식: {detected_format}
        
        코드는 다음 요구사항을 충족해야 합니다:
        1. 완전히 실행 가능해야 합니다
        2. 모든 필요한 라이브러리를 import 해야 합니다
        3. 주석을 통해 코드의 각 부분을 명확히 설명해야 합니다
        4. 한국어 주석을 사용해야 합니다
        5. 최소한의 외부 종속성을 가져야 합니다
        
        다음 데이터 샘플이 제공됩니다:
        ```
        {data[:500]}... (샘플 데이터)
        ```
        
        작업 설명: {task}
        
        응답은 코드 블록 내에 Python 코드만 포함해야 합니다. 추가 설명이나 코멘트는 생략하세요.
        """
        
        # 코드 유형에 따른 프롬프트 조정
        if code_type == "pandas":
            system_prompt += "\n\n주로 pandas 라이브러리를 사용하여 데이터 처리 및 분석 코드를 작성하세요."
        elif code_type == "numpy":
            system_prompt += "\n\n가능한 경우 numpy 함수를 활용하여 효율적인 계산 코드를 작성하세요."
        elif code_type == "matplotlib":
            system_prompt += "\n\n데이터 시각화에 matplotlib 또는 seaborn을 사용하여 시각화 코드를 작성하세요."
        
        # LLM에 코드 생성 요청
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "위 작업에 맞는 Python 코드를 생성해주세요."}
            ],
            temperature=0.1,
        )
        
        generated_code = response.choices[0].message.content
        
        # 코드 블록 추출 (```python ... ``` 형식에서)
        import re
        code_block_pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(code_block_pattern, generated_code)
        
        if matches:
            clean_code = matches[0].strip()
        else:
            clean_code = generated_code.strip()
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as code_file:
            code_file_path = code_file.name
            code_file.write(clean_code)
        
        # 데이터 파일 생성
        with tempfile.NamedTemporaryFile(suffix=f".{detected_format}", delete=False, mode="w", encoding="utf-8") as data_file:
            data_file_path = data_file.name
            data_file.write(data)
        
        # 실행 결과 요약
        result_summary = {
            "generated_code": clean_code,
            "detected_format": detected_format,
            "code_type": code_type,
            "task": task,
            "execution_info": "코드가 생성되었습니다. 실행은 별도 환경에서 해야 합니다."
        }
        
        return result_summary
        
    except Exception as e:
        return {"error": f"Error generating analysis code: {str(e)}"}
    finally:
        # 임시 파일 정리
        for path in ['code_file_path', 'data_file_path']:
            if path in locals():
                try:
                    os.unlink(locals()[path])
                except:
                    pass 