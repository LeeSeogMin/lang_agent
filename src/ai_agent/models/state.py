"""
시스템 상태 및 데이터 모델 정의
이 모듈은 시스템에서 사용되는 데이터 구조를 정의합니다.
Pydantic을 사용하여 데이터 유효성 검사와 직렬화를 지원합니다.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """
    검색 결과 모델
    웹 검색과 RAG 검색의 결과를 통합하여 표현합니다.
    
    Attributes:
        title (str): 검색 결과 제목
        snippet (str): 검색 결과 내용 요약
        source (str): 검색 결과 출처 (web/rag)
        score (float): 검색 결과 관련도 점수
        link (Optional[str]): 검색 결과 링크 (웹 검색의 경우)
    """
    title: str
    snippet: str
    source: str
    score: float = 1.0
    link: Optional[str] = None

class DocumentMetadata(BaseModel):
    """
    문서 메타데이터 모델
    업로드된 문서의 정보를 관리합니다.
    
    Attributes:
        doc_id (str): 문서 고유 식별자
        original_filename (str): 원본 파일명
        file_path (str): 저장된 파일 경로
        file_type (str): 파일 형식
        uploaded_at (str): 업로드 시간
        custom_metadata (Dict[str, Any]): 사용자 정의 메타데이터
        chunk_count (Optional[int]): 문서 청크 수
        chunk_ids (List[str]): 청크 ID 목록
    """
    doc_id: str
    original_filename: str
    file_path: str
    file_type: str
    uploaded_at: str
    custom_metadata: Dict[str, Any] = {}
    chunk_count: Optional[int] = None
    chunk_ids: List[str] = []

class SystemState(BaseModel):
    """
    시스템 전체 상태 모델
    시스템의 현재 상태를 관리합니다.
    
    Attributes:
        query (str): 현재 검색 쿼리
        search_results (List[SearchResult]): 검색 결과 목록
        documents (Dict[str, DocumentMetadata]): 업로드된 문서 정보
        conversation_history (List[Dict[str, str]]): 대화 기록
        current_agent (Optional[str]): 현재 활성화된 에이전트
        error (Optional[str]): 오류 메시지
        summary (Optional[Dict[str, Any]]): 검색 결과 요약
    """
    query: str
    search_results: List[SearchResult] = Field(default_factory=list)
    documents: Dict[str, DocumentMetadata] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    current_agent: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None  # 검색 결과 요약 추가

"""
이 파일의 주요 역할:
1. 데이터 모델 정의
2. 상태 관리
3. 데이터 유효성 검사

주요 기능:
- 검색 결과 모델링
- 문서 메타데이터 관리
- 시스템 상태 추적

사용된 주요 기술:
- Pydantic
- 타입 힌팅
- 기본값 설정
""" 