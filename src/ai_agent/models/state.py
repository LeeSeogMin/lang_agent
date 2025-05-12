"""
시스템 상태 및 데이터 모델 정의
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """검색 결과 모델"""
    title: str
    snippet: str
    source: str
    score: float = 1.0
    link: Optional[str] = None

class DocumentMetadata(BaseModel):
    """문서 메타데이터 모델"""
    doc_id: str
    original_filename: str
    file_path: str
    file_type: str
    uploaded_at: str
    custom_metadata: Dict[str, Any] = {}
    chunk_count: Optional[int] = None
    chunk_ids: List[str] = []

class SystemState(BaseModel):
    """시스템 전체 상태 모델"""
    query: str
    search_results: List[SearchResult] = Field(default_factory=list)
    documents: Dict[str, DocumentMetadata] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    current_agent: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None  # 검색 결과 요약 추가 