"""
RAG(Retrieval-Augmented Generation) 에이전트 구현
이 모듈은 문서 기반 검색과 생성 AI를 결합한 RAG 시스템을 구현합니다.
벡터 검색을 통해 관련 문서를 찾고, 이를 기반으로 검색 결과를 생성합니다.
"""
from typing import List, Dict, Any
from ..core.embedding import get_embedding_manager
from ..core.vector_store import VectorStore
from ..core.cache import CacheManager
from ..models.state import SearchResult

class RAGAgent:
    def __init__(self):
        """
        RAG 에이전트 초기화
        - 임베딩 관리자 설정
        - 벡터 저장소 초기화
        - 캐시 관리자 설정
        """
        self.embedding_manager = get_embedding_manager()
        self.vector_store = VectorStore(
            dimension=self.embedding_manager.embedding_dim
        )
        self.cache_manager = CacheManager()
    
    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        쿼리를 처리하고 관련 문서를 검색합니다.
        
        Args:
            query (str): 검색 쿼리
            top_k (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 검색 결과
                - query: 원본 쿼리
                - results: 검색된 문서 목록
        """
        # 캐시 확인
        cache_key = f"rag_query_{query}_{top_k}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 쿼리 임베딩
        query_embedding = self.embedding_manager.encode(query)
        
        # 유사 문서 검색
        search_results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )
        
        result = {
            "query": query,
            "results": search_results
        }
        
        # 결과 캐싱
        self.cache_manager.set(cache_key, result)
        
        return result
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[int]:
        """
        문서를 벡터 저장소에 추가합니다.
        
        Args:
            documents (List[Dict[str, Any]]): 추가할 문서 목록
                각 문서는 content와 메타데이터를 포함해야 함
                
        Returns:
            List[int]: 추가된 문서의 ID 목록
        """
        if not documents:
            return []
        
        # 문서 텍스트 추출
        texts = [doc.get("content", "") for doc in documents]
        
        # 임베딩 생성
        embeddings = self.embedding_manager.encode_batch(texts)
        
        # 벡터 저장소에 추가
        doc_ids = self.vector_store.add_documents(documents, embeddings)
        
        return doc_ids
    
    def save_state(self):
        """
        에이전트의 현재 상태를 저장합니다.
        벡터 저장소의 상태를 디스크에 저장합니다.
        """
        self.vector_store.save()
    
    def load_state(self):
        """
        저장된 에이전트 상태를 로드합니다.
        디스크에서 벡터 저장소의 상태를 복원합니다.
        """
        self.vector_store.load()

"""
이 파일의 주요 역할:
1. RAG 시스템의 핵심 구현
2. 문서 기반 검색 처리
3. 벡터 검색 및 임베딩 관리
4. 검색 결과 캐싱

주요 기능:
- 쿼리 처리 및 문서 검색
- 문서 추가 및 임베딩
- 상태 저장 및 복원
- 캐시 기반 성능 최적화

사용된 주요 기술:
- 벡터 임베딩
- FAISS 기반 벡터 검색
- 캐시 시스템
- 상태 관리
""" 