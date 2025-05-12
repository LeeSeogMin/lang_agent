"""
RAG(Retrieval-Augmented Generation) 에이전트 구현
"""
from typing import List, Dict, Any
from ..core.embedding import get_embedding_manager
from ..core.vector_store import VectorStore
from ..core.cache import CacheManager
from ..models.state import SearchResult

class RAGAgent:
    def __init__(self):
        self.embedding_manager = get_embedding_manager()
        self.vector_store = VectorStore(
            dimension=self.embedding_manager.embedding_dim
        )
        self.cache_manager = CacheManager()
    
    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """쿼리 처리 및 관련 문서 검색"""
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
        """문서를 벡터 저장소에 추가"""
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
        """에이전트 상태 저장"""
        self.vector_store.save()
    
    def load_state(self):
        """에이전트 상태 로드"""
        self.vector_store.load() 