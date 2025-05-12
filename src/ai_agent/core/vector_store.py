"""
Faiss 기반 벡터 저장소 구현
"""
from typing import List, Dict, Any
import pickle
import numpy as np
import faiss
from ..config.settings import VECTOR_DB_DIMENSION, CACHE_DIR
from ..models.state import SearchResult

class VectorStore:
    def __init__(self, dimension: int = VECTOR_DB_DIMENSION):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Dict[str, Any]] = []
        
        # 저장된 인덱스가 있으면 로드
        vector_store_path = str(CACHE_DIR / "vector_store")
        try:
            self.load(vector_store_path)
            print(f"기존 벡터 저장소 로드 완료: {len(self.documents)}개 문서")
        except Exception as e:
            print(f"새 벡터 저장소 초기화: {str(e)}")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[int]:
        """문서와 임베딩을 저장소에 추가"""
        if len(documents) == 0:
            return []
            
        doc_ids = list(range(
            len(self.documents),
            len(self.documents) + len(documents)
        ))
        self.documents.extend(documents)
        
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        
        return doc_ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[SearchResult]:
        """쿼리 임베딩과 가장 유사한 문서 검색"""
        query_embedding_np = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx]
                results.append(
                    SearchResult(
                        title=doc.get("title", ""),
                        snippet=doc.get("content", ""),
                        source="rag",
                        score=float(distances[0][i])
                    )
                )
        return results
    
    def save(self, filepath: str = str(CACHE_DIR / "vector_store")):
        """벡터 저장소 상태를 파일로 저장"""
        with open(f"{filepath}.documents", 'wb') as f:
            pickle.dump(self.documents, f)
        faiss.write_index(self.index, f"{filepath}.index")
    
    def load(self, filepath: str = str(CACHE_DIR / "vector_store")):
        """파일에서 벡터 저장소 상태를 로드"""
        with open(f"{filepath}.documents", 'rb') as f:
            self.documents = pickle.load(f)
        self.index = faiss.read_index(f"{filepath}.index") 