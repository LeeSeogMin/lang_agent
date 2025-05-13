"""
Faiss 기반 벡터 저장소 구현
이 모듈은 Faiss를 사용하여 고차원 벡터를 효율적으로 저장하고 검색합니다.
문서의 임베딩 벡터를 저장하고 유사도 검색을 수행합니다.
"""
from typing import List, Dict, Any
import pickle
import numpy as np
import faiss
from ..config.settings import VECTOR_DB_DIMENSION, CACHE_DIR
from ..models.state import SearchResult

class VectorStore:
    def __init__(self, dimension: int = VECTOR_DB_DIMENSION):
        """
        벡터 저장소 초기화
        
        Args:
            dimension (int): 임베딩 벡터의 차원
        """
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
        """
        문서와 임베딩을 저장소에 추가
        
        Args:
            documents (List[Dict[str, Any]]): 추가할 문서 리스트
            embeddings (List[List[float]]): 문서의 임베딩 벡터 리스트
            
        Returns:
            List[int]: 추가된 문서의 ID 리스트
        """
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
        """
        쿼리 임베딩과 가장 유사한 문서 검색
        
        Args:
            query_embedding (List[float]): 검색할 쿼리의 임베딩 벡터
            top_k (int): 반환할 결과의 수
            
        Returns:
            List[SearchResult]: 검색 결과 리스트
        """
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
        """
        벡터 저장소 상태를 파일로 저장
        
        Args:
            filepath (str): 저장할 파일 경로
        """
        with open(f"{filepath}.documents", 'wb') as f:
            pickle.dump(self.documents, f)
        faiss.write_index(self.index, f"{filepath}.index")
    
    def load(self, filepath: str = str(CACHE_DIR / "vector_store")):
        """
        파일에서 벡터 저장소 상태를 로드
        
        Args:
            filepath (str): 로드할 파일 경로
        """
        with open(f"{filepath}.documents", 'rb') as f:
            self.documents = pickle.load(f)
        self.index = faiss.read_index(f"{filepath}.index")

"""
이 파일의 주요 역할:
1. 벡터 저장소 구현
2. 문서 임베딩 저장
3. 유사도 검색

주요 기능:
- 문서 및 임베딩 추가
- 유사도 기반 검색
- 상태 저장 및 로드

사용된 주요 기술:
- Faiss (벡터 검색)
- NumPy (벡터 연산)
- Pickle (직렬화)
""" 