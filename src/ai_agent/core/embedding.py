"""
임베딩 모델 관리 및 텍스트 임베딩 기능
이 모듈은 SentenceTransformer를 사용하여 텍스트를 벡터로 변환합니다.
배치 처리를 지원하여 대량의 텍스트를 효율적으로 처리할 수 있습니다.
"""
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from ..config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

class EmbeddingManager:
    def __init__(self):
        """
        임베딩 매니저 초기화
        SentenceTransformer 모델을 지연 로딩합니다.
        """
        self._model: Optional[SentenceTransformer] = None
    
    def _ensure_model_loaded(self):
        """
        모델이 로드되어 있는지 확인하고 필요한 경우 로드
        메모리 효율을 위해 모델을 지연 로딩합니다.
        """
        if self._model is None:
            self._model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device=EMBEDDING_DEVICE
            )
    
    def encode(self, text: str) -> List[float]:
        """
        단일 텍스트를 임베딩 벡터로 변환
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            List[float]: 정규화된 임베딩 벡터
        """
        self._ensure_model_loaded()
        with torch.no_grad():
            embedding = self._model.encode(
                text,
                normalize_embeddings=True
            )
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 배치를 임베딩 벡터로 변환
        배치 처리를 통해 대량의 텍스트를 효율적으로 처리합니다.
        
        Args:
            texts (List[str]): 임베딩할 텍스트 리스트
            
        Returns:
            List[List[float]]: 정규화된 임베딩 벡터 리스트
        """
        self._ensure_model_loaded()
        with torch.no_grad():
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32
            )
        return embeddings.tolist()
    
    @property
    def embedding_dim(self) -> int:
        """
        임베딩 벡터의 차원 반환
        
        Returns:
            int: 임베딩 벡터의 차원
        """
        self._ensure_model_loaded()
        return self._model.get_sentence_embedding_dimension()

def get_embedding_manager() -> EmbeddingManager:
    """
    임베딩 매니저 인스턴스 반환
    싱글톤 패턴으로 구현되어 있습니다.
    
    Returns:
        EmbeddingManager: 임베딩 매니저 인스턴스
    """
    return EmbeddingManager()

"""
이 파일의 주요 역할:
1. 텍스트 임베딩 처리
2. 임베딩 모델 관리
3. 배치 처리 지원

주요 기능:
- 단일 텍스트 임베딩
- 배치 텍스트 임베딩
- 모델 지연 로딩
- 벡터 정규화

사용된 주요 기술:
- SentenceTransformer
- PyTorch
- 배치 처리
- 싱글톤 패턴
""" 