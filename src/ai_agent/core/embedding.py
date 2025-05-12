"""
임베딩 모델 관리 및 텍스트 임베딩 기능
"""
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from ..config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

class EmbeddingManager:
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
    
    def _ensure_model_loaded(self):
        """모델이 로드되어 있는지 확인하고 필요한 경우 로드"""
        if self._model is None:
            self._model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device=EMBEDDING_DEVICE
            )
    
    def encode(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩 벡터로 변환"""
        self._ensure_model_loaded()
        with torch.no_grad():
            embedding = self._model.encode(
                text,
                normalize_embeddings=True
            )
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """텍스트 배치를 임베딩 벡터로 변환"""
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
        """임베딩 벡터의 차원 반환"""
        self._ensure_model_loaded()
        return self._model.get_sentence_embedding_dimension()

def get_embedding_manager() -> EmbeddingManager:
    """임베딩 매니저 인스턴스 반환"""
    return EmbeddingManager() 