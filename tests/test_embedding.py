"""
임베딩 모델 테스트
"""
import pytest
import numpy as np
from src.ai_agent.core.embedding import EmbeddingManager

def test_embedding_manager_singleton():
    """EmbeddingManager 싱글톤 패턴 테스트"""
    manager1 = EmbeddingManager()
    manager2 = EmbeddingManager()
    assert manager1 is manager2

def test_embedding_dimension():
    """임베딩 차원 테스트"""
    manager = EmbeddingManager()
    assert manager.embedding_dim == 768  # ko-sroberta-multitask의 임베딩 차원

def test_text_encoding():
    """텍스트 인코딩 테스트"""
    manager = EmbeddingManager()
    text = "안녕하세요, 테스트 문장입니다."
    
    # 단일 텍스트 인코딩
    embedding = manager.encode(text)
    assert isinstance(embedding, list)
    assert len(embedding) == manager.embedding_dim
    assert all(isinstance(x, float) for x in embedding)

def test_batch_encoding():
    """배치 인코딩 테스트"""
    manager = EmbeddingManager()
    texts = [
        "첫 번째 테스트 문장입니다.",
        "두 번째 테스트 문장입니다.",
        "세 번째 테스트 문장입니다."
    ]
    
    # 배치 텍스트 인코딩
    embeddings = manager.encode_batch(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(len(emb) == manager.embedding_dim for emb in embeddings)
    assert all(isinstance(x, float) for emb in embeddings for x in emb)

def test_semantic_similarity():
    """의미적 유사도 테스트"""
    manager = EmbeddingManager()
    
    # 유사한 문장들
    text1 = "오늘 날씨가 좋습니다."
    text2 = "날씨가 맑고 화창하네요."
    text3 = "미분방정식은 어렵습니다."
    
    # 임베딩 생성
    emb1 = np.array(manager.encode(text1))
    emb2 = np.array(manager.encode(text2))
    emb3 = np.array(manager.encode(text3))
    
    # 코사인 유사도 계산
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # 유사한 문장 간 유사도가 더 높아야 함
    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)
    assert sim_12 > sim_13 