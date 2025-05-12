"""
Faiss 벡터 저장소 테스트
"""
import pytest
import numpy as np
from pathlib import Path
from src.ai_agent.core.vector_store import VectorStore
from src.ai_agent.models.state import SearchResult
from src.ai_agent.config.settings import CACHE_DIR

@pytest.fixture
def vector_store():
    """테스트용 벡터 저장소 인스턴스"""
    return VectorStore(dimension=768)

@pytest.fixture
def sample_documents():
    """테스트용 샘플 문서"""
    return [
        {
            "title": "파이썬 프로그래밍",
            "content": "파이썬은 간단하고 배우기 쉬운 프로그래밍 언어입니다.",
            "metadata": {"type": "tutorial"}
        },
        {
            "title": "머신러닝 기초",
            "content": "머신러닝은 데이터로부터 학습하는 알고리즘을 연구하는 분야입니다.",
            "metadata": {"type": "article"}
        },
        {
            "title": "딥러닝 소개",
            "content": "딥러닝은 인공신경망을 기반으로 하는 머신러닝의 한 분야입니다.",
            "metadata": {"type": "article"}
        }
    ]

@pytest.fixture
def sample_embeddings():
    """테스트용 샘플 임베딩"""
    # 임의의 정규화된 벡터 생성
    rng = np.random.default_rng(42)
    embeddings = rng.random((3, 768))
    # L2 정규화
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    return embeddings.tolist()

def test_vector_store_initialization(vector_store):
    """벡터 저장소 초기화 테스트"""
    assert vector_store.dimension == 768
    assert len(vector_store.documents) == 0
    assert vector_store.index is not None

def test_add_documents(vector_store, sample_documents, sample_embeddings):
    """문서 추가 테스트"""
    doc_ids = vector_store.add_documents(sample_documents, sample_embeddings)
    
    assert len(doc_ids) == len(sample_documents)
    assert len(vector_store.documents) == len(sample_documents)
    assert all(isinstance(id, int) for id in doc_ids)

def test_search(vector_store, sample_documents, sample_embeddings):
    """문서 검색 테스트"""
    # 문서 추가
    vector_store.add_documents(sample_documents, sample_embeddings)
    
    # 임의의 쿼리 벡터로 검색
    query_embedding = np.random.random(768).tolist()
    results = vector_store.search(query_embedding, top_k=2)
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, SearchResult) for result in results)
    assert all(hasattr(result, 'score') for result in results)
    
    # 점수 순서 확인
    scores = [result.score for result in results]
    assert scores == sorted(scores)  # 오름차순 (L2 거리)

def test_save_and_load(vector_store, sample_documents, sample_embeddings):
    """상태 저장 및 로드 테스트"""
    # 문서 추가
    vector_store.add_documents(sample_documents, sample_embeddings)
    
    # 상태 저장
    test_filepath = str(CACHE_DIR / "test_vector_store")
    vector_store.save(test_filepath)
    
    # 새 인스턴스 생성 및 상태 로드
    new_store = VectorStore(dimension=768)
    new_store.load(test_filepath)
    
    # 상태 비교
    assert len(new_store.documents) == len(vector_store.documents)
    
    # 동일한 쿼리로 검색 결과 비교
    query_embedding = np.random.random(768).tolist()
    results1 = vector_store.search(query_embedding, top_k=2)
    results2 = new_store.search(query_embedding, top_k=2)
    
    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1.title == r2.title
        assert r1.snippet == r2.snippet
        assert r1.score == r2.score
    
    # 테스트 파일 정리
    Path(f"{test_filepath}.documents").unlink(missing_ok=True)
    Path(f"{test_filepath}.index").unlink(missing_ok=True)

def test_empty_documents(vector_store):
    """빈 문서 리스트 처리 테스트"""
    doc_ids = vector_store.add_documents([], [])
    assert len(doc_ids) == 0
    
    query_embedding = np.random.random(768).tolist()
    results = vector_store.search(query_embedding, top_k=5)
    assert len(results) == 0 