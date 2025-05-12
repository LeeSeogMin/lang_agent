"""
컨트롤러 에이전트 테스트
"""
import pytest
from unittest.mock import Mock, patch
from src.ai_agent.agents.controller_agent import ControllerAgent
from src.ai_agent.models.state import SearchResult, SystemState

@pytest.fixture
def mock_web_search_agent():
    """모의 웹 검색 에이전트"""
    mock = Mock()
    
    # 웹 검색 결과 설정
    mock.web_search.return_value = {
        "results": [
            SearchResult(
                title="웹 검색 결과",
                snippet="웹에서 찾은 정보입니다.",
                source="web",
                score=0.9
            )
        ]
    }
    
    # 학술 검색 결과 설정
    mock.scholar_search.return_value = {
        "results": [
            SearchResult(
                title="학술 검색 결과",
                snippet="학술 논문에서 찾은 정보입니다.",
                source="scholar",
                score=0.95
            )
        ]
    }
    
    # 통합 검색 결과 설정
    mock.combined_search.return_value = {
        "results": {
            "web": [
                SearchResult(
                    title="통합 웹 검색 결과",
                    snippet="웹에서 찾은 정보입니다.",
                    source="web",
                    score=0.85
                )
            ],
            "scholar": [
                SearchResult(
                    title="통합 학술 검색 결과",
                    snippet="학술 논문에서 찾은 정보입니다.",
                    source="scholar",
                    score=0.92
                )
            ]
        }
    }
    
    return mock

@pytest.fixture
def mock_rag_agent():
    """모의 RAG 에이전트"""
    mock = Mock()
    mock.process_query.return_value = {
        "results": [
            SearchResult(
                title="RAG 검색 결과",
                snippet="로컬 문서에서 찾은 정보입니다.",
                source="rag",
                score=0.88
            )
        ]
    }
    return mock

@pytest.fixture
def mock_cache_manager():
    """모의 캐시 매니저"""
    mock = Mock()
    mock.get.return_value = None  # 캐시 미스 시뮬레이션
    mock.set.return_value = True
    return mock

@pytest.fixture
def controller_agent(mock_web_search_agent, mock_rag_agent, mock_cache_manager):
    """테스트용 컨트롤러 에이전트"""
    with patch('src.ai_agent.agents.controller_agent.CacheManager') as mock_cm_cls:
        mock_cm_cls.return_value = mock_cache_manager
        agent = ControllerAgent(
            web_search_agent=mock_web_search_agent,
            rag_agent=mock_rag_agent
        )
        return agent

def test_analyze_query(controller_agent, mock_cache_manager):
    """쿼리 분석 테스트"""
    # 일반 쿼리
    analysis = controller_agent.analyze_query("일반적인 검색 쿼리")
    assert analysis["search_type"] == "combined"
    assert analysis["priority"]["web"] > analysis["priority"]["rag"]
    
    # 학술 쿼리
    analysis = controller_agent.analyze_query("이 논문의 연구 방법론")
    assert analysis["search_type"] == "scholar"
    assert analysis["priority"]["web"] < analysis["priority"]["rag"]
    
    # 캐시 동작 확인
    mock_cache_manager.get.assert_called()
    mock_cache_manager.set.assert_called()

def test_execute_search(controller_agent, mock_web_search_agent, mock_rag_agent):
    """검색 실행 테스트"""
    # 일반 검색
    results = controller_agent.execute_search("일반 검색 쿼리")
    assert "web_results" in results
    assert "rag_results" in results
    mock_web_search_agent.combined_search.assert_called_once()
    mock_rag_agent.process_query.assert_called_once()
    
    # 학술 검색
    results = controller_agent.execute_search("논문 검색 쿼리")
    assert "web_results" in results
    assert "rag_results" in results
    mock_web_search_agent.scholar_search.assert_called_once()

def test_rank_results(controller_agent):
    """결과 순위 지정 테스트"""
    results = {
        "web_results": [
            SearchResult(
                title="웹 결과 1",
                snippet="웹 내용 1",
                source="web",
                score=0.9
            ),
            SearchResult(
                title="웹 결과 2",
                snippet="웹 내용 2",
                source="web",
                score=0.8
            )
        ],
        "rag_results": [
            SearchResult(
                title="RAG 결과 1",
                snippet="RAG 내용 1",
                source="rag",
                score=0.95
            ),
            SearchResult(
                title="RAG 결과 2",
                snippet="RAG 내용 2",
                source="rag",
                score=0.85
            )
        ]
    }
    
    analysis = {
        "priority": {
            "web": 0.6,
            "rag": 0.4
        }
    }
    
    ranked_results = controller_agent.rank_results(results, analysis)
    
    assert len(ranked_results) == 4
    assert all(isinstance(result, SearchResult) for result in ranked_results)
    
    # 점수 순서 확인
    scores = [result.score for result in ranked_results]
    assert scores == sorted(scores, reverse=True)

def test_process_query(controller_agent):
    """쿼리 처리 테스트"""
    state = controller_agent.process_query("테스트 쿼리")
    
    assert isinstance(state, SystemState)
    assert state.query == "테스트 쿼리"
    assert len(state.search_results) > 0
    assert state.current_agent == "controller"
    assert all(isinstance(result, SearchResult) for result in state.search_results)

def test_cache_hit(controller_agent, mock_cache_manager):
    """캐시 히트 테스트"""
    # 캐시된 분석 결과 설정
    cached_analysis = {
        "query": "캐시된 쿼리",
        "search_type": "web",
        "requires_web_search": True,
        "requires_rag": True,
        "priority": {"web": 0.7, "rag": 0.3}
    }
    mock_cache_manager.get.return_value = cached_analysis
    
    analysis = controller_agent.analyze_query("캐시된 쿼리")
    assert analysis == cached_analysis
    mock_cache_manager.set.assert_not_called()  # 캐시 설정 호출되지 않음 