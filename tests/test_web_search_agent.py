"""
웹 검색 에이전트 테스트
"""
import pytest
from unittest.mock import Mock, patch
from src.ai_agent.agents.web_search_agent import WebSearchAgent
from src.ai_agent.models.state import SearchResult

@pytest.fixture
def mock_google_search():
    """모의 Google 검색 API"""
    mock = Mock()
    
    # 웹 검색 결과 설정
    mock.web_search.return_value = [
        SearchResult(
            title="테스트 웹 페이지",
            snippet="이것은 테스트 웹 페이지입니다.",
            source="web",
            score=0.9
        )
    ]
    
    # 학술 검색 결과 설정
    mock.scholar_search.return_value = [
        SearchResult(
            title="테스트 학술 논문",
            snippet="이것은 테스트 학술 논문입니다.",
            source="scholar",
            score=0.95
        )
    ]
    
    return mock

@pytest.fixture
def mock_cache_manager():
    """모의 캐시 매니저"""
    mock = Mock()
    mock.get.return_value = None  # 캐시 미스 시뮬레이션
    mock.set.return_value = True
    return mock

@pytest.fixture
def web_search_agent(mock_google_search, mock_cache_manager):
    """테스트용 웹 검색 에이전트"""
    with patch('src.ai_agent.agents.web_search_agent.GoogleSearchAPI') as mock_gs_cls, \
         patch('src.ai_agent.agents.web_search_agent.CacheManager') as mock_cm_cls:
        mock_gs_cls.return_value = mock_google_search
        mock_cm_cls.return_value = mock_cache_manager
        agent = WebSearchAgent()
        return agent

def test_web_search(web_search_agent, mock_google_search, mock_cache_manager):
    """웹 검색 테스트"""
    result = web_search_agent.web_search("테스트 쿼리")
    
    # 검색 호출 확인
    mock_google_search.web_search.assert_called_once_with(
        "테스트 쿼리",
        num_results=10
    )
    
    # 결과 구조 확인
    assert result["query"] == "테스트 쿼리"
    assert result["search_type"] == "web"
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 1
    
    # 캐시 동작 확인
    mock_cache_manager.get.assert_called_once()
    mock_cache_manager.set.assert_called_once()

def test_scholar_search(web_search_agent, mock_google_search, mock_cache_manager):
    """학술 검색 테스트"""
    result = web_search_agent.scholar_search("학술 쿼리")
    
    # 검색 호출 확인
    mock_google_search.scholar_search.assert_called_once_with(
        "학술 쿼리",
        num_results=10
    )
    
    # 결과 구조 확인
    assert result["query"] == "학술 쿼리"
    assert result["search_type"] == "scholar"
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 1

def test_combined_search(web_search_agent, mock_google_search):
    """통합 검색 테스트"""
    result = web_search_agent.combined_search("통합 쿼리", num_results=5)
    
    # 검색 호출 확인
    mock_google_search.web_search.assert_called_once_with(
        "통합 쿼리",
        num_results=5
    )
    mock_google_search.scholar_search.assert_called_once_with(
        "통합 쿼리",
        num_results=5
    )
    
    # 결과 구조 확인
    assert result["query"] == "통합 쿼리"
    assert "web" in result["results"]
    assert "scholar" in result["results"]
    assert len(result["results"]["web"]) == 1
    assert len(result["results"]["scholar"]) == 1

def test_cache_hit(web_search_agent, mock_google_search, mock_cache_manager):
    """캐시 히트 테스트"""
    # 캐시 히트 시뮬레이션
    cached_result = {
        "query": "캐시 쿼리",
        "search_type": "web",
        "results": [
            SearchResult(
                title="캐시된 결과",
                snippet="이것은 캐시된 결과입니다.",
                source="web",
                score=0.8
            )
        ]
    }
    mock_cache_manager.get.return_value = cached_result
    
    result = web_search_agent.web_search("캐시 쿼리")
    
    # 캐시에서 결과를 가져왔는지 확인
    assert result == cached_result
    mock_google_search.web_search.assert_not_called()  # API 호출하지 않음

def test_invalid_api_key():
    """잘못된 API 키 테스트"""
    with patch('src.ai_agent.agents.web_search_agent.GoogleSearchAPI') as mock_gs_cls:
        mock_gs_cls.side_effect = ValueError("Google API 키와 CSE ID가 필요합니다.")
        with pytest.raises(ValueError) as exc_info:
            WebSearchAgent(api_key="invalid_key")
        assert "Google API 키" in str(exc_info.value) 