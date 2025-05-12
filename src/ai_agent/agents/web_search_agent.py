"""
웹 검색 에이전트
"""
from typing import Dict, List, Any, Optional
from ..utils.tavily_search import TavilySearchAPI
from ..core.cache import CacheManager
from ..models.state import SearchResult

class WebSearchAgent:
    def __init__(
        self,
        api_key: Optional[str] = None
    ):
        try:
            self.search_api = TavilySearchAPI(api_key=api_key)
        except ValueError as e:
            raise ValueError(f"Tavily API 키가 잘못되었습니다: {str(e)}")
        
        self.cache_manager = CacheManager()
    
    def process_query(
        self,
        query: str,
        search_type: str = "regular",
        num_results: int = 10
    ) -> Dict[str, Any]:
        """쿼리 처리 및 검색 결과 반환"""
        # 캐시 확인
        cache_key = f"web_search_{search_type}_{query}_{num_results}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 검색 수행
        try:
            if search_type == "scholar":
                search_results = self.search_api.scholar_search(
                    query,
                    num_results=num_results
                )
            else:
                search_results = self.search_api.web_search(
                    query,
                    num_results=num_results
                )
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            search_results = []
        
        result = {
            "query": query,
            "search_type": search_type,
            "results": search_results
        }
        
        # 결과 캐싱
        self.cache_manager.set(cache_key, result)
        
        return result
    
    def web_search(
        self,
        query: str,
        num_results: int = 10
    ) -> Dict[str, Any]:
        """일반 웹 검색 수행"""
        return self.process_query(query, "regular", num_results)
    
    def scholar_search(
        self,
        query: str,
        num_results: int = 10
    ) -> Dict[str, Any]:
        """학술 검색 수행"""
        return self.process_query(query, "scholar", num_results)
    
    def combined_search(
        self,
        query: str,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """일반 웹 검색과 학술 검색 결과 통합"""
        web_results = self.web_search(query, num_results)
        scholar_results = self.scholar_search(query, num_results)
        
        combined_results = {
            "query": query,
            "results": {
                "web": web_results["results"],
                "scholar": scholar_results["results"]
            }
        }
        
        return combined_results 