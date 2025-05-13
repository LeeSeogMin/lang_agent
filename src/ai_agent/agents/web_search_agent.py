"""
웹 검색 에이전트
이 모듈은 Tavily Search API를 사용하여 웹 검색과 학술 검색을 수행합니다.
검색 결과를 캐싱하여 성능을 최적화하고, 다양한 검색 유형을 지원합니다.
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
        """
        웹 검색 에이전트 초기화
        
        Args:
            api_key (Optional[str]): Tavily API 키
                제공되지 않으면 환경 변수에서 로드
        """
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
        """
        쿼리를 처리하고 검색 결과를 반환합니다.
        
        Args:
            query (str): 검색 쿼리
            search_type (str): 검색 유형 ("regular" 또는 "scholar")
            num_results (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 검색 결과
                - query: 원본 쿼리
                - search_type: 검색 유형
                - results: 검색 결과 목록
        """
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
        """
        일반 웹 검색을 수행합니다.
        
        Args:
            query (str): 검색 쿼리
            num_results (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 웹 검색 결과
        """
        return self.process_query(query, "regular", num_results)
    
    def scholar_search(
        self,
        query: str,
        num_results: int = 10
    ) -> Dict[str, Any]:
        """
        학술 검색을 수행합니다.
        
        Args:
            query (str): 검색 쿼리
            num_results (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 학술 검색 결과
        """
        return self.process_query(query, "scholar", num_results)
    
    def combined_search(
        self,
        query: str,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """
        일반 웹 검색과 학술 검색 결과를 통합합니다.
        
        Args:
            query (str): 검색 쿼리
            num_results (int): 각 검색 유형별 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 통합된 검색 결과
                - query: 원본 쿼리
                - results: 
                    - web: 일반 웹 검색 결과
                    - scholar: 학술 검색 결과
        """
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

"""
이 파일의 주요 역할:
1. 웹 검색 기능 구현
2. 학술 검색 기능 구현
3. 검색 결과 캐싱
4. 다중 검색 소스 통합

주요 기능:
- 일반 웹 검색
- 학술 검색
- 통합 검색
- 결과 캐싱

사용된 주요 기술:
- Tavily Search API
- 캐시 시스템
- 에러 처리
- 결과 통합
""" 