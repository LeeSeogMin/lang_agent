"""
Tavily Search API 통합
이 모듈은 Tavily Search API를 사용하여 웹 검색과 학술 검색을 수행합니다.
API 요청 처리, 응답 파싱, 에러 핸들링을 담당합니다.
"""
from typing import List, Optional
import logging
import json
import os
import requests
from ..models.state import SearchResult
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_API_URL = "https://api.tavily.com/v1/search"

class TavilySearchAPI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Tavily Search API 클라이언트 초기화
        
        Args:
            api_key (Optional[str]): Tavily API 키
                제공되지 않으면 환경 변수에서 로드
                
        Raises:
            ValueError: API 키가 없는 경우
        """
        self.api_key = api_key or TAVILY_API_KEY
        
        logger.debug(f"API Key: {self.api_key}")
        
        if not self.api_key:
            raise ValueError("Tavily API 키가 필요합니다.")
    
    def search(
        self,
        query: str,
        search_type: str = "regular",
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        Tavily 검색 API를 사용하여 검색을 수행합니다.
        
        Args:
            query (str): 검색 쿼리
            search_type (str): 검색 유형 ("regular" 또는 "scholar")
            num_results (int): 반환할 결과 수
            
        Returns:
            List[SearchResult]: 검색 결과 목록
                각 결과는 제목, 링크, 스니펫, 소스, 점수를 포함
                
        Note:
            - 학술 검색의 경우 고급 검색 깊이 사용
            - API 오류 발생 시 빈 목록 반환
        """
        logger.debug(f"검색 시작 - 쿼리: {query}, 타입: {search_type}")
        
        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            logger.debug(f"요청 헤더: {headers}")
            
            params = {
                "query": query,
                "search_depth": "advanced" if search_type == "scholar" else "basic",
                "max_results": num_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False
            }
            logger.debug(f"요청 파라미터: {json.dumps(params, indent=2)}")
            
            response = requests.post(
                TAVILY_API_URL,
                headers=headers,
                json=params
            )
            
            logger.debug(f"응답 상태 코드: {response.status_code}")
            logger.debug(f"응답 헤더: {dict(response.headers)}")
            logger.debug(f"응답 내용: {response.text[:500]}...")  # 처음 500자만 로깅
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"API 응답 받음: {result.keys()}")
            
            search_results = []
            if "results" in result:
                for item in result["results"]:
                    search_results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            link=item.get("url", ""),
                            snippet=item.get("description", ""),
                            source=search_type,
                            score=item.get("relevance_score", 1.0)
                        )
                    )
                logger.debug(f"검색 결과 {len(search_results)}개 처리됨")
            else:
                logger.warning("검색 결과 없음")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Tavily 검색 중 오류 발생: {str(e)}", exc_info=True)
            return []
    
    def web_search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        일반 웹 검색을 수행합니다.
        
        Args:
            query (str): 검색 쿼리
            num_results (int): 반환할 결과 수
            
        Returns:
            List[SearchResult]: 웹 검색 결과 목록
        """
        return self.search(query, "regular", num_results)
    
    def scholar_search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        학술 검색을 수행합니다.
        
        Args:
            query (str): 검색 쿼리
            num_results (int): 반환할 결과 수
            
        Returns:
            List[SearchResult]: 학술 검색 결과 목록
        """
        return self.search(query, "scholar", num_results)

"""
이 파일의 주요 역할:
1. Tavily Search API 통합
2. 웹 및 학술 검색 기능 제공
3. API 요청/응답 처리
4. 에러 핸들링 및 로깅

주요 기능:
- 일반 웹 검색
- 학술 검색
- API 응답 파싱
- 결과 변환

사용된 주요 기술:
- Tavily Search API
- requests 라이브러리
- 로깅 시스템
- 환경 변수 관리
""" 