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

def create_error_result(error_msg: str) -> List[SearchResult]:
    """오류 결과를 생성합니다."""
    logger.error(error_msg)
    return [SearchResult(
        title="검색 오류",
        link="",
        snippet=error_msg,
        source="error",
        score=0.0
    )]

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
        
        if not self.api_key:
            raise ValueError("Tavily API 키가 필요합니다. TAVILY_API_KEY 환경 변수를 설정해주세요.")
    
    def search(
        self,
        query: str,
        search_type: str = "regular",
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        Tavily 검색 API를 사용하여 검색을 수행합니다.
        """
        logger.info(f"Tavily 검색 시작 - 쿼리: {query}, 타입: {search_type}")
        
        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # API 요청 데이터
            data = {
                "query": query,
                "search_depth": "advanced" if search_type == "scholar" else "basic",
                "max_results": num_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False
            }
            
            logger.debug(f"API 요청 데이터: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # API 요청 수행
            response = requests.post(
                TAVILY_API_URL,
                headers=headers,
                json=data,
                timeout=30
            )
            
            logger.info(f"API 응답 상태 코드: {response.status_code}")
            
            # 인증 오류 처리
            if response.status_code == 401:
                return create_error_result("Tavily API 인증 오류: API 키가 유효하지 않거나 만료되었습니다.")
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"API 응답 데이터: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                search_results = []
                if "results" in result:
                    for item in result["results"]:
                        search_results.append(
                            SearchResult(
                                title=item.get("title", ""),
                                link=item.get("url", ""),
                                snippet=item.get("content", ""),
                                source="web",
                                score=item.get("relevance_score", 0.0)
                            )
                        )
                    logger.info(f"검색 결과 {len(search_results)}개 처리됨")
                    return search_results
                else:
                    logger.warning("검색 결과가 없습니다.")
                    return []
            
            return create_error_result(f"API 오류 응답: {response.text}")
                
        except requests.exceptions.Timeout:
            return create_error_result("Tavily API 요청 시간 초과")
            
        except Exception as e:
            return create_error_result(f"예상치 못한 오류 발생: {str(e)}")
    
    def web_search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """일반 웹 검색을 수행합니다."""
        return self.search(query, "regular", num_results)
    
    def scholar_search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """학술 검색을 수행합니다."""
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