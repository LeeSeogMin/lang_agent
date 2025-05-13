"""
컨트롤러 에이전트 구현
이 모듈은 AI 검색 에이전트의 중앙 제어 시스템을 구현합니다.
웹 검색과 RAG(Retrieval Augmented Generation) 검색을 조율하고,
검색 결과를 통합하여 최종 응답을 생성합니다.
"""
from typing import Dict, List, Any, Optional
import os
from openai import OpenAI
from .web_search_agent import WebSearchAgent
from .rag_agent import RAGAgent
from ..core.cache import CacheManager
from ..models.state import SystemState, SearchResult
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class ControllerAgent:
    def __init__(
        self,
        web_search_agent: Optional[WebSearchAgent] = None,
        rag_agent: Optional[RAGAgent] = None
    ):
        """
        컨트롤러 에이전트 초기화
        
        Args:
            web_search_agent: 웹 검색 에이전트 인스턴스 (선택적)
            rag_agent: RAG 에이전트 인스턴스 (선택적)
        """
        self.web_search_agent = web_search_agent or WebSearchAgent()
        self.rag_agent = rag_agent or RAGAgent()
        self.cache_manager = CacheManager()
        
        # OpenAI 클라이언트 초기화
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.openai_client = OpenAI(api_key=openai_api_key)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리를 분석하고 실행 계획을 수립합니다.
        
        Args:
            query (str): 사용자 검색 쿼리
            
        Returns:
            Dict[str, Any]: 쿼리 분석 결과
                - requires_web_search: 웹 검색 필요 여부
                - requires_rag: RAG 검색 필요 여부
                - search_type: 검색 유형 (web/scholar/combined)
                - priority: 각 검색 방식의 가중치
        """
        # 캐시 확인
        cache_key = f"query_analysis_{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 쿼리 분석 결과
        analysis = {
            "query": query,
            "requires_web_search": False,  # 웹 검색 비활성화 (API 키 오류 때문)
            "requires_rag": True,  # 기본적으로 RAG 검색 수행
            "search_type": "combined",  # web, scholar, combined 중 하나
            "priority": {
                "web": 0.0,  # 웹 검색 결과 가중치
                "rag": 1.0   # RAG 결과 가중치
            }
        }
        
        # 학술적 키워드 포함 여부 확인
        academic_keywords = ["논문", "연구", "학술", "이론", "실험"]
        if any(keyword in query for keyword in academic_keywords):
            analysis["search_type"] = "scholar"
            analysis["priority"]["web"] = 0.4
            analysis["priority"]["rag"] = 0.6
        
        # 결과 캐싱
        self.cache_manager.set(cache_key, analysis)
        
        return analysis
    
    def execute_search(
        self,
        query: str,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """
        검색을 실행하고 결과를 통합합니다.
        
        Args:
            query (str): 검색 쿼리
            num_results (int): 각 검색 방식별 결과 수
            
        Returns:
            Dict[str, Any]: 통합된 검색 결과
                - web_results: 웹 검색 결과
                - rag_results: RAG 검색 결과
        """
        # 쿼리 분석
        analysis = self.analyze_query(query)
        
        results = {
            "query": query,
            "web_results": [],
            "rag_results": []
        }
        
        # 웹 검색 수행
        if analysis["requires_web_search"]:
            if analysis["search_type"] == "scholar":
                web_search = self.web_search_agent.scholar_search(
                    query,
                    num_results=num_results
                )
                results["web_results"] = web_search["results"]
            elif analysis["search_type"] == "combined":
                web_search = self.web_search_agent.combined_search(
                    query,
                    num_results=num_results
                )
                results["web_results"] = web_search["results"]["web"] + web_search["results"]["scholar"]
            else:
                web_search = self.web_search_agent.web_search(
                    query,
                    num_results=num_results
                )
                results["web_results"] = web_search["results"]
        
        # RAG 검색 수행
        if analysis["requires_rag"]:
            rag_search = self.rag_agent.process_query(
                query,
                top_k=num_results
            )
            results["rag_results"] = rag_search["results"]
        
        return results
    
    def rank_results(
        self,
        results: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        검색 결과에 가중치를 적용하고 순위를 매깁니다.
        
        Args:
            results (Dict[str, Any]): 검색 결과
            analysis (Dict[str, Any]): 쿼리 분석 결과
            
        Returns:
            List[SearchResult]: 순위가 매겨진 검색 결과 목록
        """
        ranked_results = []
        
        # 웹 검색 결과 가중치 적용
        for result in results["web_results"]:
            result.score *= analysis["priority"]["web"]
            ranked_results.append(result)
        
        # RAG 결과 가중치 적용
        for result in results["rag_results"]:
            result.score *= analysis["priority"]["rag"]
            ranked_results.append(result)
        
        # 점수 기준 정렬 (높은 점수가 더 관련성 높음)
        ranked_results.sort(key=lambda x: x.score, reverse=True)
        
        return ranked_results
    
    def summarize_search_results(self, query: str, results: List[SearchResult]) -> Dict[str, Any]:
        """
        검색 결과를 종합적으로 분석하고 요약합니다.
        
        Args:
            query (str): 원본 검색 쿼리
            results (List[SearchResult]): 검색 결과 목록
            
        Returns:
            Dict[str, Any]: 요약 결과
                - summary: AI가 생성한 요약
                - result_count: 결과 수
                - sources: 출처 목록
        """
        try:
            # 검색 결과 텍스트 구성
            result_texts = []
            for result in results:
                result_texts.append(f"제목: {result.title}\n내용: {result.snippet}\n출처: {result.source}")
            
            combined_results = "\n\n".join(result_texts)
            
            # GPT 프롬프트 구성
            prompt = f"""다음은 "{query}"에 대한 검색 결과들입니다. 이 결과들을 종합적으로 분석하여 다음 형식으로 요약해주세요:

검색 결과:
{combined_results}

요약 형식:
1. 핵심 내용 요약
2. 주요 발견점
3. 출처별 특징
4. 추가 조사가 필요한 부분

답변은 한국어로 작성해주세요."""

            # GPT API 호출
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 검색 결과를 분석하고 요약하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content
            
            return {
                "summary": summary,
                "result_count": len(results),
                "sources": list(set(result.source for result in results))
            }
            
        except Exception as e:
            print(f"요약 생성 중 오류 발생: {str(e)}")
            return {
                "summary": "요약을 생성하는 중 오류가 발생했습니다.",
                "result_count": len(results),
                "sources": list(set(result.source for result in results))
            }
    
    def process_query(
        self,
        query: str,
        num_results: int = 5
    ) -> SystemState:
        """
        사용자 쿼리를 처리하고 검색 결과를 반환합니다.
        
        Args:
            query (str): 사용자 검색 쿼리
            num_results (int): 반환할 결과 수
            
        Returns:
            SystemState: 시스템 상태 객체
        """
        try:
            # 쿼리 분석
            analysis = self.analyze_query(query)
            
            # 검색 실행
            results = self.execute_search(query, num_results)
            
            # 결과 순위 매기기
            ranked_results = self.rank_results(results, analysis)
            
            # 결과 요약
            summary = self.summarize_search_results(query, ranked_results)
            
            # 시스템 상태 반환
            return SystemState(
                query=query,
                search_results=ranked_results,
                summary=summary
            )
            
        except Exception as e:
            print(f"쿼리 처리 중 오류 발생: {str(e)}")
            return SystemState(
                query=query,
                error=str(e)
            )

"""
이 파일의 주요 역할:
1. 검색 에이전트 중앙 제어
2. 검색 결과 통합 및 순위 매기기
3. 검색 결과 요약

주요 기능:
- 쿼리 분석
- 웹/RAG 검색 조율
- 결과 순위 매기기
- AI 기반 요약

사용된 주요 기술:
- OpenAI GPT
- 캐싱 시스템
- 가중치 기반 순위 매기기
""" 