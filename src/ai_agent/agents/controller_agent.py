"""
컨트롤러 에이전트 구현
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
        self.web_search_agent = web_search_agent or WebSearchAgent()
        self.rag_agent = rag_agent or RAGAgent()
        self.cache_manager = CacheManager()
        
        # OpenAI 클라이언트 초기화
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.openai_client = OpenAI(api_key=openai_api_key)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석 및 실행 계획 수립"""
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
        """검색 실행 및 결과 통합"""
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
        """검색 결과 순위 지정"""
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
        """검색 결과를 종합적으로 분석하고 요약"""
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
1. 핵심 주제: 검색 결과들의 중심 주제와 핵심 내용을 한 문단으로 설명
2. 주요 발견:
   - 가장 중요한 발견점들을 bullet point로 정리
   - 상반된 의견이나 다양한 관점이 있다면 이를 포함
3. 시간적 맥락: 기술/연구의 발전 과정이나 시간순 정리 (해당되는 경우)
4. 한계 및 과제: 현재 한계점이나 향후 과제 (언급된 경우)
5. 실용적 시사점: 실제 적용이나 활용 방안

전문가의 관점에서 객관적으로 분석하되, 한국어로 명확하게 작성해주세요."""

            # GPT로 요약 생성
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",  # 또는 사용 가능한 최신 모델
                messages=[
                    {"role": "system", "content": "당신은 연구 및 기술 문헌을 분석하고 종합하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            summary = response.choices[0].message.content
            
            return {
                "query": query,
                "summary": summary,
                "result_count": len(results),
                "sources": list(set(r.source for r in results))
            }
            
        except Exception as e:
            print(f"요약 생성 중 오류 발생: {str(e)}")
            return {
                "query": query,
                "summary": "요약을 생성할 수 없습니다.",
                "error": str(e)
            }
    
    def process_query(
        self,
        query: str,
        num_results: int = 5
    ) -> SystemState:
        """쿼리 처리 및 시스템 상태 반환"""
        try:
            # 쿼리 분석
            analysis = self.analyze_query(query)
            
            # 검색 실행
            search_results = self.execute_search(query, num_results)
            
            # 결과 순위 지정
            ranked_results = self.rank_results(search_results, analysis)
            
            # 결과가 없으면 빈 결과 반환 대신 기본 메시지 추가
            if not ranked_results:
                default_result = SearchResult(
                    title="검색 결과 없음",
                    snippet="문서를 업로드하고 다시 검색해보세요. 트랜스포머, 시계열과 같은 키워드로 검색하면 결과를 볼 수 있습니다.",
                    source="system",
                    score=1.0
                )
                ranked_results = [default_result]
            
            # 상위 결과만 선택
            top_results = ranked_results[:num_results]
            
            # 검색 결과 요약 생성
            summary = self.summarize_search_results(query, top_results)
            
            # 시스템 상태 생성
            state = SystemState(
                query=query,
                search_results=top_results,
                current_agent="controller",
                summary=summary
            )
            
            return state
            
        except Exception as e:
            print(f"쿼리 처리 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 상태 반환
            return SystemState(
                query=query,
                search_results=[],
                current_agent="controller",
                summary={"error": str(e)}
            ) 