"""RAG 기반 데이터 필터링 및 코드 생성 도구 테스트."""

import os
import sys
import asyncio
import unittest
from typing import Dict, Any
import tempfile

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backend.agents.data_analysis.tools import (
    filter_data_rag,
    generate_analysis_code,
    validate_data
)


class TestDataAnalysisTools(unittest.TestCase):
    """RAG 및 코드 생성 도구 테스트 클래스."""

    def setUp(self):
        """테스트 데이터 설정."""
        # 테스트 CSV 데이터
        self.csv_data = """Name,Age,Job,Salary
Alice,28,Engineer,75000
Bob,35,Manager,90000
Charlie,22,Designer,65000
David,42,Director,120000
Eva,31,Developer,80000
"""

        # 테스트 JSON 데이터
        self.json_data = """{
  "employees": [
    {"name": "Alice", "age": 28, "job": "Engineer", "salary": 75000},
    {"name": "Bob", "age": 35, "job": "Manager", "salary": 90000},
    {"name": "Charlie", "age": 22, "job": "Designer", "salary": 65000},
    {"name": "David", "age": 42, "job": "Director", "salary": 120000},
    {"name": "Eva", "age": 31, "job": "Developer", "salary": 80000}
  ]
}
"""

        # 테스트 텍스트 데이터
        self.text_data = """
Alice (28세)는 엔지니어로 일하며 연봉은 75,000달러이다.
Bob (35세)는 매니저로 일하며 연봉은 90,000달러이다.
Charlie (22세)는 디자이너로 일하며 연봉은 65,000달러이다.
David (42세)는 디렉터로 일하며 연봉은 120,000달러이다.
Eva (31세)는 개발자로 일하며 연봉은 80,000달러이다.
"""

    def test_validate_data(self):
        """데이터 유효성 검사 도구 테스트."""
        # 비동기 함수를 동기적으로 실행하기 위한 래퍼
        result = asyncio.run(validate_data(self.csv_data))
        
        self.assertTrue(result["is_valid"])
        self.assertEqual(result.get("detected_format"), "csv")
        
        # JSON 데이터 테스트
        result = asyncio.run(validate_data(self.json_data))
        self.assertTrue(result["is_valid"])
        self.assertEqual(result.get("detected_format"), "json")
        
        # 텍스트 데이터 테스트
        result = asyncio.run(validate_data(self.text_data))
        self.assertTrue(result["is_valid"])
        self.assertEqual(result.get("detected_format"), "text")

    def test_filter_data_rag(self):
        """RAG 기반 데이터 필터링 도구 테스트."""
        # CSV 데이터에서 특정 조건 필터링
        result = asyncio.run(filter_data_rag(
            data=self.csv_data,
            query="30세 이상인 사람들만 보여주세요",
            data_format="csv"
        ))
        
        self.assertNotIn("error", result)
        self.assertIn("filtered_result", result)
        
        # 필터링 결과에 'Bob', 'David', 'Eva'가 포함되어 있는지 확인
        filtered_data = result["filtered_result"]
        found_names = any(name in filtered_data for name in ["Bob", "David", "Eva"])
        self.assertTrue(found_names, "필터링 결과에 30세 이상인 사람들이 포함되어 있지 않습니다.")
        
        # 원본 행 수가 메타데이터에 정확히 포함되어 있는지 확인
        self.assertEqual(result["metadata"].get("original_rows"), 5)
    
    def test_generate_analysis_code(self):
        """코드 생성 도구 테스트."""
        # CSV 데이터에 대한 분석 코드 생성
        result = asyncio.run(generate_analysis_code(
            data=self.csv_data,
            task="나이별 평균 급여를 계산하고 시각화해주세요",
            code_type="pandas"
        ))
        
        self.assertNotIn("error", result)
        self.assertIn("generated_code", result)
        
        # 생성된 코드에 'pandas'와 'mean'이 포함되어 있는지 확인
        generated_code = result["generated_code"]
        self.assertTrue("import pandas" in generated_code, "생성된 코드에 pandas import가 없습니다.")
        self.assertTrue("mean()" in generated_code, "생성된 코드에 평균 계산 함수가 없습니다.")
        
        # 시각화 옵션이 포함되어 있는지 확인
        self.assertTrue(any(viz_lib in generated_code for viz_lib in ["matplotlib", "plot", "seaborn"]),
                       "생성된 코드에 시각화 라이브러리가 없습니다.")


class TestDataAnalysisGraph(unittest.TestCase):
    """데이터 분석 그래프 테스트 클래스."""
    
    def setUp(self):
        """테스트 데이터 및 그래프 설정."""
        from src.backend.agents.data_analysis.graph import create_data_analysis_graph
        from src.backend.agents.data_analysis.schemas import DataAnalysisState
        
        # 테스트 데이터
        self.test_data = """Name,Age,Job,Salary
Alice,28,Engineer,75000
Bob,35,Manager,90000
Charlie,22,Designer,65000
"""
        
        # 그래프 및 초기 상태 생성
        self.graph = create_data_analysis_graph()
        self.state = DataAnalysisState(
            document_content=self.test_data,
            data=self.test_data
        )
    
    def test_rag_filtering_node(self):
        """RAG 필터링 노드 테스트."""
        # 필터링 활성화 및 쿼리 설정
        self.state.apply_rag_filtering = True
        self.state.filter_query = "30세 이상인 사람을 찾아주세요"
        
        # 비동기 함수를 동기적으로 실행
        from src.backend.agents.data_analysis.graph import filter_data_with_rag
        result = asyncio.run(filter_data_with_rag(self.state))
        
        # 결과 확인
        self.assertTrue("filtered_data_result" in result)
        self.assertTrue("Bob" in result["filtered_data_result"], 
                       "RAG 필터링 결과에 30세 이상인 'Bob'이 포함되어 있지 않습니다.")
        self.assertTrue(result.get("filtered_data_success", False), 
                       "RAG 필터링이 성공하지 못했습니다.")
    
    def test_code_generation_node(self):
        """코드 생성 노드 테스트."""
        # 코드 생성 활성화 및 작업 설정
        self.state.apply_code_generation = True
        self.state.code_generation_task = "Job별 평균 급여를 계산하세요"
        self.state.code_type = "pandas"
        
        # 비동기 함수를 동기적으로 실행
        from src.backend.agents.data_analysis.graph import generate_analysis_code_node
        result = asyncio.run(generate_analysis_code_node(self.state))
        
        # 결과 확인
        self.assertTrue("generated_code_result" in result)
        self.assertTrue("generated_code" in result)
        self.assertTrue(result.get("generated_code_success", False), 
                       "코드 생성이 성공하지 못했습니다.")
        
        # 생성된 코드가 적절한지 확인
        generated_code = result.get("generated_code", "")
        self.assertTrue("import pandas" in generated_code)
        self.assertTrue("groupby" in generated_code)
        self.assertTrue("Job" in generated_code)


if __name__ == "__main__":
    # OpenAI API 키가 설정되어 있는지 확인
    if not os.environ.get("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        print("테스트를 실행하기 전에 API 키를 설정하세요.")
        sys.exit(1)
    
    unittest.main() 