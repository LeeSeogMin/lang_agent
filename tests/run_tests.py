#!/usr/bin/env python
"""테스트 스크립트 실행기."""

import os
import sys
import unittest
import argparse

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_all_tests():
    """모든 테스트를 실행합니다."""
    # 테스트 디렉토리의 모든 테스트 파일을 찾습니다
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern="test_*.py")
    
    # 테스트 실행
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # 결과 반환 (테스트 실패가 있는 경우 1, 그렇지 않으면 0)
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_name):
    """특정 테스트만 실행합니다."""
    # 테스트 디렉토리의 모든 테스트 파일을 찾습니다
    test_loader = unittest.TestLoader()
    
    # 특정 테스트 패턴에 해당하는 테스트 실행
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern=f"test_{test_name}*.py")
    
    # 테스트 실행
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # 결과 반환 (테스트 실패가 있는 경우 1, 그렇지 않으면 0)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    # OpenAI API 키 확인
    if not os.environ.get("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        print("테스트를 실행하기 전에 API 키를 설정하세요.")
        sys.exit(1)
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="테스트 스크립트 실행기")
    parser.add_argument("--test", help="실행할 특정 테스트 이름 (예: data_analysis)")
    args = parser.parse_args()
    
    # 특정 테스트 또는 모든 테스트 실행
    if args.test:
        sys.exit(run_specific_test(args.test))
    else:
        sys.exit(run_all_tests()) 