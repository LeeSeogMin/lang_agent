"""
시스템 설정 및 환경 변수 관리
이 모듈은 애플리케이션의 전역 설정과 환경 변수를 관리합니다.
API 키, 모델 설정, 경로 설정 등을 포함합니다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # 프로젝트 루트 디렉토리
CACHE_DIR = BASE_DIR / "cache"  # 캐시 파일 저장 디렉토리
DOCUMENTS_DIR = BASE_DIR / "documents"  # 업로드된 문서 저장 디렉토리

# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API 키
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Tavily Search API 키

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"  # 한국어 특화 임베딩 모델
EMBEDDING_DEVICE = "cpu"  # 임베딩 모델 실행 디바이스

# 벡터 DB 설정
VECTOR_DB_DIMENSION = 768  # ko-sroberta-multitask의 임베딩 차원

# 캐시 설정
CACHE_TTL = 3600  # 캐시 유효 시간 (초 단위, 1시간)

# 디렉토리 생성
CACHE_DIR.mkdir(exist_ok=True)  # 캐시 디렉토리 생성
DOCUMENTS_DIR.mkdir(exist_ok=True)  # 문서 디렉토리 생성

# 기타 설정
MAX_RESULTS = 10  # 기본 검색 결과 수
MODEL_NAME = "gpt-3.5-turbo"  # 기본 GPT 모델

"""
이 파일의 주요 역할:
1. 전역 설정 관리
2. 환경 변수 로드
3. 경로 설정
4. API 키 관리

주요 설정:
- 경로 설정: 캐시, 문서 저장소 등
- API 키: OpenAI, Tavily Search
- 모델 설정: 임베딩 모델, GPT 모델
- 시스템 설정: 캐시 TTL, 최대 결과 수 등

사용된 주요 기술:
- python-dotenv: 환경 변수 관리
- pathlib: 경로 관리
- os: 시스템 인터페이스
""" 