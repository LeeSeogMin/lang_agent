"""
시스템 설정 및 환경 변수 관리
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = BASE_DIR / "cache"
DOCUMENTS_DIR = BASE_DIR / "documents"

# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
EMBEDDING_DEVICE = "cpu"

# 벡터 DB 설정
VECTOR_DB_DIMENSION = 768  # ko-sroberta-multitask의 임베딩 차원

# 캐시 설정
CACHE_TTL = 3600  # 1시간

# 디렉토리 생성
CACHE_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)

# 기타 설정
MAX_RESULTS = 10
MODEL_NAME = "gpt-3.5-turbo" 