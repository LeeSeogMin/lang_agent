# AI 검색 에이전트

이 프로젝트는 웹 검색과 문서 검색을 통합한 하이브리드 검색 시스템입니다. 다양한 포맷의 문서를 업로드하고 검색할 수 있으며, 웹 검색 결과와 함께 종합적인 정보를 제공합니다.

## 주요 기능

-   **하이브리드 검색**: 웹 검색과 로컬 문서 검색을 결합
-   **문서 관리**: PDF, DOCX, TXT 파일 지원 및 OCR 처리
-   **한국어 최적화**: 한국어 특화 임베딩 모델(SRoBERTa) 사용
-   **사용자 친화적 UI**: Streamlit 기반의 직관적인 인터페이스
-   **고성능 벡터 검색**: FAISS 기반의 빠른 임베딩 검색
-   **자동 요약**: 검색 결과의 AI 기반 자동 요약 기능

## 설치 방법

1. Python 3.11 이상이 필요합니다.
2. Poetry를 사용하여 의존성을 설치합니다:

```bash
# Poetry 설치 (필요한 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 프로젝트 의존성 설치
poetry install
```

3. 환경 변수 설정:

    - `.env.example`을 `.env`로 복사하고 필요한 API 키를 설정합니다.
    - 필요한 API 키: OpenAI, Tavily Search (또는 Google Custom Search), HuggingFace

4. (선택 사항) OCR 기능을 위해 Tesseract OCR을 설치합니다:
    - macOS: `brew install tesseract`
    - Ubuntu: `sudo apt-get install tesseract-ocr`
    - Windows: [Tesseract 설치 프로그램](https://github.com/UB-Mannheim/tesseract/wiki)

## 실행 방법

다음 두 가지 방법 중 하나로 애플리케이션을 실행할 수 있습니다:

### 1. Streamlit 직접 실행

#### macOS/Linux:

```bash
# 가상환경 활성화
poetry shell

# Streamlit 애플리케이션 실행
streamlit run src/ai_agent/ui.py
```

#### Windows:

```
# 가상환경 활성화 (명령 프롬프트)
poetry shell

# Streamlit 애플리케이션 실행
streamlit run src/ai_agent/ui.py
```

### 2. 실행 스크립트 사용

#### macOS/Linux:

```bash
# 가상환경 활성화
poetry shell

# 실행 스크립트로 애플리케이션 실행
python run_app.py
```

#### Windows:

```
# 가상환경 활성화 (명령 프롬프트)
poetry shell

# 실행 스크립트로 애플리케이션 실행
python run_app.py
```

실행 스크립트(`run_app.py`)는 경로 설정과 서버 포트 지정 등을 자동으로 처리하여 더 편리하게 애플리케이션을 시작할 수 있습니다.

브라우저에서 http://localhost:8501 로 접속하여 애플리케이션을 사용할 수 있습니다.

## 사용 방법

1. **문서 업로드**:

    - 사이드바의 '문서 업로드' 섹션에서 PDF, DOCX, TXT 파일을 업로드할 수 있습니다.
    - 파일 크기는 최대 10MB로 제한됩니다.
    - 업로드 시 자동으로 문서 처리 및 인덱싱이 진행됩니다.

2. **검색 수행**:

    - 메인 화면에서 검색어를 입력하고 '검색' 버튼을 클릭합니다.
    - 검색 결과는 관련성 점수와 함께 표시됩니다.
    - AI 요약 기능으로 검색 결과의 핵심 내용을 확인할 수 있습니다.

3. **문서 관리**:
    - 사이드바에서 업로드된 문서 목록을 확인할 수 있습니다.
    - 각 문서는 '삭제' 버튼을 통해 제거할 수 있습니다.

## 시스템 구조

```
lang_agent/
├── src/
│   └── ai_agent/
│       ├── __init__.py
│       ├── agents/             # 에이전트 구현 (컨트롤러, 웹 검색, RAG)
│       ├── config/             # 설정 파일
│       ├── core/               # 핵심 기능 구현 (문서 처리, 벡터 저장소)
│       ├── models/             # 데이터 모델 정의
│       ├── utils/              # 유틸리티 함수
│       └── ui.py               # Streamlit UI 구현
├── documents/                  # 문서 저장 디렉토리
├── cache/                      # 캐시 저장 디렉토리
├── temp/                       # 임시 파일 디렉토리
├── tests/                      # 테스트 코드
├── .env                        # 환경 변수
├── pyproject.toml              # 프로젝트 설정
└── README.md                   # 프로젝트 설명서
```

## 최신 업데이트

-   **동시성 처리 개선**: 문서 업로드 및 처리 시 발생하는 경쟁 상태(race condition) 문제 해결
-   **폼 기반 업로드**: Streamlit 폼을 활용한 안정적인 파일 업로드 구현
-   **OCR 지원 추가**: 텍스트 추출이 어려운 PDF에 대한 OCR 처리 기능
-   **세션 상태 관리 개선**: 안정적인 UI 상태 관리로 사용자 경험 향상

## 라이선스

MIT License
