# LangGraph 기반 멀티에이전트 시스템 (Streamlit UI)

## 주요 구성
- **Streamlit UI**: 웹 기반 대화, 문서 업로드, 데이터 분석, 세션 관리
- **Orchestrator**: 대화 흐름/에이전트 라우팅 관리
- **Knowledge Agent**: 문서 RAG(파싱, 청킹, 임베딩, ChromaDB 저장)
- **Data Analysis Agent**: 데이터 분석(파일/텍스트)

## 주요 기능
- 문서 업로드 후 "문서 파싱/분석" 버튼으로 RAG 파이프라인 실행 (PDF, TXT, MD 지원)
- 데이터 분석: 파일 업로드(.csv, .txt, .md) 또는 텍스트 입력 지원
- 대화 세션 관리: Start New Chat로 새로운 세션 생성, 세션별 대화 기록
- 모든 기능은 Streamlit 웹 UI에서 통합 제공
- Claude 3.7 Sonnet 모델 기본 사용 (Anthropic API)

## 실행 방법
1. 의존성 설치
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```
2. 환경 설정
   ```bash
   # .env 파일 생성
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key  # 임베딩용
   ```
3. 실행
   ```bash
   streamlit run src/backend/app.py
   ```
4. 브라우저에서 접속: [http://localhost:8501](http://localhost:8501)

## 사용법
- **문서 업로드 & 분석**: 사이드바에서 파일 업로드 → "문서 파싱/분석" 클릭 → 청크/벡터화 결과 확인
- **데이터 분석**: 파일 업로드 또는 텍스트 입력 → "Analyze Data" 클릭
- **대화 세션**: "Start New Chat" 클릭 → 세션별 대화 진행

---

# LangGraph Multi-Agent System (Streamlit UI)

## Main Components
- **Streamlit UI**: Web chat, document upload, data analysis, session management
- **Orchestrator**: Conversation flow & agent routing
- **Knowledge Agent**: Document RAG (parsing, chunking, embedding, ChromaDB)
- **Data Analysis Agent**: Data analysis (file/text)

## Features
- Document upload & RAG pipeline (PDF, TXT, MD)
- Data analysis: file upload (.csv, .txt, .md) or text input
- Session management: Start New Chat, per-session history
- All-in-one Streamlit web UI
- Uses Claude 3.7 Sonnet model (Anthropic API)

## How to Run
1. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```
2. Environment Setup
   ```bash
   # Create .env file
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key  # for embeddings
   ```
3. Run
   ```bash
   streamlit run src/backend/app.py
   ```
4. Open [http://localhost:8501](http://localhost:8501) in your browser

## Usage
- **Document Upload & Analysis**: Upload file → Click "문서 파싱/분석" → See chunk/vectorization result
- **Data Analysis**: Upload file or enter text → Click "Analyze Data"
- **Chat Sessions**: Click "Start New Chat" → Chat per session

## 최근 추가된 기능

### RAG 기반 데이터 필터링

Neptune.ai 블로그 글에서 제안된 방법을 활용하여 구현된 이 기능은 자연어 쿼리를 사용하여 구조화된 데이터에서 관련 정보를 추출합니다.

- **작동 방식**:
  - 데이터를 청크로 분할하고 벡터 데이터베이스에 저장
  - 사용자의 자연어 쿼리와 관련성이 높은 청크를 검색
  - LLM을 사용하여 검색된 데이터 세그먼트에서 관련 정보 추출 및 요약
  - CSV 데이터의 경우 pandas DataFrame 에이전트를 활용해 직접 쿼리 실행

- **사용 예**:
  - "30세 이상인 직원들만 보여주세요"
  - "마케팅 부서의 평균 급여는 얼마인가요?"
  - "급여가 가장 높은 상위 3명은 누구인가요?"

### LLM 기반 코드 생성

복잡한 데이터 분석을 위한 Python 코드를 자동으로 생성하는 기능으로, 사용자가 자연어로 설명한 작업을 수행하는 실행 가능한 코드를 제공합니다.

- **지원하는 코드 유형**:
  - **pandas**: 데이터 조작 및 분석을 위한 코드
  - **numpy**: 수치 연산 및 고급 계산을 위한 코드
  - **matplotlib/seaborn**: 데이터 시각화를 위한 코드

- **사용 예**:
  - "부서별 평균 급여를 계산하고 막대 그래프로 시각화해주세요"
  - "연령대별 직원 수를 계산하고 원형 차트로 표시해주세요"
  - "급여와 연령 간의 상관관계를 분석해주세요"

## 사용 방법

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/multi_agent_system.git
cd multi_agent_system

# 의존성 설치
pip install -r requirements.txt
```

### 실행

```bash
# Streamlit 앱 실행
python -m src.backend.app_streamlit
```

### 데이터 분석 기능 사용

1. **파일 업로드**: CSV, JSON 또는 텍스트 파일을 업로드
2. **고급 분석 옵션 설정**:
   - RAG 기반 데이터 필터링 활성화 및 쿼리 입력
   - 코드 생성 활성화 및 작업 설명 입력
3. **분석 시작**: "지금 분석하기" 버튼 클릭

## 테스트

테스트 실행:

```bash
# 모든 테스트 실행
python tests/run_tests.py

# 특정 테스트만 실행
python tests/run_tests.py --test data_analysis
```

## 라이센스

MIT

