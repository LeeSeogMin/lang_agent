가이드는 여러 시행착오를 거쳐 매뉴얼 형식으로 작성함. 만일 다른 프로젝트 작업을 할 경우에 이 매뉴얼을 참조하여 유사 형식으로 개발가이드를 작성하라고 하면 됩니다. 

## Phase 1: 프로젝트 준비 및 기초 설정

### 1-1: 프로젝트 초기 설정 및 Poetry 환경 구성

프로젝트 저장소를 초기화하고 Poetry를 사용하여 가상환경과 의존성을 설정한다.

```
다음 구조로 프로젝트 디렉토리를 생성하고 Poetry 환경을 구성해주세요:

1. 프로젝트 구조 생성:
   lang_agent/
   ├── src/
   │   └── ai_agent/
   │       └── __init__.py
   ├── tests/
   │   └── __init__.py
   ├── .gitignore
   └── README.md

2. Poetry 초기화 및 설정:
   - poetry init 실행하여 pyproject.toml 생성
   - Python 3.11 버전 지정
   - poetry config virtualenvs.in-project true 설정

3. 필수 패키지 설치:
   - langchain, langgraph, langchain-openai, langchain-community
   - faiss-cpu, streamlit
   - google-api-python-client, google-auth-httplib2, google-auth-oauthlib
   - sentence-transformers, huggingface_hub
   - pandas, numpy, requests, beautifulsoup4
   - python-dotenv, pypdf2, docx

4. 개발 도구 설치:
   - pytest, black, isort, mypy

5. 환경 파일 생성:
   - .env 파일 생성하여 API 키 템플릿 작성
   - .gitignore에 .env 추가
   - README.md에 프로젝트 설명, 사용방법 설명 (한글)

poetry install 명령으로 환경 설정을 완료하고 정상 작동 여부를 확인해주세요.

패키지 설치시 진행상황을 보여준다. 

패키지 설치, 폴더와 파일의 생성은 command prompt를 사용한다. 
```

필수 패키지는 AI를 통해 확인함: 프로젝트 목적을 설명하고 필요한 패키지를 질문하여 응답받음

### 1-2: 시스템 아키텍처 설계

멀티에이전트 시스템의 전체 아키텍처와 컴포넌트 간 상호작용을 설계한다.

```
LangGraph 기반 멀티에이전트 시스템의 아키텍처를 다음과 같이 설계해주세요:
필요한 디렉토리를 생성하고 생성된 파일들을 저장한다. 

1. 시스템 컴포넌트 정의 및 책임 할당:
   - 총괄 에이전트 (Controller Agent): 질의 분석 및 작업 위임
   - 웹 검색 에이전트 (Web Search Agent): Google API 통합
   - RAG 에이전트 (RAG Agent): 로컬 문서 검색
   - 문서 관리자 (Document Manager): 문서 파싱/저장
   - 사용자 인터페이스 (Streamlit UI): 입출력 관리

2. 책임 할당 매트릭스 작성:
   | Component      | Primary Role       | Responsibilities                    |
   |---------------|-------------------|-------------------------------------|
   | Controller    | 질의 분석, 위임    | 질의 분류, 라우팅, 결과 통합         |
   | Web Search    | 외부 검색         | Google API 통합, 결과 파싱          |
   | RAG Agent     | 로컬 문서 검색    | 벡터 DB 통합, 임베딩 검색           |
   | Document Mgr  | 문서 파싱/저장    | 파일 파싱, 메타데이터 관리          |
   | UI           | 사용자 인터페이스  | 입출력 처리, 파일 업로드            |

3. 컴포넌트 간 데이터 흐름 설계:
   - 공통 상태 객체(SystemState) 정의
   - 상태 지속성 전략 (단기/장기 메모리)
   - 데이터 변환 규칙 정의

4. LangGraph 기반 에이전트 그래프 설계:
   - 노드(상태/액션) 및 엣지(전이) 정의
   - 진입/종료 지점 및 조건부 로직
   - 오류 처리 경로 설계

5. 모듈 간 인터페이스 정의:
   - 함수 시그니처 및 입출력 계약
   - TypedDict 구조 정의
   - 오류 처리 프로토콜

6. UML 클래스 다이어그램 작성:
   - 모든 시스템 클래스 및 관계 표현
   - 상속, 구성, 의존성 패턴 포함

각 컴포넌트는 독립적으로 테스트 가능하도록 명확한 인터페이스를 가져야 합니다.

```

## Phase 2: 핵심 기반 구현

### 2-1: 한글 임베딩 모델 통합

HuggingFace에서 한글 전용 임베딩 모델을 시스템에 통합한다.

```
한글 임베딩 모델 'jhgan/ko-sroberta-multitask'를 다음과 같이 통합해주세요:
새로운 디렉토리가 필요한 경우 생성하고 생성된 파일들을 저장한다. 

1. 모델 로딩 함수 구현:
   from langchain.embeddings import HuggingFaceEmbeddings

   def load_korean_embedding_model():
       embedding = HuggingFaceEmbeddings(
           model_name="jhgan/ko-sroberta-multitask",
           model_kwargs={'device': 'cpu'},
           encode_kwargs={'normalize_embeddings': True}
       )
       return embedding

2. API 인증 처리:
   import os
   from dotenv import load_dotenv

   def load_api_credentials():
       load_dotenv()
       hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
       return hf_api_key

3. 텍스트 임베딩 함수:
   def embed_text(embedding_model, text):
       return embedding_model.embed_query(text)

4. 모델 캐싱 메커니즘:
   - 싱글톤 패턴 활용한 모델 인스턴스 관리
   - 메모리 효율적인 캐싱 전략

5. 성능 최적화:
   - 배치 처리 구현
   - 임베딩 차원 및 품질 검증
   - 메모리 사용량 모니터링

한글 텍스트 샘플로 임베딩 생성을 테스트하고, 유사 텍스트 간 코사인 유사도를 측정해주세요.

```

### 2-2: Faiss 벡터 DB 구현

문서 임베딩을 저장하고 검색할 수 있는 Faiss 벡터 데이터베이스를 구현한다.

```
Faiss 벡터 데이터베이스를 구현해주세요:
새로운 디렉토리가 필요한 경우 생성하고 생성된 파일들을 저장한다. 

1. FaissVectorStore 클래스 구현:
   import faiss
   import numpy as np
   import pickle

   class FaissVectorStore:
       def __init__(self, embedding_dim=768):
           self.index = faiss.IndexFlatL2(embedding_dim)
           self.documents = []
           self.embedding_dim = embedding_dim

       def add_documents(self, documents, embeddings):
           if len(documents) == 0:
               return

           doc_ids = list(range(len(self.documents),
                               len(self.documents) + len(documents)))
           self.documents.extend(documents)

           embeddings_np = np.array(embeddings).astype('float32')
           self.index.add(embeddings_np)
           return doc_ids

       def search(self, query_embedding, top_k=5):
           query_embedding_np = np.array([query_embedding]).astype('float32')
           distances, indices = self.index.search(query_embedding_np, top_k)

           results = []
           for i, idx in enumerate(indices[0]):
               if idx < len(self.documents) and idx >= 0:
                   results.append({
                       'document': self.documents[idx],
                       'score': float(distances[0][i])
                   })
           return results

       def save(self, filepath):
           with open(filepath + '.documents', 'wb') as f:
               pickle.dump(self.documents, f)
           faiss.write_index(self.index, filepath + '.index')

       def load(self, filepath):
           with open(filepath + '.documents', 'rb') as f:
               self.documents = pickle.load(f)
           self.index = faiss.read_index(filepath + '.index')

2. 문서 메타데이터 관리 기능:
   - 문서 ID, 소스, 커스텀 필드 연계
   - 메타데이터 기반 필터링

3. 성능 최적화:
   - 적절한 Faiss 인덱스 타입 선택 (IVF, HNSW)
   - 검색 파라미터 튜닝

4. 테스트:
   - 샘플 문서로 벡터 DB 생성/저장/로드
   - 유사도 검색 정확도 검증
   - 대량 문서 처리 성능 측정
   
  생성된 파일들이 저장되었는지 확인한다. 

```

### 2-3: 문서 파싱 및 관리 모듈 구현

다양한 형식의 문서를 파싱하고 관리하는 모듈을 구현한다.

```
문서 파싱 및 관리 모듈을 구현해주세요:
새로운 디렉토리가 필요한 경우 생성하고 생성된 파일들을 저장한다. 

1. DocumentParser 클래스 구현:
   from PyPDF2 import PdfReader
   import docx
   import os

   class DocumentParser:
       def parse(self, file_path):
           ext = os.path.splitext(file_path)[1].lower()
           if ext == '.pdf':
               return self.parse_pdf(file_path)
           elif ext == '.txt':
               return self.parse_txt(file_path)
           elif ext == '.docx':
               return self.parse_docx(file_path)
           else:
               raise ValueError(f"지원하지 않는 파일 형식: {ext}")

       def parse_pdf(self, file_path):
           text = ""
           with open(file_path, 'rb') as file:
               reader = PdfReader(file)
               for page in reader.pages:
                   text += page.extract_text() + "\n"
           return self.chunk_text(text)

       def parse_txt(self, file_path):
           with open(file_path, 'r', encoding='utf-8') as file:
               text = file.read()
           return self.chunk_text(text)

       def parse_docx(self, file_path):
           doc = docx.Document(file_path)
           text = "\n".join([para.text for para in doc.paragraphs])
           return self.chunk_text(text)

       def chunk_text(self, text, chunk_size=1000, overlap=200):
           chunks = []
           start = 0
           text_len = len(text)

           while start < text_len:
               end = min(start + chunk_size, text_len)
               chunks.append(text[start:end])
               start += (chunk_size - overlap)

           return chunks

2. 한글 최적화:
   - 한글 토크나이저 적용
   - 문장 경계 인식 개선
   - 인코딩 문제 처리

3. 메타데이터 추출:
   - 제목, 저자, 날짜 등 추출
   - 문서 형식별 메타데이터 스키마

4. 청킹 전략:
   - 고정 크기 청킹
   - 의미 기반 청킹
   - 하이브리드 접근

5. 테스트:
   - 다양한 한글 문서 파싱
   - 청킹 품질 검증
   - 대용량 파일 처리 성능

```

## Phase 3: 외부 서비스 통합

### 3-1: Google Custom Search API 통합

구글 일반검색과 학술검색을 위한 Google Custom Search API를 통합한다.

```
Google Custom Search API를 다음과 같이 통합해주세요:
새로운 디렉토리가 필요한 경우 생성하고 생성된 파일들을 저장한다. 
GOOGLE_API_KEY, GOOGLE_CSE_ID_GENERAL, GOOGLE_CSE_ID_SCHOLAR를 연결해야 한다.

from googleapiclient.discovery import build
import os

class GoogleSearchAPI:
    def __init__(self, api_key=None, cse_id_general=None, cse_id_scholar=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id_general = cse_id_general or os.getenv("GOOGLE_CSE_ID_GENERAL")
        self.cse_id_scholar = cse_id_scholar or os.getenv("GOOGLE_CSE_ID_SCHOLAR")
        if not self.api_key or not self.cse_id_general or not self.cse_id_scholar:
            raise ValueError("Google API 키와 두 종류의 CSE ID가 필요합니다.")

    def search(self, query, num_results=10, search_type="web"):
        service = build("customsearch", "v1", developerKey=self.api_key)
        if search_type == "scholar":
            cse_id = self.cse_id_scholar
        else:
            cse_id = self.cse_id_general
        params = {
            "q": query,
            "cx": cse_id,
            "num": num_results
        }
        if search_type == "scholar":
            params["as_sitesearch"] = "scholar.google.com"
        result = service.cse().list(**params).execute()
        search_results = []
        if "items" in result:
            for item in result["items"]:
                search_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
        return search_results

    def web_search(self, query, num_results=10):
        return self.search(query, num_results, "web")

    def scholar_search(self, query, num_results=10):
        return self.search(query, num_results, "scholar")

2. API 설정:
   - Google Cloud Console에서 프로젝트 생성
   - Custom Search API 활성화
   - API 키 생성 및 CSE ID 설정

3. 에러 처리:
   - API 할당량 초과 처리
   - 네트워크 오류 재시도 메커니즘
   - Exponential backoff 구현

4. 결과 캐싱:
   - TTL 기반 캐시 구현
   - 중복 쿼리 최적화

5. 테스트:
   - 한글 쿼리 검색 결과 검증
   - 일반/학술 검색 결과 비교
   - API 제한 사항 확인

```

## Phase 4: 에이전트 구현

### 4-1: 총괄 에이전트(Controller Agent) 구현

사용자 질의를 분석하고 작업을 적절한 에이전트에 위임하는 총괄 에이전트를 구현한다.

```
LangGraph 기반 총괄 에이전트를 구현해주세요:

1. ControllerAgent 클래스 구현:
   from langgraph.graph import StateGraph, END
   from langchain.chat_models import ChatOpenAI
   from langchain.prompts import ChatPromptTemplate

   class ControllerAgent:
       def __init__(self):
           self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")
           self.workflow = self._create_workflow()

       def _create_workflow(self):
           workflow = StateGraph()

           # 노드 추가
           workflow.add_node("분석", self._analyze_query)
           workflow.add_node("웹검색", self._delegate_to_web_search)
           workflow.add_node("RAG검색", self._delegate_to_rag)
           workflow.add_node("문서관리", self._handle_document_management)
           workflow.add_node("응답생성", self._generate_response)

           # 엣지 설정
           workflow.add_edge("분석", "웹검색")
           workflow.add_edge("분석", "RAG검색")
           workflow.add_edge("분석", "문서관리")
           workflow.add_edge("웹검색", "응답생성")
           workflow.add_edge("RAG검색", "응답생성")
           workflow.add_edge("문서관리", "응답생성")
           workflow.add_edge("응답생성", END)

           # 조건부 엣지
           workflow.add_conditional_edges(
               "분석",
               self._route_query,
               {
                   "웹검색": "웹검색",
                   "RAG검색": "RAG검색",
                   "문서관리": "문서관리",
                   "모두": ["웹검색", "RAG검색"]
               }
           )

           return workflow.compile()

       def _analyze_query(self, state):
           query = state["query"]
           prompt = ChatPromptTemplate.from_template(
               """사용자의 질의를 분석하고 어떤 작업이 필요한지 판단하세요:
               질의: {query}

               이 질의는 다음 중 어떤 유형인가요?
               1. 웹 검색이 필요한 질의
               2. 로컬 문서(RAG) 검색이 필요한 질의
               3. 문서 관리 작업(저장, 삭제 등)이 필요한 질의
               4. 웹 검색과 RAG 검색 모두 필요한 질의

               분석 결과를 JSON 형식으로 제공하세요."""
           )
           response = self.llm.invoke(prompt.format(query=query))
           return {"query": query, "analysis": response.content}

       def _route_query(self, state):
           analysis = state["analysis"]
           if "웹 검색" in analysis and "RAG 검색" in analysis:
               return "모두"
           elif "웹 검색" in analysis:
               return "웹검색"
           elif "RAG 검색" in analysis:
               return "RAG검색"
           elif "문서 관리" in analysis:
               return "문서관리"
           else:
               return "웹검색"

       def process_query(self, query):
           result = self.workflow.invoke({"query": query})
           return result["response"]

2. 상태 관리:
   - SystemState 정의
   - 상태 추적 및 로깅

3. 에러 처리:
   - 복구 메커니즘
   - 폴백 전략

4. 테스트:
   - 다양한 질의 유형 테스트
   - 라우팅 정확도 검증
   - 오류 처리 시나리오

```

### 4-2: 웹 검색 에이전트(Web Search Agent) 구현

Google Custom Search API를 활용하여 웹 검색을 수행하는 에이전트를 구현한다.

```
웹 검색 에이전트를 구현해주세요:

1. WebSearchAgent 클래스 구현:
   from langchain.chat_models import ChatOpenAI
   from langchain.prompts import ChatPromptTemplate

   class WebSearchAgent:
       def __init__(self, google_search_api):
           self.google_search_api = google_search_api
           self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")

       def search(self, query, search_type="web", num_results=5):
           if search_type == "web":
               results = self.google_search_api.web_search(query, num_results)
           elif search_type == "scholar":
               results = self.google_search_api.scholar_search(query, num_results)
           else:
               raise ValueError(f"지원하지 않는 검색 유형: {search_type}")
           return results

       def determine_search_type(self, query):
           prompt = ChatPromptTemplate.from_template(
               """다음 질의가 일반 웹 검색이 적합한지, 학술 검색이 적합한지 판단하세요:
               질의: {query}

               '일반' 또는 '학술' 중 하나로만 응답하세요."""
           )
           response = self.llm.invoke(prompt.format(query=query))
           if "학술" in response.content:
               return "scholar"
           else:
               return "web"

       def process_query(self, query):
           search_type = self.determine_search_type(query)
           search_results = self.search(query, search_type)

           if not search_results:
               return {"results": [], "summary": "검색 결과가 없습니다."}

           summary = self.summarize_results(query, search_results)

           return {
               "results": search_results,
               "summary": summary,
               "search_type": search_type
           }

       def summarize_results(self, query, results):
           results_text = "\n\n".join([
               f"제목: {r['title']}\n링크: {r['link']}\n내용: {r['snippet']}"
               for r in results
           ])

           prompt = ChatPromptTemplate.from_template(
               """다음은 '{query}'에 대한 검색 결과입니다:

               {results_text}

               이 검색 결과를 종합하여 질의에 대한 답변을 한글로 작성하세요.
               정보의 출처를 명확히 언급하고, 검색 결과에 없는 내용은 추가하지 마세요."""
           )

           response = self.llm.invoke(
               prompt.format(query=query, results_text=results_text)
           )
           return response.content

2. 검색 결과 처리:
   - 결과 필터링 및 랭킹
   - 관련성 점수 계산
   - 중복 제거

3. 요약 품질 개선:
   - 프롬프트 엔지니어링
   - 출처 인용 형식 표준화

4. 테스트:
   - 검색 유형 자동 결정 정확도
   - 요약 품질 평가
   - 다양한 쿼리 처리

```

### 4-3: RAG 에이전트(RAG Agent) 구현

Faiss 벡터 DB를 활용하여 로컬 문서 검색을 수행하는 RAG 에이전트를 구현한다.

```
RAG 에이전트를 구현해주세요:

1. RAGAgent 클래스 구현:
   from langchain.chat_models import ChatOpenAI
   from langchain.prompts import ChatPromptTemplate

   class RAGAgent:
       def __init__(self, vector_store, embedding_model):
           self.vector_store = vector_store
           self.embedding_model = embedding_model
           self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")

       def search(self, query, top_k=5):
           query_embedding = self.embedding_model.encode(query)
           search_results = self.vector_store.search(query_embedding, top_k=top_k)
           return search_results

       def process_query(self, query):
           search_results = self.search(query)

           if not search_results:
               return {
                   "results": [],
                   "answer": "로컬 문서에서 관련 정보를 찾을 수 없습니다."
               }

           answer = self.generate_answer(query, search_results)

           return {
               "results": search_results,
               "answer": answer
           }

       def generate_answer(self, query, search_results):
           context = "\n\n".join([
               f"문서 {i+1}:\n{result['document']}"
               for i, result in enumerate(search_results)
           ])

           prompt = ChatPromptTemplate.from_template(
               """다음은 사용자의 질의에 관련된 문서 내용입니다:

               {context}

               위 문서 내용을 바탕으로 다음 질의에 답변하세요:
               질의: {query}

               문서에 포함된 정보만 사용하여 답변하세요.
               문서에 없는 내용은 추가하지 마세요.
               답변은 한글로 작성하고, 필요한 경우 문서의 어느 부분에서
               정보를 얻었는지 언급하세요."""
           )

           response = self.llm.invoke(
               prompt.format(context=context, query=query)
           )
           return response.content

2. 검색 품질 개선:
   - 하이브리드 검색 (벡터 + 키워드)
   - 리랭킹 알고리즘 적용
   - 다양성 샘플링

3. 컨텍스트 조립:
   - 토큰 제한 고려
   - 우선순위 기반 선택
   - 메타데이터 포함

4. 답변 생성:
   - 출처 인용 시스템
   - 일관된 인용 스타일
   - 검색 실패 시 폴백

5. 테스트:
   - 검색 관련성 평가
   - 답변 정확도 측정
   - 대용량 문서 성능

```

### 4-4: 문서 관리 기능 구현

사용자가 문서를 업로드하고 관리할 수 있는 기능을 구현한다.

```
문서 관리 기능을 구현해주세요:

1. DocumentManager 클래스 구현:
   import os
   import shutil
   import uuid
   from datetime import datetime

   class DocumentManager:
       def __init__(self, document_parser, vector_store, embedding_model,
                    storage_dir="./documents"):
           self.document_parser = document_parser
           self.vector_store = vector_store
           self.embedding_model = embedding_model
           self.storage_dir = storage_dir

           os.makedirs(storage_dir, exist_ok=True)
           self.document_metadata = {}

       def upload_document(self, file_path, metadata=None):
           _, ext = os.path.splitext(file_path)
           doc_id = str(uuid.uuid4())
           save_path = os.path.join(self.storage_dir, f"{doc_id}{ext}")

           shutil.copy2(file_path, save_path)

           self.document_metadata[doc_id] = {
               "original_filename": os.path.basename(file_path),
               "path": save_path,
               "extension": ext,
               "uploaded_at": datetime.now().isoformat(),
               "custom_metadata": metadata or {}
           }

           return doc_id

       def index_document(self, doc_id):
           metadata = self.document_metadata.get(doc_id)
           if not metadata:
               raise ValueError(f"문서 ID {doc_id}를 찾을 수 없습니다.")

           chunks = self.document_parser.parse(metadata["path"])
           embeddings = [self.embedding_model.encode(chunk) for chunk in chunks]

           chunk_ids = self.vector_store.add_documents(
               [{
                   "content": chunk,
                   "doc_id": doc_id,
                   "chunk_index": i,
                   "metadata": metadata
               } for i, chunk in enumerate(chunks)],
               embeddings
           )

           metadata["indexed"] = True
           metadata["chunk_count"] = len(chunks)
           metadata["chunk_ids"] = chunk_ids

           return len(chunks)

       def delete_document(self, doc_id):
           metadata = self.document_metadata.get(doc_id)
           if not metadata:
               raise ValueError(f"문서 ID {doc_id}를 찾을 수 없습니다.")

           if os.path.exists(metadata["path"]):
               os.remove(metadata["path"])

           del self.document_metadata[doc_id]
           # TODO: 벡터 DB에서 관련 청크 삭제

           return True

       def list_documents(self):
           return self.document_metadata

2. 파일 처리:
   - 다중 파일 형식 지원
   - 에러 처리 및 복구
   - 트랜잭션 관리

3. 인덱싱 시스템:
   - 자동 인덱싱
   - 배치 처리
   - 진행 상황 추적

4. 메타데이터 관리:
   - 버전 관리
   - 접근 권한
   - 관계 설정

5. 테스트:
   - 파일 업로드/삭제
   - 인덱싱 정확성
   - 대용량 처리

```

## Phase 5: 시스템 통합 및 최적화

### 5-1: 에이전트 간 병렬 처리 및 상호작용 구현

여러 에이전트가 병렬적으로 작업을 수행하고 결과를 통합하는 메커니즘을 구현한다.

```
병렬 에이전트 실행기를 구현해주세요:

1. ParallelAgentExecutor 클래스 구현:
   import asyncio
   from typing import Dict, List, Any

   class ParallelAgentExecutor:
       def __init__(self, controller_agent, web_search_agent, rag_agent):
           self.controller_agent = controller_agent
           self.web_search_agent = web_search_agent
           self.rag_agent = rag_agent

       async def execute_web_search(self, query):
           return self.web_search_agent.process_query(query)

       async def execute_rag_search(self, query):
           return self.rag_agent.process_query(query)

       async def execute_parallel(self, query, agents_to_run):
           tasks = []

           if "web" in agents_to_run:
               tasks.append(self.execute_web_search(query))

           if "rag" in agents_to_run:
               tasks.append(self.execute_rag_search(query))

           results = await asyncio.gather(*tasks)

           mapped_results = {}
           for i, agent_type in enumerate(
               [a for a in ["web", "rag"] if a in agents_to_run]
           ):
               mapped_results[agent_type] = results[i]

           return mapped_results

       def integrate_results(self, query, results):
           web_result = results.get("web", {}).get("summary", "")
           rag_result = results.get("rag", {}).get("answer", "")

           prompt = ChatPromptTemplate.from_template(
               """다음은 '{query}'에 대한 여러 소스의 정보입니다:

               [웹 검색 결과]
               {web_result}

               [로컬 문서 검색 결과]
               {rag_result}

               위 정보를 종합하여 사용자의 질의에 대한 최종 답변을 한글로 작성하세요.
               각 정보 소스의 장단점을 고려하고, 상충되는 정보가 있다면
               그 차이점을 명시하세요.
               답변은 논리적이고 일관성 있게 구성하세요."""
           )

           response = self.controller_agent.llm.invoke(
               prompt.format(
                   query=query,
                   web_result=web_result,
                   rag_result=rag_result
               )
           )

           return response.content

       def process_query(self, query):
           analysis = self.controller_agent._analyze_query({"query": query})
           analysis = analysis["analysis"]

           agents_to_run = []
           if "웹 검색" in analysis:
               agents_to_run.append("web")
           if "RAG 검색" in analysis:
               agents_to_run.append("rag")

           if not agents_to_run:
               agents_to_run = ["web"]

           results = asyncio.run(self.execute_parallel(query, agents_to_run))

           if len(agents_to_run) == 1:
               if "web" in agents_to_run:
                   return results["web"]["summary"]
               else:
                   return results["rag"]["answer"]

           integrated_result = self.integrate_results(query, results)
           return integrated_result

2. 비동기 처리:
   - 이벤트 루프 관리
   - 태스크 스케줄링
   - 리소스 활용 최적화

3. 결과 통합:
   - 충돌 해결 전략
   - 일관성 보장
   - 품질 평가

4. 오류 처리:
   - 타임아웃 관리
   - 재시도 로직
   - 격리된 실패 처리

5. 테스트:
   - 병렬 성능 측정
   - 통합 품질 평가
   - 부하 테스트

```

### 5-2: Streamlit 기반 웹 인터페이스 구현

사용자가 시스템과 상호작용할 수 있는 Streamlit 기반 웹 인터페이스를 구현한다.

```
Streamlit 웹 인터페이스를 구현해주세요:

1. 메인 애플리케이션 구현:
   import streamlit as st
   import os
   import tempfile

   def main():
       st.title("LangGraph 기반 멀티에이전트 시스템")

       # 세션 상태 초기화
       if "chat_history" not in st.session_state:
           st.session_state.chat_history = []

       # 사이드바 - 문서 관리
       with st.sidebar:
           st.header("문서 관리")

           # 문서 업로드
           uploaded_file = st.file_uploader(
               "문서 업로드",
               type=["pdf", "txt", "docx"]
           )

           if uploaded_file is not None:
               with tempfile.NamedTemporaryFile(
                   delete=False,
                   suffix=os.path.splitext(uploaded_file.name)[1]
               ) as tmp_file:
                   tmp_file.write(uploaded_file.getvalue())
                   tmp_path = tmp_file.name

               if st.button("문서 저장 및 인덱싱"):
                   with st.spinner("문서를 처리 중입니다..."):
                       doc_id = document_manager.upload_document(
                           tmp_path,
                           {"name": uploaded_file.name}
                       )
                       chunk_count = document_manager.index_document(doc_id)
                       st.success(
                           f"문서가 성공적으로 저장되고 {chunk_count}개의 "
                           f"청크로 인덱싱되었습니다."
                       )
                       os.unlink(tmp_path)

           # 저장된 문서 목록
           st.subheader("저장된 문서")
           documents = document_manager.list_documents()
           for doc_id, metadata in documents.items():
               col1, col2 = st.columns([3, 1])
               with col1:
                   st.write(f"{metadata['original_filename']}")
               with col2:
                   if st.button("삭제", key=f"delete_{doc_id}"):
                       document_manager.delete_document(doc_id)
                       st.experimental_rerun()

       # 메인 영역 - 채팅 인터페이스
       for message in st.session_state.chat_history:
           with st.chat_message(message["role"]):
               st.write(message["content"])

       # 사용자 입력
       user_input = st.chat_input("질문을 입력하세요...")
       if user_input:
           # 사용자 메시지 추가
           st.session_state.chat_history.append({
               "role": "user",
               "content": user_input
           })
           with st.chat_message("user"):
               st.write(user_input)

           # 시스템 응답 생성
           with st.chat_message("assistant"):
               with st.spinner("답변을 생성 중입니다..."):
                   message_placeholder = st.empty()
                   response = parallel_agent_executor.process_query(user_input)
                   message_placeholder.write(response)

           # 시스템 메시지 추가
           st.session_state.chat_history.append({
               "role": "assistant",
               "content": response
           })

   if __name__ == "__main__":
       main()

2. UI 컴포넌트:
   - 채팅 히스토리 관리
   - 문서 업로드 인터페이스
   - 검색 결과 시각화
   - 진행 상태 표시

3. 상태 관리:
   - 세션 상태 활용
   - 히스토리 저장
   - 설정 관리

4. UX 개선:
   - 반응형 디자인
   - 로딩 인디케이터
   - 에러 메시지
   - 도움말 시스템

5. 테스트:
   - UI 사용성
   - 브라우저 호환성
   - 성능 측정

```

### 5-3: 시스템 통합 및 에러 처리

모든 컴포넌트를 통합하고 종합적인 에러 처리 메커니즘을 구현한다.

```
시스템 통합 클래스를 구현해주세요:

1. MultiAgentSystem 클래스 구현:
   import logging
   from typing import Dict, Any, Optional

   class MultiAgentSystem:
       def __init__(self, config_path=None):
           self.logger = self._setup_logging()
           self.config = self._load_config(config_path)

           # 컴포넌트 초기화
           self.embedding_model = self._init_embedding_model()
           self.vector_store = self._init_vector_store()
           self.document_parser = self._init_document_parser()
           self.google_search_api = self._init_google_search_api()

           # 에이전트 초기화
           self.document_manager = self._init_document_manager()
           self.web_search_agent = self._init_web_search_agent()
           self.rag_agent = self._init_rag_agent()
           self.controller_agent = self._init_controller_agent()

           # 병렬 실행기 초기화
           self.parallel_executor = self._init_parallel_executor()

       def _setup_logging(self):
           logger = logging.getLogger("multi_agent_system")
           logger.setLevel(logging.INFO)

           console_handler = logging.StreamHandler()
           console_handler.setLevel(logging.INFO)

           file_handler = logging.FileHandler("multi_agent_system.log")
           file_handler.setLevel(logging.DEBUG)

           formatter = logging.Formatter(
               '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
           )
           console_handler.setFormatter(formatter)
           file_handler.setFormatter(formatter)

           logger.addHandler(console_handler)
           logger.addHandler(file_handler)

           return logger

       def _load_config(self, config_path):
           # 설정 파일 로드 로직
           pass

       def process_query(self, query):
           try:
               self.logger.info(f"질의 처리 시작: {query}")
               result = self.parallel_executor.process_query(query)
               self.logger.info("질의 처리 완료")
               return result
           except Exception as e:
               self.logger.error(
                   f"질의 처리 중 오류 발생: {str(e)}",
                   exc_info=True
               )
               return f"죄송합니다. 질의 처리 중 오류가 발생했습니다: {str(e)}"

       def upload_and_index_document(self, file_path, metadata=None):
           try:
               self.logger.info(f"문서 업로드 시작: {file_path}")
               doc_id = self.document_manager.upload_document(
                   file_path,
                   metadata
               )
               chunk_count = self.document_manager.index_document(doc_id)
               self.logger.info(
                   f"문서 업로드 및 인덱싱 완료: {doc_id}, {chunk_count} 청크"
               )
               return doc_id, chunk_count
           except Exception as e:
               self.logger.error(
                   f"문서 처리 중 오류 발생: {str(e)}",
                   exc_info=True
               )
               raise

       def save_state(self, path):
           # 시스템 상태 저장
           pass

       def load_state(self, path):
           # 시스템 상태 로드
           pass

2. 컴포넌트 초기화:
   - 의존성 관리
   - 초기화 순서
   - 헬스 체크

3. 로깅 시스템:
   - 구조화된 로깅
   - 로그 로테이션
   - 로그 집계

4. 에러 처리:
   - 예외 분류
   - 복구 전략
   - 서킷 브레이커

5. 상태 관리:
   - 상태 직렬화
   - 체크포인팅
   - 상태 복원

```

### 5-4: 성능 최적화 및 캐싱 구현

시스템 성능을 최적화하고 결과 캐싱 메커니즘을 구현한다.

```
캐싱 시스템과 성능 최적화를 구현해주세요:

1. ResultCache 클래스 구현:
   import hashlib
   import json
   import os
   import time
   from typing import Dict, Any, Optional

   class ResultCache:
       def __init__(self, cache_dir="./cache", ttl=3600):
           self.cache_dir = cache_dir
           self.ttl = ttl

           os.makedirs(cache_dir, exist_ok=True)

       def _get_cache_key(self, query, agent_type):
           key = f"{query}_{agent_type}"
           return hashlib.md5(key.encode()).hexdigest()

       def _get_cache_path(self, cache_key):
           return os.path.join(self.cache_dir, f"{cache_key}.json")

       def get(self, query, agent_type):
           cache_key = self._get_cache_key(query, agent_type)
           cache_path = self._get_cache_path(cache_key)

           if not os.path.exists(cache_path):
               return None

           try:
               with open(cache_path, 'r', encoding='utf-8') as f:
                   cache_data = json.load(f)

               if time.time() - cache_data["timestamp"] > self.ttl:
                   os.remove(cache_path)
                   return None

               return cache_data["result"]
           except Exception:
               if os.path.exists(cache_path):
                   os.remove(cache_path)
               return None

       def set(self, query, agent_type, result):
           cache_key = self._get_cache_key(query, agent_type)
           cache_path = self._get_cache_path(cache_key)

           cache_data = {
               "query": query,
               "agent_type": agent_type,
               "result": result,
               "timestamp": time.time()
           }

           try:
               with open(cache_path, 'w', encoding='utf-8') as f:
                   json.dump(cache_data, f, ensure_ascii=False, indent=2)
               return True
           except Exception:
               return False

       def clear(self, max_age=None):
           if max_age is None:
               for filename in os.listdir(self.cache_dir):
                   file_path = os.path.join(self.cache_dir, filename)
                   if os.path.isfile(file_path) and filename.endswith('.json'):
                       os.remove(file_path)
           else:
               now = time.time()
               for filename in os.listdir(self.cache_dir):
                   file_path = os.path.join(self.cache_dir, filename)
                   if os.path.isfile(file_path) and filename.endswith('.json'):
                       try:
                           with open(file_path, 'r', encoding='utf-8') as f:
                               cache_data = json.load(f)
                           if now - cache_data["timestamp"] > max_age:
                               os.remove(file_path)
                       except Exception:
                           os.remove(file_path)

2. 캐싱 전략:
   - 쿼리 결과 캐싱
   - 임베딩 캐싱
   - LLM 응답 캐싱

3. 캐시 무효화:
   - TTL 기반 만료
   - 이벤트 기반 무효화
   - 수동 제거

4. 메모리 최적화:
   - 효율적인 자료구조
   - LRU/LFU 정책
   - 메모리 프로파일링

5. 응답 스트리밍:
   - 점진적 응답
   - 부분 결과 전송
   - 지연 시간 최소화

```

## Phase 6: 테스트 및 문서화

### 6-1: 시스템 테스트 및 문서화

전체 시스템을 테스트하고 사용자 및 개발자 문서를 작성한다.

```
포괄적인 테스트 스위트와 문서를 작성해주세요:

1. 단위 테스트 구현:
   - 각 컴포넌트별 테스트 케이스
   - 90% 이상 코드 커버리지 목표
   - 경계값 분석 및 엣지 케이스
   - 자동화된 테스트 실행

2. 통합 테스트:
   - 컴포넌트 간 상호작용 검증
   - 엔드투엔드 워크플로우 테스트
   - 성능 및 부하 테스트
   - 오류 시나리오 테스트

3. 사용자 문서:
   - 시스템 개요 및 아키텍처 설명
   - 설치 및 설정 가이드
   - 사용법 및 예제
   - 트러블슈팅 가이드
   - FAQ 섹션

4. 개발자 문서:
   - API 레퍼런스
   - 코드 구조 및 설계 패턴
   - 확장 가이드
   - 기여 가이드라인
   - 배포 가이드

5. 예제 및 데모:
   - 샘플 애플리케이션
   - 일반적인 사용 케이스
   - 데모 스크립트
   - 비디오 튜토리얼

각 문서는 한글로 작성하며, 명확하고 이해하기 쉬운 설명을 포함해야 합니다.
실제 코드 예제와 스크린샷을 포함하여 실용적인 문서를 만들어주세요.

```

### 6-2: 테스트

```jsx
“command prompt로 단위 테스트를 실행하세요.  “

“command prompt로 통합 테스트를 실행하세요. “
```

### 6-3. 시스템 실행

```bash
streamlit run src/ai_agent/ui.py

```