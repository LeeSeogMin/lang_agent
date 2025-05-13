"""
Streamlit 기반 웹 인터페이스
이 모듈은 AI 검색 에이전트의 웹 사용자 인터페이스를 구현합니다.
Streamlit을 사용하여 직관적이고 반응형인 UI를 제공합니다.
"""
import streamlit as st
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import threading
import sys
import os
from threading import Lock

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ai_agent.agents.controller_agent import ControllerAgent
from src.ai_agent.core.document_manager import DocumentManager
from src.ai_agent.models.state import SearchResult, DocumentMetadata

@st.cache_resource
def get_controller():
    """
    컨트롤러 에이전트 인스턴스를 반환합니다.
    Streamlit의 캐시 기능을 사용하여 인스턴스를 재사용합니다.
    """
    return ControllerAgent()

@st.cache_resource
def get_document_manager():
    """
    문서 관리자 인스턴스를 반환합니다.
    Streamlit의 캐시 기능을 사용하여 인스턴스를 재사용합니다.
    """
    return DocumentManager()

class WebUI:
    def __init__(self):
        """
        웹 UI 클래스 초기화
        - 컨트롤러와 문서 관리자 인스턴스 생성
        - 세션 상태 초기화
        - 문서 처리용 락 생성
        """
        self.controller = get_controller()
        self.document_manager = get_document_manager()
        self.processing_lock = Lock()  # 문서 처리를 위한 락 추가
        
        # 세션 상태 초기화
        if "conversation_history" not in st.session_state:
            st.session_state["conversation_history"] = []
        if "documents" not in st.session_state:
            st.session_state["documents"] = {}
        if "uploaded_files" not in st.session_state:
            st.session_state["uploaded_files"] = set()
            # 기존 문서의 파일명도 추가
            for metadata in self.document_manager.list_documents().values():
                st.session_state["uploaded_files"].add(metadata.original_filename)
        if "processing_file" not in st.session_state:
            st.session_state["processing_file"] = None
        if "processing_status" not in st.session_state:
            st.session_state["processing_status"] = None
        if "upload_error" not in st.session_state:
            st.session_state["upload_error"] = None
    
    def render_header(self):
        """
        헤더 섹션을 렌더링합니다.
        애플리케이션의 제목과 주요 기능을 설명합니다.
        """
        st.title("🤖 AI 검색 에이전트")
        st.markdown("""
        웹 검색과 문서 검색을 결합한 하이브리드 검색 시스템입니다.
        - 📚 **문서 관리**: PDF, DOCX, TXT 파일 지원
        - 🔍 **통합 검색**: 웹 검색 + 문서 검색
        - 🎯 **정확도**: 컨텍스트 기반 순위 지정
        """)
    
    def process_document(self, file_path: str, file_name: str):
        """
        문서를 처리하고 인덱싱합니다.
        
        Args:
            file_path (str): 처리할 파일의 경로
            file_name (str): 원본 파일명
        
        처리 과정:
        1. 중복 문서 확인 및 삭제
        2. 문서 업로드
        3. 문서 인덱싱
        4. 상태 업데이트
        5. 임시 파일 정리
        """
        with self.processing_lock:  # 락을 사용하여 동시 처리 방지
            try:
                # 진행 상태 로깅
                print(f"문서 처리 시작: {file_name}")
                st.session_state["processing_status"] = "uploading"
                
                # 중복 문서 확인 및 삭제
                existing_docs = [
                    doc_id for doc_id, metadata in self.document_manager.list_documents().items()
                    if metadata.original_filename == file_name
                ]
                
                # 중복 문서가 있으면 삭제
                if existing_docs:
                    print(f"기존 문서 삭제: {file_name}")
                    for doc_id in existing_docs:
                        self.document_manager.delete_document(doc_id)
                
                # 문서 업로드
                doc_id = self.document_manager.upload_document(
                    file_path,
                    custom_metadata={
                        "original_name": file_name,
                        "processed_at": str(time.time()),
                        "version": "2.0"
                    }
                )
                
                print(f"문서 업로드 완료: {doc_id}")
                
                # 문서 인덱싱
                st.session_state["processing_status"] = "indexing"
                start_time = time.time()
                chunk_count = self.document_manager.index_document(doc_id)
                end_time = time.time()
                
                processing_time = end_time - start_time
                print(f"문서 인덱싱 완료: {chunk_count}개 청크, {processing_time:.2f}초 소요")
                
                # 상태 업데이트
                st.session_state["documents"] = self.document_manager.list_documents()
                st.session_state["uploaded_files"].add(file_name)
                st.session_state["processing_file"] = None
                st.session_state["processing_status"] = "completed"
                st.session_state["upload_error"] = None
                
                # 임시 파일 정리
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"임시 파일 삭제: {file_path}")
                except Exception as cleanup_error:
                    print(f"임시 파일 정리 오류: {str(cleanup_error)}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"문서 처리 오류: {file_name} - {error_msg}")
                st.session_state["upload_error"] = error_msg
                st.session_state["processing_file"] = None
                st.session_state["processing_status"] = "error"
                
                # 오류 발생 시 임시 파일 정리 시도
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
    
    def render_document_upload(self):
        """문서 업로드 섹션 렌더링"""
        st.sidebar.header("📄 문서 관리")
        
        # 처리 중인 파일이 있으면 상태 표시
        if st.session_state.get("processing_file"):
            status = st.session_state.get("processing_status", "uploading")
            
            # 상태 메시지 설정
            status_messages = {
                "uploading": "업로드 중...",
                "indexing": "인덱싱 중...",
                "completed": "처리 완료!",
                "error": "오류 발생"
            }
            
            # 진행 상태 표시
            progress_placeholder = st.sidebar.empty()
            status_text = st.sidebar.empty()
            
            # 상태별 진행률 표시
            progress_values = {
                "uploading": 25,
                "indexing": 75,
                "completed": 100,
                "error": 0
            }
            
            # 진행 상태 업데이트
            current_progress = progress_values.get(status, 0)
            progress_placeholder.progress(current_progress)
            status_text.text(f"'{st.session_state['processing_file']}' {status_messages.get(status, '')}")
            
            if status == "completed":
                st.sidebar.success(f"'{st.session_state['processing_file']}' 처리 완료!")
                st.session_state["processing_file"] = None
                st.session_state["processing_status"] = None
                st.rerun()
                return
            elif status == "error":
                return
        
        # 오류가 있으면 표시
        if st.session_state.get("upload_error"):
            st.sidebar.error(f"문서 처리 중 오류 발생: {st.session_state['upload_error']}")
            
            # 오류 유형에 따른 해결책 제안
            if "인코딩" in st.session_state["upload_error"]:
                st.sidebar.warning("텍스트 파일 인코딩 문제입니다. UTF-8 인코딩으로 저장된 파일을 업로드해주세요.")
            elif "PDF" in st.session_state["upload_error"]:
                st.sidebar.warning("PDF 파일 파싱 중 문제가 발생했습니다. 다른 PDF 파일을 시도해보세요.")
            
            if st.sidebar.button("다시 시도", key="retry_upload"):
                st.session_state["upload_error"] = None
                st.rerun()
        
        # 지원되는 파일 형식 및 크기 제한 안내
        st.sidebar.caption("지원 형식: PDF, DOCX, TXT (최대 10MB)")
        
        # 업로드 폼으로 감싸기
        with st.sidebar.form("upload_form"):
            # 파일 업로더
            uploaded_file = st.file_uploader(
                "문서 업로드",
                type=["pdf", "docx", "txt"],
                help="PDF, DOCX, TXT 파일을 업로드하세요."
            )
            
            # 업로드 버튼
            upload_button = st.form_submit_button("업로드 시작")
            
            # 업로드 버튼이 눌리면
            if upload_button and uploaded_file:
                # 파일 크기 검증 (10MB 제한)
                if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
                    st.error("파일 크기가 너무 큽니다. 10MB 이하의 파일을 업로드하세요.")
                    return
                
                # 이미 업로드된 파일인지 확인
                if uploaded_file.name in st.session_state["uploaded_files"]:
                    # 확인 메시지
                    if not st.session_state.get("confirm_overwrite"):
                        st.warning(f"'{uploaded_file.name}'은(는) 이미 업로드된 파일입니다. 덮어쓰시겠습니까?")
                        st.session_state["confirm_overwrite"] = True
                        return
                    else:
                        # 확인 후 세션 상태 초기화
                        st.session_state["confirm_overwrite"] = False
                
                # 상태 업데이트
                st.session_state["processing_file"] = uploaded_file.name
                st.session_state["processing_status"] = "uploading"
                st.session_state["upload_error"] = None
                
                try:
                    # 임시 파일 저장
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / uploaded_file.name
                    
                    # 파일이 이미 존재하면 삭제
                    if temp_path.exists():
                        os.remove(temp_path)
                    
                    # 새 파일 저장
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # 동기적으로 문서 처리
                    with self.processing_lock:  # 락 사용하여 동시 처리 방지
                        # 중복 문서 확인
                        existing_docs = [
                            doc_id for doc_id, metadata in self.document_manager.list_documents().items()
                            if metadata.original_filename == uploaded_file.name
                        ]
                        
                        # 중복 문서가 있으면 삭제
                        if existing_docs:
                            for doc_id in existing_docs:
                                self.document_manager.delete_document(doc_id)
                        
                        # 문서 업로드
                        doc_id = self.document_manager.upload_document(
                            str(temp_path),
                            custom_metadata={
                                "original_name": uploaded_file.name,
                                "processed_at": str(time.time()),
                                "version": "2.0"
                            }
                        )
                        
                        # 인덱싱 상태 업데이트
                        st.session_state["processing_status"] = "indexing"
                        
                        # 문서 인덱싱
                        chunk_count = self.document_manager.index_document(doc_id)
                        
                        # 상태 업데이트
                        st.session_state["documents"] = self.document_manager.list_documents()
                        st.session_state["uploaded_files"].add(uploaded_file.name)
                        st.session_state["processing_status"] = "completed"
                        
                    # 임시 파일 정리
                    if temp_path.exists():
                        os.remove(temp_path)
                    
                    # 업로드 성공 메시지
                    st.success(f"'{uploaded_file.name}' 문서가 성공적으로 처리되었습니다. ({chunk_count}개 청크)")
                    st.rerun()
                    
                except Exception as e:
                    error_msg = str(e)
                    st.session_state["upload_error"] = error_msg
                    st.session_state["processing_status"] = "error"
                    st.error(f"문서 처리 중 오류 발생: {error_msg}")
                    
                    # 임시 파일 정리 시도
                    try:
                        if temp_path.exists():
                            os.remove(temp_path)
                    except:
                        pass
                    
                finally:
                    # 처리 완료 후 상태 정리
                    st.session_state["processing_file"] = None
    
    def render_document_list(self):
        """문서 목록 섹션 렌더링"""
        st.sidebar.subheader("📚 업로드된 문서")
        
        # 문서 메타데이터 새로 로드
        documents = self.document_manager.list_documents()
        st.session_state["documents"] = documents
        
        # 문서 정보 요약 표시
        doc_count = len(documents)
        if doc_count > 0:
            total_chunks = sum(metadata.chunk_count or 0 for metadata in documents.values())
            st.sidebar.caption(f"총 {doc_count}개 문서, {total_chunks}개 청크")
        
        if not documents:
            st.sidebar.info("업로드된 문서가 없습니다.")
            return
        
        # 문서 목록 정렬 (최근 업로드 순)
        sorted_docs = sorted(
            documents.items(),
            key=lambda x: x[1].uploaded_at,
            reverse=True
        )
        
        # 문서 목록 출력
        for doc_id, metadata in sorted_docs:
            # 문서 컨테이너
            with st.sidebar.container():
                # 문서 제목 및 정보
                doc_title = metadata.original_filename
                doc_type = metadata.file_type.upper()
                doc_chunks = metadata.chunk_count or 0
                
                # 업로드 시간 계산
                upload_time = ""
                try:
                    from datetime import datetime
                    upload_dt = datetime.fromisoformat(metadata.uploaded_at)
                    now = datetime.now()
                    delta = now - upload_dt
                    
                    if delta.days > 0:
                        upload_time = f"{delta.days}일 전"
                    elif delta.seconds // 3600 > 0:
                        upload_time = f"{delta.seconds // 3600}시간 전"
                    else:
                        upload_time = f"{delta.seconds // 60}분 전"
                except:
                    pass
                
                # 문서 헤더
                col1, col2 = st.sidebar.columns([3, 1])
                
                # 문서 정보
                with col1:
                    st.markdown(f"**{doc_title}**")
                    st.caption(f"유형: {doc_type} | 청크: {doc_chunks} | {upload_time}")
                
                # 문서 작업 버튼
                with col2:
                    # 삭제 버튼
                    if st.button("삭제", key=f"del_{doc_id}"):
                        try:
                            self.document_manager.delete_document(doc_id)
                            if "uploaded_files" in st.session_state:
                                st.session_state["uploaded_files"].remove(metadata.original_filename)
                            st.session_state["documents"] = self.document_manager.list_documents()
                            st.rerun()
                        except Exception as e:
                            st.error(f"문서 삭제 중 오류 발생: {str(e)}")
                
                # 구분선 추가
                st.sidebar.markdown("---")
    
    def render_search_results(self, results: List[SearchResult], summary: Optional[Dict[str, Any]] = None):
        """검색 결과 렌더링"""
        if not results:
            st.warning("검색 결과가 없습니다.")
            return
            
        # 검색 결과 요약
        st.markdown("### 📋 검색 결과 종합")
        
        if summary and "summary" in summary:
            # AI 생성 요약 표시
            st.markdown(summary["summary"])
        else:
            st.info("요약을 생성할 수 없습니다.")
        
        st.markdown("---")
        
        # 결과 소스별 개수 계산 및 표시
        source_counts = {}
        for result in results:
            source_counts[result.source] = source_counts.get(result.source, 0) + 1
        
        # 소스별 통계
        source_icons = {
            "web": "🌐 웹 검색",
            "scholar": "📚 학술 검색",
            "rag": "📄 문서 검색"
        }
        
        with st.expander("검색 소스 통계", expanded=False):
            stats_cols = st.columns(len(source_counts))
            for i, (source, count) in enumerate(source_counts.items()):
                with stats_cols[i]:
                    st.metric(
                        source_icons.get(source, f"ℹ️ {source}"),
                        f"{count}개"
                    )
        
        # 개별 검색 결과 표시
        st.markdown("### 🔍 상세 검색 결과")
        for i, result in enumerate(results):
            # 결과 소스에 따라 아이콘 설정
            if result.source == "web":
                icon = "🌐"
                source_name = "웹 검색"
            elif result.source == "scholar":
                icon = "📚"
                source_name = "학술 검색"
            elif result.source == "rag":
                icon = "📄"
                source_name = "문서 검색"
            else:
                icon = "ℹ️"
                source_name = result.source
            
            # 관련도 점수 계산 (0-100%)
            relevance = min(100, max(0, int(result.score * 100))) if result.score > 0 else int(abs(result.score) * 100)
            
            # 결과 컨테이너
            with st.expander(f"{icon} {result.title}", expanded=i < 3):
                # 메타데이터 표시
                st.markdown(f"**출처**: {source_name} | **관련도**: {relevance}%")
                
                # 본문 내용 표시 (마크다운 지원)
                st.markdown(result.snippet)
                
                # 페이지 번호 정보가 있으면 표시 (PDF 문서 경우)
                if "[페이지" in result.snippet:
                    page_match = result.snippet.split("[페이지")[1].split("]")[0].strip()
                    if page_match:
                        st.caption(f"페이지: {page_match}")
                
                # 관련도 표시 (시각적 지표)
                relevance_color = "green" if relevance > 70 else "orange" if relevance > 40 else "red"
                st.progress(relevance / 100)
    
    def render_search_interface(self):
        """검색 인터페이스 렌더링"""
        st.subheader("🔍 검색")
        
        # 검색 설명
        st.caption("문서 내용을 검색하려면 키워드를 입력하세요. 예: '트랜스포머', '시계열 분석'")
        
        # 예시 검색어 제공
        example_cols = st.columns(3)
        with example_cols[0]:
            if st.button("트랜스포머"):
                st.session_state.search_query = "트랜스포머"
                st.rerun()
        with example_cols[1]:
            if st.button("시계열"):
                st.session_state.search_query = "시계열"
                st.rerun()
        with example_cols[2]:
            if st.button("데이터 분석"):
                st.session_state.search_query = "데이터 분석"
                st.rerun()
        
        # 검색 양식
        with st.form(key="search_form"):
            # 검색 입력
            query = st.text_input(
                "검색어",
                value=st.session_state.get("search_query", ""),
                placeholder="예: 트랜스포머 모델의 작동 원리",
                help="정확한 검색을 위해 구체적인 키워드를 사용하세요."
            )
            
            # 검색 옵션
            num_results = st.slider(
                "표시할 결과 수",
                min_value=3,
                max_value=15,
                value=5,
                help="검색 결과 수를 선택하세요."
            )
            
            # 검색 버튼
            search_submitted = st.form_submit_button("검색", type="primary", use_container_width=True)
        
        # 검색 실행
        if search_submitted and query:
            # 검색 시작 시간 기록
            start_time = time.time()
            
            # 검색 실행 (스피너 표시)
            with st.spinner(f"'{query}' 검색 중..."):
                # 검색 실행
                state = self.controller.process_query(
                    query,
                    num_results=num_results
                )
            
            # 검색 완료 시간 계산
            end_time = time.time()
            search_time = end_time - start_time
            
            # 검색 결과 요약
            result_count = len(state.search_results)
            if result_count > 0:
                st.success(f"검색 완료: {result_count}개 결과 ({search_time:.2f}초)")
                
                # 결과 표시
                st.subheader("📊 검색 결과")
                self.render_search_results(state.search_results, state.summary)
            else:
                st.warning(f"'{query}'에 대한 검색 결과가 없습니다. 다른 키워드로 시도해보세요.")
            
            # 대화 기록 업데이트
            st.session_state.conversation_history.append({
                "query": query,
                "results": state.search_results,
                "summary": state.summary,
                "timestamp": time.time()
            })
    
    def render_conversation_history(self):
        """대화 기록 렌더링"""
        if st.session_state.conversation_history:
            st.subheader("📝 최근 검색 기록")
            
            # 히스토리 항목 개수
            history_count = len(st.session_state.conversation_history)
            st.caption(f"전체 {history_count}개 검색 기록 중 최근 5개 표시")
            
            # 검색 기록 정리 버튼
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("기록 초기화", key="clear_history"):
                    st.session_state.conversation_history = []
                    st.rerun()
            
            # 검색 기록이 있으면 표시
            if st.session_state.conversation_history:
                # 최신 기록 5개만 표시 (역순)
                for i, item in enumerate(reversed(st.session_state.conversation_history[-5:])):
                    # 검색 쿼리
                    query = item['query']
                    
                    # 결과 개수
                    result_count = len(item['results'])
                    
                    # 타임스탬프 (있으면 표시)
                    timestamp_str = ""
                    if 'timestamp' in item:
                        from datetime import datetime
                        ts = datetime.fromtimestamp(item['timestamp'])
                        timestamp_str = ts.strftime("%H:%M:%S")
                    
                    # 히스토리 헤더 설정
                    history_header = f"🔍 {query} ({result_count}개 결과)"
                    if timestamp_str:
                        history_header += f" - {timestamp_str}"
                    
                    # 쿼리 재시도 버튼
                    if st.button(f"'{query}' 다시 검색", key=f"retry_{i}", on_click=self.set_query, args=(query,)):
                        st.rerun()
                    
                    # 검색 결과 표시
                    st.markdown(f"**{history_header}**")
                    if result_count > 0:
                        # 요약이 있으면 표시
                        if 'summary' in item:
                            st.markdown(item['summary'].get('summary', ''))
                        
                        # 상세 결과는 expander로 표시
                        with st.expander("상세 결과 보기", expanded=False):
                            for result in item['results']:
                                st.markdown(f"**{result.title}**")
                                st.markdown(result.snippet)
                                st.markdown("---")
            else:
                st.info("검색 기록이 없습니다.")
    
    def set_query(self, query):
        """검색어 설정"""
        # 상태에 검색어 저장 (다음 양식 렌더링에서 사용)
        if "query" not in st.session_state:
            st.session_state.query = query
        else:
            st.session_state.query = query
    
    def render(self):
        """전체 UI 렌더링"""
        self.render_header()
        self.render_document_upload()
        self.render_document_list()
        self.render_search_interface()
        self.render_conversation_history()

def main():
    """메인 함수"""
    # 페이지 설정
    st.set_page_config(
        page_title="AI 검색 에이전트",
        page_icon="🤖",
        layout="wide"
    )
    
    # UI 인스턴스 생성 및 렌더링
    ui = WebUI()
    ui.render()

if __name__ == "__main__":
    main() 