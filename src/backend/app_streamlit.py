"""Multi-Agent System Streamlit Application"""

import os
import asyncio
import streamlit as st
import uuid
import time
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

from backend.agents.orchestrator.graph import create_graph, AgentState
from backend.utils.document_ingestion import ingest_file
from backend.agents.data_analysis.graph import create_data_analysis_graph
from backend.agents.data_analysis.schemas import DataAnalysisState
import backend.config as config

# 기본 설정
st.set_page_config(
    page_title="LangGraph Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 관리
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState(messages=[])

# 사이드바
with st.sidebar:
    st.title("LangGraph Multi-Agent")
    
    # 세션 관리
    st.header("세션 관리")
    if st.button("Start New Chat"):
        session_id = str(uuid.uuid4())
        st.session_state.sessions[session_id] = {
            "created_at": datetime.now(),
            "messages": []
        }
        st.session_state.current_session_id = session_id
        st.session_state.chat_history = []
        st.session_state.agent_state = AgentState(messages=[])
        st.rerun()
    
    # 세션 선택
    if st.session_state.sessions:
        st.subheader("세션 선택")
        session_options = {f"{sid[:6]}... ({info['created_at'].strftime('%H:%M:%S')})": sid 
                         for sid, info in st.session_state.sessions.items()}
        selected_session = st.selectbox(
            "Choose a session", 
            options=list(session_options.keys()),
            index=list(session_options.keys()).index(f"{st.session_state.current_session_id[:6]}... ({st.session_state.sessions[st.session_state.current_session_id]['created_at'].strftime('%H:%M:%S')})") if st.session_state.current_session_id else 0
        )
        
        if st.button("Load Selected Session"):
            session_id = session_options[selected_session]
            st.session_state.current_session_id = session_id
            st.session_state.chat_history = st.session_state.sessions[session_id].get("messages", [])
            st.session_state.agent_state = AgentState(messages=st.session_state.chat_history)
            st.rerun()
    
    # 문서 업로드 섹션
    st.header("문서 업로드 (RAG)")
    uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf", "md"])
    
    if uploaded_file is not None:
        # 임시 파일로 저장
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("문서 파싱/분석"):
            with st.spinner("문서 처리 중..."):
                # 비동기 함수를 위한 래퍼
                result = asyncio.run(ingest_file(file_path))
                if result.get("status") == "success":
                    st.success(f"✅ {uploaded_file.name} 처리 완료!")
                    st.write(f"총 {result.get('num_chunks', 0)}개의 청크로 나누어 저장되었습니다.")
                else:
                    st.error(f"❌ 실패: {result.get('error', '알 수 없는 오류')}")
    
    # 데이터 분석 섹션 - expander 대신 일반 헤더 사용
    st.header("데이터 분석")
    
    # 파일 업로드
    st.subheader("파일 업로드")
    data_file = st.file_uploader("CSV, TXT 또는 JSON 파일을 업로드하세요", type=["csv", "txt", "json"], key="data_analysis_uploader")
    
    if data_file is not None:
        # 업로드된 파일 저장
        save_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, data_file.name)
        
        with open(file_path, "wb") as f:
            f.write(data_file.getbuffer())
        
        # 세션에 파일 경로 저장
        st.session_state.uploaded_data_file = file_path
        
        # 파일 미리보기 표시
        st.success(f"파일이 업로드되었습니다: {data_file.name}")
        
        # 파일 미리보기 (처음 10줄)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                lines = f.readlines()[:5]  # 5줄로 제한
                
                # CSV 파일 특별 처리 (헤더와 몇 개의 행만 표시)
                if file_path.endswith('.csv'):
                    if len(lines) > 1:
                        preview = lines[0] + "".join(lines[1:3])
                        if len(lines) > 3:
                            preview += "...(생략)..."
                    else:
                        preview = "".join(lines)
                else:
                    # 일반 텍스트 파일은 5줄만 표시
                    preview = "".join(lines)
                    if len(lines) >= 5:
                        preview += "...(생략)..."
            except Exception as e:
                preview = f"파일 미리보기 오류: {str(e)}"

        st.subheader("파일 미리보기")
        st.text(preview)
        
        # 고급 분석 옵션
        st.subheader("고급 분석 옵션")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RAG 필터링 옵션
            rag_enabled = st.checkbox("RAG 기반 데이터 필터링 활성화", 
                        key="rag_filtering_enabled",
                        help="자연어로 데이터를 필터링하고 특정 정보를 추출할 수 있습니다")
            
            if rag_enabled:
                filter_query = st.text_input("필터링 쿼리 (자연어)", 
                             key="filter_query",
                             value="중요한 정보와 주요 패턴을 추출해주세요",
                             help="자연어로 필터링하려는 내용을 설명하세요")
        
        with col2:
            # 코드 생성 옵션
            code_gen_enabled = st.checkbox("데이터 분석 코드 생성", 
                        key="code_generation_enabled",
                        help="데이터를 분석하는 Python 코드를 자동으로 생성합니다")
            
            if code_gen_enabled:
                code_task = st.text_input("분석 작업 설명", 
                             key="code_generation_task",
                             value="데이터 분석 및 시각화",
                             help="생성하려는 코드의 목적을 설명하세요")
                
                code_type = st.selectbox("코드 유형", 
                            ["pandas", "numpy", "matplotlib"],
                            key="code_type",
                            help="생성할 코드의 주요 라이브러리 유형을 선택하세요")
        
        # 즉시 분석 버튼
        if st.button("지금 분석하기"):
            try:
                # 파일 읽기
                with open(file_path, "r", encoding="utf-8") as f:
                    data_content = f.read()
                
                # 오케스트레이터를 통한 데이터 분석을 위해 특수 프리픽스 추가
                message_content = f"ANALYZE DATA:\n\n{data_content}"
                
                # 메시지 추가 및 표시
                prompt = f"파일 '{data_file.name}'을 분석해주세요."
                st.session_state.chat_history.append(HumanMessage(content=message_content))
                with st.chat_message("user"):
                    st.write(prompt)  # 사용자에게 보여줄 때는 간략한 프롬프트 표시
                
                # 메시지를 에이전트 상태에 추가
                st.session_state.agent_state.messages = st.session_state.chat_history
                st.session_state.agent_state.document_content = data_content
                
                # 파일 경로 유지
                st.session_state.agent_state.uploaded_data_file = file_path
                
                # 고급 분석 옵션 설정
                if rag_enabled:
                    st.session_state.agent_state.apply_rag_filtering = True
                    st.session_state.agent_state.filter_query = filter_query
                else:
                    st.session_state.agent_state.apply_rag_filtering = False
                
                if code_gen_enabled:
                    st.session_state.agent_state.apply_code_generation = True
                    st.session_state.agent_state.code_generation_task = code_task
                    st.session_state.agent_state.code_type = code_type
                else:
                    st.session_state.agent_state.apply_code_generation = False
                
                # 분석 플래그 설정
                st.session_state.analyzing_data = True
                st.session_state.data_to_analyze = data_content
                st.session_state.analysis_source = "file"
                
                st.rerun()  # 강제 리런
                
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
    
    # 직접 데이터 입력
    st.subheader("직접 데이터 입력")
    data_text = st.text_area("분석할 데이터를 입력하세요", height=150)
    
    # 직접 입력에도 고급 분석 옵션 추가
    if data_text:
        st.subheader("고급 분석 옵션")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RAG 필터링 옵션
            rag_text_enabled = st.checkbox("RAG 기반 데이터 필터링 활성화", 
                        key="rag_filtering_enabled_text",
                        help="자연어로 데이터를 필터링하고 특정 정보를 추출할 수 있습니다")
            
            if rag_text_enabled:
                filter_text_query = st.text_input("필터링 쿼리 (자연어)", 
                             key="filter_query_text",
                             value="중요한 정보와 주요 패턴을 추출해주세요",
                             help="자연어로 필터링하려는 내용을 설명하세요")
        
        with col2:
            # 코드 생성 옵션
            code_gen_text_enabled = st.checkbox("데이터 분석 코드 생성", 
                        key="code_generation_enabled_text",
                        help="데이터를 분석하는 Python 코드를 자동으로 생성합니다")
            
            if code_gen_text_enabled:
                code_text_task = st.text_input("분석 작업 설명", 
                             key="code_generation_task_text",
                             value="데이터 분석 및 시각화",
                             help="생성하려는 코드의 목적을 설명하세요")
                
                code_text_type = st.selectbox("코드 유형", 
                            ["pandas", "numpy", "matplotlib"],
                            key="code_type_text",
                            help="생성할 코드의 주요 라이브러리 유형을 선택하세요")
    
    if data_text and st.button("입력 데이터 분석하기"):
        # 세션에 분석할 데이터 저장
        st.session_state.data_to_analyze = data_text
        st.session_state.analyzing_data = True
        st.session_state.analysis_source = "input"
        
        # 메시지 형식으로 추가 (내부 처리용)
        message_content = f"ANALYZE DATA:\n\n{data_text}"
        
        # 분석 요청 메시지를 채팅 기록에 추가 (실제 표시할 때는 변환됨)
        st.session_state.chat_history.append(HumanMessage(content=message_content))
        
        # 메시지를 에이전트 상태에 추가
        st.session_state.agent_state.messages = st.session_state.chat_history
        st.session_state.agent_state.document_content = data_text
        
        # 고급 분석 옵션 설정
        if rag_text_enabled:
            st.session_state.agent_state.apply_rag_filtering = True
            st.session_state.agent_state.filter_query = filter_text_query
        else:
            st.session_state.agent_state.apply_rag_filtering = False
        
        if code_gen_text_enabled:
            st.session_state.agent_state.apply_code_generation = True
            st.session_state.agent_state.code_generation_task = code_text_task
            st.session_state.agent_state.code_type = code_text_type
        else:
            st.session_state.agent_state.apply_code_generation = False
        
        # 업로드된 파일 정보는 유지 (분석에는 직접 데이터 사용)
        if hasattr(st.session_state, "uploaded_data_file") and st.session_state.uploaded_data_file:
            st.session_state.agent_state.uploaded_data_file = st.session_state.uploaded_data_file
            print(f"[DEBUG] Direct input analysis with file info preserved: {st.session_state.uploaded_data_file}")
        
        st.rerun()  # 강제 리런

# 메인 영역
st.title("LangGraph 기반 멀티에이전트 시스템")

# 채팅 인터페이스
st.header("AI 어시스턴트와 채팅")

# 채팅 메시지 표시
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        # ANALYZE DATA: 프리픽스가 있는 메시지는 사용자 친화적인 내용으로 변경
        content = message.content
        if content.startswith("ANALYZE DATA:"):
            # 데이터 소스 확인
            if st.session_state.get("analysis_source") == "file" and hasattr(st.session_state, "uploaded_data_file"):
                file_name = os.path.basename(st.session_state.uploaded_data_file)
                display_content = f"파일 '{file_name}'을 분석해주세요."
            else:
                display_content = "입력한 데이터를 분석해주세요."
        else:
            display_content = content
            
        with st.chat_message("user"):
            st.write(display_content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# 세션에 저장된 데이터 분석 요청 처리
if st.session_state.get("analyzing_data", False) and st.session_state.get("data_to_analyze"):
    data_content = st.session_state.data_to_analyze
    source = st.session_state.get("analysis_source", "unknown")
    
    # 오케스트레이터를 통한 데이터 분석을 위해 특수 프리픽스 추가
    message_content = f"ANALYZE DATA:\n\n{data_content}"
    
    # 메시지 추가 및 표시 
    if source == "file" and hasattr(st.session_state, "uploaded_data_file"):
        file_name = os.path.basename(st.session_state.uploaded_data_file)
        prompt = f"파일 '{file_name}'을 분석해주세요."
    else:
        prompt = "입력한 데이터를 분석해주세요."
        
    # 메시지를 내부적으로 추가하고 사용자 화면에 표시
    st.session_state.chat_history.append(HumanMessage(content=message_content))
    
    # 사용자 메시지로 표시 (이미 위의 반복문에서 표시되므로 여기서는 생략)
    
    # 메시지를 에이전트 상태에 추가
    st.session_state.agent_state.messages = st.session_state.chat_history
    
    # AI 응답 준비
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("데이터를 분석 중입니다... 잠시만 기다려주세요.")
        
        try:
            # 비동기 함수 호출을 위한 도우미 함수
            async def get_ai_response():
                graph = create_graph()
                result = await graph.ainvoke(st.session_state.agent_state)
                
                # 결과에서 AI 메시지 찾기
                if hasattr(result, "messages") and result.messages:
                    for msg in reversed(result.messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            return msg.content
                
                # 직접 answer 함수 호출
                from backend.agents.orchestrator.graph import answer
                answer_result = await answer(st.session_state.agent_state)
                
                if hasattr(answer_result, "messages") and answer_result.messages:
                    for msg in answer_result.messages:
                        if hasattr(msg, "content") and msg.content:
                            return msg.content
                
                return "응답을 생성할 수 없습니다."
            
            # 비동기 함수 실행
            response = asyncio.run(get_ai_response())
            
            # 응답 표시
            message_placeholder.write(response)
            
            # 응답을 채팅 기록에 추가
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # 세션 업데이트
            if st.session_state.current_session_id:
                st.session_state.sessions[st.session_state.current_session_id]["messages"] = st.session_state.chat_history
            
        except Exception as e:
            message_placeholder.error(f"오류 발생: {str(e)}")
        
        # 분석 상태 초기화
        st.session_state.analyzing_data = False
        st.session_state.data_to_analyze = None

# 일반 채팅 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 입력 메시지 추가
    message_content = prompt
    
    # 데이터 분석 요청 감지
    data_analysis_keywords = ["데이터 분석", "분석해줘", "데이터를 분석", "방금 업로드한 데이터"]
    is_data_analysis_request = any(keyword in prompt.lower() for keyword in data_analysis_keywords)
    
    # 데이터 분석 요청이고 업로드된 데이터 파일이 있는 경우
    if is_data_analysis_request and hasattr(st.session_state, "uploaded_data_file"):
        try:
            # 파일 읽기
            with open(st.session_state.uploaded_data_file, "r", encoding="utf-8") as f:
                data_content = f.read()
            
            # 오케스트레이터를 통한 데이터 분석을 위해 특수 프리픽스 추가
            message_content = f"ANALYZE DATA:\n\n{data_content}"
            st.session_state.analyzing_data = True
            st.session_state.analysis_source = "file"
            
        except Exception as e:
            st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
    
    # 메시지 추가 및 표시
    st.session_state.chat_history.append(HumanMessage(content=message_content))
    with st.chat_message("user"):
        st.write(prompt)  # 사용자에게 보여줄 때는 원래 프롬프트 표시
    
    # 메시지를 에이전트 상태에 추가
    st.session_state.agent_state.messages = st.session_state.chat_history
    
    # AI 응답 준비
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if st.session_state.get("analyzing_data", False):
            message_placeholder.text("데이터를 분석 중입니다... 잠시만 기다려주세요.")
        else:
            message_placeholder.text("생각 중...")
        
        try:
            # 비동기 함수 호출을 위한 도우미 함수
            async def get_ai_response():
                graph = create_graph()
                result = await graph.ainvoke(st.session_state.agent_state)
                
                # 결과에서 AI 메시지 찾기
                if hasattr(result, "messages") and result.messages:
                    for msg in reversed(result.messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            return msg.content
                
                # 직접 answer 함수 호출
                from backend.agents.orchestrator.graph import answer
                answer_result = await answer(st.session_state.agent_state)
                
                if hasattr(answer_result, "messages") and answer_result.messages:
                    for msg in answer_result.messages:
                        if hasattr(msg, "content") and msg.content:
                            return msg.content
                
                return "응답을 생성할 수 없습니다."
            
            # 비동기 함수 실행
            response = asyncio.run(get_ai_response())
            
            # 데이터 분석 완료 표시
            if st.session_state.get("analyzing_data", False):
                st.session_state.analyzing_data = False
            
            # 응답 표시
            message_placeholder.write(response)
            
            # 응답을 채팅 기록에 추가
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # 세션 업데이트
            if st.session_state.current_session_id:
                st.session_state.sessions[st.session_state.current_session_id]["messages"] = st.session_state.chat_history
            
        except Exception as e:
            message_placeholder.error(f"오류 발생: {str(e)}")
            # 오류 발생시 분석 플래그 초기화
            if st.session_state.get("analyzing_data", False):
                st.session_state.analyzing_data = False

# 앱 시작 시 세션 생성 (없는 경우)
if not st.session_state.current_session_id and not st.session_state.sessions:
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "created_at": datetime.now(),
        "messages": []
    }
    st.session_state.current_session_id = session_id 