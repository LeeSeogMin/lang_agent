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
    
    # 데이터 파일 업로드 (분석은 채팅에서 진행)
    st.header("데이터 파일 업로드")
    data_file = st.file_uploader("데이터 파일 업로드", type=["csv", "txt", "md"])
    
    if data_file is not None:
        # 임시 파일로 저장
        file_path = f"temp_data_{data_file.name}"
        with open(file_path, "wb") as f:
            f.write(data_file.getbuffer())
        st.success(f"✅ 데이터 파일 '{data_file.name}'가 업로드되었습니다.")
        st.info("채팅창에서 '이 데이터를 분석해줘' 또는 '방금 업로드한 데이터 분석'과 같이 입력하여 분석을 요청하세요.")
        
        # 세션 상태에 데이터 파일 경로 저장
        st.session_state.uploaded_data_file = file_path
        st.session_state.uploaded_data_filename = data_file.name

# 메인 영역
st.title("LangGraph 기반 멀티에이전트 시스템")

# 데이터 분석 결과 표시 (있는 경우)
if st.session_state.get("show_analysis", False):
    st.header("Analysis Results")
    with st.spinner("분석 중..."):
        try:
            data = st.session_state.analysis_data
            state = DataAnalysisState(data=data, messages=[])
            graph = create_data_analysis_graph()
            results = asyncio.run(graph.ainvoke(state))
            
            st.write("### 분석 결과")
            if hasattr(results, "analysis") and results.analysis:
                st.write(results.analysis)
            else:
                st.error("분석 결과를 가져오지 못했습니다.")
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
    
    st.session_state.show_analysis = False

# 채팅 인터페이스
st.header("AI 어시스턴트")

# 채팅 메시지 표시
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# 사용자 입력 처리
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


# CLI 호환성을 위한 메인 함수
def main():
    """CLI에서 실행 시 Streamlit 실행 안내"""
    print("\n=== Multi-Agent System ===")
    print("이 앱은 Streamlit으로 실행되어야 합니다.")
    print("실행 방법: streamlit run src/backend/app.py")
    print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    # 직접 실행될 때만 main() 호출
    # Streamlit은 이 부분을 무시하고 위의 코드를 실행
    if not os.getenv("STREAMLIT_RUNTIME"):
        main()
