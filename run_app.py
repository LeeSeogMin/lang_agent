"""
Streamlit 애플리케이션 실행 스크립트
이 스크립트는 AI 검색 에이전트의 메인 실행 파일입니다.
Streamlit 웹 애플리케이션을 시작하고 필요한 환경을 설정합니다.
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
# 이를 통해 프로젝트 내의 모듈들을 어디서든 import할 수 있게 됨
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Streamlit 애플리케이션 실행
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Streamlit 실행을 위한 명령행 인자 설정
    # - server.port: 웹 서버 포트 (8501)
    # - server.address: 서버 주소 (localhost)
    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "src" / "ai_agent" / "ui.py"),  # 메인 UI 파일 경로
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    # Streamlit 서버 시작
    sys.exit(stcli.main())

"""
이 파일의 주요 역할:
1. 프로젝트의 진입점 역할
2. Python 경로 설정으로 모듈 import 문제 해결
3. Streamlit 서버 설정 및 실행
4. 웹 애플리케이션의 기본 포트와 주소 설정

실행 방법:
- 터미널에서 'python run_app.py' 명령으로 실행
- 자동으로 Streamlit 서버가 시작되고 웹 브라우저에서 접속 가능
""" 