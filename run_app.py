"""
Streamlit 애플리케이션 실행 스크립트
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Streamlit 애플리케이션 실행
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "src" / "ai_agent" / "ui.py"),
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    sys.exit(stcli.main()) 