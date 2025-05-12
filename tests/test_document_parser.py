"""
문서 파서 테스트
"""
import os
import pytest
from pathlib import Path
from src.ai_agent.core.document_parser import DocumentParser
from src.ai_agent.config.settings import DOCUMENTS_DIR

@pytest.fixture
def document_parser():
    """테스트용 문서 파서 인스턴스"""
    return DocumentParser()

@pytest.fixture
def sample_files(tmp_path):
    """테스트용 샘플 파일 생성"""
    # TXT 파일
    txt_path = tmp_path / "test.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("이것은 테스트 문서입니다.\n" * 10)  # 줄임
    
    # PDF 파일 (텍스트만 포함)
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import letter
    
    # 한글 폰트 등록 (시스템에 설치된 폰트 사용)
    try:
        pdfmetrics.registerFont(TTFont('AppleGothic', '/System/Library/Fonts/AppleGothic.ttf'))
        font_name = 'AppleGothic'
    except:
        font_name = 'Helvetica'  # 폴백 폰트
    
    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont(font_name, 12)
    c.drawString(100, 700, "PDF 테스트 페이지")
    c.save()
    
    # DOCX 파일
    from docx import Document
    docx_path = tmp_path / "test.docx"
    doc = Document()
    doc.add_paragraph("DOCX 테스트 문단")
    doc.save(docx_path)
    
    return {
        "txt": txt_path,
        "pdf": pdf_path,
        "docx": docx_path
    }

def test_parse_txt(document_parser, sample_files):
    """TXT 파일 파싱 테스트"""
    chunks = document_parser.parse(str(sample_files["txt"]))
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)
    assert "테스트 문서" in chunks[0]

def test_parse_pdf(document_parser, sample_files):
    """PDF 파일 파싱 테스트"""
    chunks = document_parser.parse(str(sample_files["pdf"]))
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)
    assert "PDF" in chunks[0]  # 일단 PDF라는 텍스트만 확인

def test_parse_docx(document_parser, sample_files):
    """DOCX 파일 파싱 테스트"""
    chunks = document_parser.parse(str(sample_files["docx"]))
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)
    assert "DOCX 테스트" in chunks[0]

def test_text_chunking(document_parser):
    """텍스트 청킹 테스트"""
    text = "테스트 문장입니다. " * 20  # 줄임
    chunks = list(document_parser.chunk_text(text, chunk_size=100, overlap=20))
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)

def test_invalid_file_type(document_parser, tmp_path):
    """지원하지 않는 파일 형식 테스트"""
    invalid_file = tmp_path / "test.xyz"
    invalid_file.touch()
    
    with pytest.raises(ValueError) as exc_info:
        document_parser.parse(str(invalid_file))
    assert "지원하지 않는 파일 형식" in str(exc_info.value)

def test_empty_file(document_parser, tmp_path):
    """빈 파일 처리 테스트"""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    
    chunks = document_parser.parse(str(empty_file))
    assert isinstance(chunks, list)
    assert len(chunks) == 0

def test_unicode_handling(document_parser, tmp_path):
    """유니코드 텍스트 처리 테스트"""
    unicode_file = tmp_path / "unicode.txt"
    text = "한글 테스트 🌟"  # 줄임
    with open(unicode_file, "w", encoding="utf-8") as f:
        f.write(text)
    
    chunks = document_parser.parse(str(unicode_file))
    assert len(chunks) > 0
    assert text in chunks[0] 