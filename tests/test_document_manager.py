"""
문서 관리자 테스트
"""
import os
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime
import numpy as np
from src.ai_agent.core.document_manager import DocumentManager
from src.ai_agent.models.state import DocumentMetadata

@pytest.fixture
def mock_document_parser():
    """모의 문서 파서"""
    mock = Mock()
    mock.parse.return_value = ["청크 1"]  # 단일 청크 반환
    return mock

@pytest.fixture
def mock_embedding_manager():
    """모의 임베딩 매니저"""
    mock = Mock()
    mock.embedding_dim = 768
    mock.encode.return_value = np.random.random(768).tolist()
    return mock

@pytest.fixture
def mock_vector_store():
    """모의 벡터 저장소"""
    mock = Mock()
    mock.add_documents.return_value = [0]  # 단일 문서 ID 반환
    return mock

@pytest.fixture
def document_manager(tmp_path, mock_document_parser, mock_embedding_manager, mock_vector_store):
    """테스트용 문서 관리자 인스턴스"""
    with patch('src.ai_agent.core.document_manager.DocumentParser') as mock_dp_cls, \
         patch('src.ai_agent.core.document_manager.EmbeddingManager') as mock_em_cls, \
         patch('src.ai_agent.core.document_manager.VectorStore') as mock_vs_cls:
        mock_dp_cls.return_value = mock_document_parser
        mock_em_cls.return_value = mock_embedding_manager
        mock_vs_cls.return_value = mock_vector_store
        manager = DocumentManager(storage_dir=tmp_path)
        return manager

@pytest.fixture
def sample_files(tmp_path):
    """테스트용 샘플 파일 생성"""
    # TXT 파일
    txt_path = tmp_path / "test.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("이것은 테스트 문서입니다.\n" * 10)
    
    # PDF 파일
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import letter
    
    # 한글 폰트 등록
    try:
        pdfmetrics.registerFont(TTFont('AppleGothic', '/System/Library/Fonts/AppleGothic.ttf'))
        font_name = 'AppleGothic'
    except:
        font_name = 'Helvetica'
    
    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont(font_name, 12)
    c.drawString(100, 700, "PDF 테스트")
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

def test_upload_document(document_manager, sample_files):
    """문서 업로드 테스트"""
    # TXT 파일 업로드
    doc_id = document_manager.upload_document(
        str(sample_files["txt"]),
        custom_metadata={"type": "test"}
    )
    
    assert isinstance(doc_id, str)
    metadata = document_manager.get_document_metadata(doc_id)
    assert metadata is not None
    assert metadata.original_filename == "test.txt"
    assert metadata.file_type == "txt"
    assert metadata.custom_metadata["type"] == "test"
    assert Path(metadata.file_path).exists()

def test_index_document(document_manager, sample_files):
    """문서 인덱싱 테스트"""
    # 문서 업로드
    doc_id = document_manager.upload_document(str(sample_files["txt"]))
    
    # 인덱싱
    chunk_count = document_manager.index_document(doc_id)
    assert chunk_count == 1  # 단일 청크
    
    # 메타데이터 확인
    metadata = document_manager.get_document_metadata(doc_id)
    assert metadata.chunk_count == chunk_count
    assert len(metadata.chunk_ids) == chunk_count
    assert metadata.chunk_ids == ["0"]  # 단일 문서 ID

def test_delete_document(document_manager, sample_files):
    """문서 삭제 테스트"""
    # 문서 업로드
    doc_id = document_manager.upload_document(str(sample_files["txt"]))
    file_path = document_manager.get_document_metadata(doc_id).file_path
    
    # 삭제
    assert document_manager.delete_document(doc_id)
    assert not Path(file_path).exists()
    assert document_manager.get_document_metadata(doc_id) is None

def test_list_documents(document_manager, sample_files):
    """문서 목록 조회 테스트"""
    # 여러 문서 업로드
    doc_ids = []
    for file_path in sample_files.values():
        doc_id = document_manager.upload_document(str(file_path))
        doc_ids.append(doc_id)
    
    # 목록 조회
    documents = document_manager.list_documents()
    assert len(documents) == len(doc_ids)
    assert all(doc_id in documents for doc_id in doc_ids)

def test_save_and_load_state(document_manager, sample_files, tmp_path):
    """상태 저장 및 로드 테스트"""
    # 문서 업로드 및 인덱싱
    doc_id = document_manager.upload_document(str(sample_files["txt"]))
    document_manager.index_document(doc_id)
    
    # 상태 저장
    document_manager.save_state()
    
    # 새 인스턴스 생성 및 상태 로드
    with patch('src.ai_agent.core.document_manager.DocumentParser') as mock_dp_cls, \
         patch('src.ai_agent.core.document_manager.EmbeddingManager') as mock_em_cls, \
         patch('src.ai_agent.core.document_manager.VectorStore') as mock_vs_cls:
        mock_dp_cls.return_value = Mock()
        mock_em_cls.return_value = Mock(embedding_dim=768)
        mock_vs_cls.return_value = Mock()
        new_manager = DocumentManager(storage_dir=tmp_path)
        new_manager.load_state()
    
    # 상태 비교
    assert len(new_manager.list_documents()) == len(document_manager.list_documents())
    assert new_manager.get_document_metadata(doc_id) is not None

def test_invalid_file_type(document_manager, tmp_path):
    """지원하지 않는 파일 형식 테스트"""
    invalid_file = tmp_path / "test.xyz"
    invalid_file.touch()
    
    with pytest.raises(ValueError) as exc_info:
        document_manager.upload_document(str(invalid_file))
    assert "지원하지 않는 파일 형식" in str(exc_info.value)

def test_nonexistent_document(document_manager):
    """존재하지 않는 문서 처리 테스트"""
    with pytest.raises(ValueError) as exc_info:
        document_manager.index_document("nonexistent_id")
    assert "찾을 수 없습니다" in str(exc_info.value)
    
    assert not document_manager.delete_document("nonexistent_id")
    assert document_manager.get_document_metadata("nonexistent_id") is None 