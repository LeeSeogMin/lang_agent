"""
문서 관리 및 인덱싱 기능
이 모듈은 문서의 업로드, 저장, 인덱싱, 삭제를 관리합니다.
벡터 저장소를 활용한 문서 검색 기능을 제공합니다.
"""
from typing import Dict, List, Optional, Any
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import uuid
import logging
import gc

from .document_parser import DocumentParser
from .vector_store import VectorStore
from .embedding import EmbeddingManager
from ..config.settings import DOCUMENTS_DIR
from ..models.state import DocumentMetadata

# 로깅 설정
logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(
        self,
        storage_dir: Path = DOCUMENTS_DIR,
        chunk_size: int = 1000
    ):
        """
        문서 관리자 초기화
        
        Args:
            storage_dir (Path): 문서 저장 디렉토리
            chunk_size (int): 문서 청크 크기
        """
        self.storage_dir = storage_dir
        self.chunk_size = chunk_size
        self.metadata_file = storage_dir / "metadata.json"
        
        logger.info(f"문서 관리자 초기화: 저장소={storage_dir}, 청크 크기={chunk_size}")
        
        # 컴포넌트 초기화
        self.document_parser = DocumentParser()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(
            dimension=self.embedding_manager.embedding_dim
        )
        
        # 디렉토리 생성
        self.storage_dir.mkdir(exist_ok=True)
        
        # 메타데이터 로드
        self.document_metadata: Dict[str, DocumentMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """
        메타데이터 파일 로드
        JSON 파일에서 문서 메타데이터를 로드하여 메모리에 저장합니다.
        """
        if self.metadata_file.exists():
            logger.info(f"메타데이터 파일 로드: {self.metadata_file}")
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.document_metadata = {
                    doc_id: DocumentMetadata(**metadata)
                    for doc_id, metadata in data.items()
                }
            logger.info(f"메타데이터 로드 완료: {len(self.document_metadata)}개 문서")
    
    def _save_metadata(self):
        """
        메타데이터 파일 저장
        현재 메모리에 있는 문서 메타데이터를 JSON 파일로 저장합니다.
        """
        logger.info(f"메타데이터 저장: {self.metadata_file}")
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            data = {
                doc_id: metadata.model_dump()
                for doc_id, metadata in self.document_metadata.items()
            }
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("메타데이터 저장 완료")
    
    def upload_document(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        문서 업로드 및 저장
        
        Args:
            file_path (str): 업로드할 파일 경로
            custom_metadata (Optional[Dict[str, Any]]): 추가 메타데이터
            
        Returns:
            str: 생성된 문서 ID
            
        Raises:
            ValueError: 지원하지 않는 파일 형식인 경우
        """
        logger.info(f"문서 업로드 시작: {file_path}")
        
        # 파일 확장자 확인
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.txt', '.pdf', '.docx']:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")
        
        # 문서 ID 생성 및 저장 경로 설정
        doc_id = str(uuid.uuid4())
        save_path = self.storage_dir / f"{doc_id}{ext}"
        
        # 파일 복사
        shutil.copy2(file_path, save_path)
        logger.info(f"파일 복사 완료: {save_path}")
        
        # 메타데이터 생성
        metadata = DocumentMetadata(
            doc_id=doc_id,
            original_filename=os.path.basename(file_path),
            file_path=str(save_path),
            file_type=ext[1:],  # 점 제거
            uploaded_at=datetime.now().isoformat(),
            custom_metadata=custom_metadata or {}
        )
        
        # 메타데이터 저장
        self.document_metadata[doc_id] = metadata
        self._save_metadata()
        
        logger.info(f"문서 업로드 완료: {doc_id}")
        return doc_id
    
    def index_document(self, doc_id: str) -> int:
        """
        문서를 벡터 저장소에 인덱싱
        
        Args:
            doc_id (str): 인덱싱할 문서 ID
            
        Returns:
            int: 인덱싱된 청크 수
            
        Raises:
            ValueError: 문서를 찾을 수 없는 경우
        """
        logger.info(f"문서 인덱싱 시작: {doc_id}")
        
        metadata = self.document_metadata.get(doc_id)
        if not metadata:
            raise ValueError(f"문서 ID {doc_id}를 찾을 수 없습니다.")
        
        try:
            # 문서 파싱
            logger.info(f"문서 파싱 시작: {metadata.original_filename}")
            chunks = self.document_parser.parse(
                metadata.file_path,
                chunk_size=self.chunk_size
            )
            
            if not chunks:
                logger.warning("파싱된 청크가 없습니다")
                return 0
            
            chunk_list = list(chunks)  # 제너레이터를 리스트로 변환
            logger.info(f"파싱 완료: {len(chunk_list)}개 청크")
            
            # 임베딩 생성 (배치 처리)
            logger.info("임베딩 생성 시작")
            embeddings = self.embedding_manager.encode_batch(chunk_list)
            logger.info(f"임베딩 생성 완료: {len(embeddings)}개")
            
            # 메모리 정리
            del chunks
            gc.collect()
            
            # 벡터 저장소에 추가
            logger.info("벡터 저장소에 문서 추가 중")
            chunk_ids = self.vector_store.add_documents(
                [
                    {
                        "content": chunk,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "title": f"{metadata.original_filename} - 청크 {i+1}",
                        "metadata": metadata.model_dump()
                    }
                    for i, chunk in enumerate(chunk_list)
                ],
                embeddings
            )
            logger.info(f"벡터 저장소 추가 완료: {len(chunk_ids)}개 청크")
            
            # 메모리 정리
            del chunk_list
            del embeddings
            gc.collect()
            
            # 메타데이터 업데이트
            metadata.chunk_count = len(chunk_ids)
            metadata.chunk_ids = [str(chunk_id) for chunk_id in chunk_ids]
            self._save_metadata()
            logger.info("메타데이터 업데이트 완료")
            
            # 벡터 저장소 상태 저장
            self.vector_store.save()
            logger.info("벡터 저장소 상태 저장 완료")
            
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"문서 인덱싱 중 오류 발생: {str(e)}", exc_info=True)
            # 실패 시 메타데이터 초기화
            metadata.chunk_count = 0
            metadata.chunk_ids = []
            self._save_metadata()
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """
        문서 삭제
        
        Args:
            doc_id (str): 삭제할 문서 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        logger.info(f"문서 삭제 시작: {doc_id}")
        
        metadata = self.document_metadata.get(doc_id)
        if not metadata:
            logger.warning(f"삭제할 문서를 찾을 수 없음: {doc_id}")
            return False
        
        # 파일 삭제
        try:
            os.remove(metadata.file_path)
            logger.info(f"파일 삭제 완료: {metadata.file_path}")
        except FileNotFoundError:
            logger.warning(f"파일을 찾을 수 없음: {metadata.file_path}")
            pass
        
        # 메타데이터 삭제
        del self.document_metadata[doc_id]
        self._save_metadata()
        logger.info("메타데이터에서 문서 정보 삭제 완료")
        
        return True
    
    def get_document_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        문서 메타데이터 조회
        
        Args:
            doc_id (str): 조회할 문서 ID
            
        Returns:
            Optional[DocumentMetadata]: 문서 메타데이터 또는 None
        """
        return self.document_metadata.get(doc_id)
    
    def list_documents(self) -> Dict[str, DocumentMetadata]:
        """
        모든 문서 메타데이터 조회
        
        Returns:
            Dict[str, DocumentMetadata]: 문서 ID를 키로 하는 메타데이터 딕셔너리
        """
        return self.document_metadata.copy()
    
    def save_state(self):
        """
        현재 상태 저장
        메타데이터와 벡터 저장소의 상태를 저장합니다.
        """
        self._save_metadata()
        self.vector_store.save()
    
    def load_state(self):
        """
        저장된 상태 로드
        메타데이터와 벡터 저장소의 상태를 로드합니다.
        """
        self._load_metadata()
        self.vector_store.load()

"""
이 파일의 주요 역할:
1. 문서 관리 시스템 구현
2. 문서 업로드 및 저장
3. 문서 인덱싱 및 벡터화
4. 메타데이터 관리

주요 기능:
- 문서 업로드 및 저장
- 문서 파싱 및 청크 분할
- 임베딩 생성
- 벡터 저장소 관리
- 메타데이터 관리

사용된 주요 기술:
- UUID 기반 문서 식별
- 벡터 임베딩
- 메모리 관리 (GC)
- 로깅 시스템
""" 