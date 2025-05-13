"""
검색 결과 및 임베딩 캐싱 기능
이 모듈은 검색 결과와 임베딩을 디스크 기반으로 캐싱하여 성능을 최적화합니다.
TTL(Time To Live) 기반의 캐시 만료 시스템을 구현합니다.
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Optional
from ..config.settings import CACHE_DIR, CACHE_TTL

class CacheManager:
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl: int = CACHE_TTL):
        """
        캐시 관리자 초기화
        
        Args:
            cache_dir (Path): 캐시 파일 저장 디렉토리
            ttl (int): 캐시 유효 시간 (초)
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        """
        캐시 키를 MD5 해시로 변환
        
        Args:
            key (str): 원본 캐시 키
            
        Returns:
            str: MD5 해시된 캐시 키
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        캐시 파일의 전체 경로 생성
        
        Args:
            cache_key (str): 해시된 캐시 키
            
        Returns:
            Path: 캐시 파일 경로
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """
        캐시된 값을 조회합니다.
        
        Args:
            key (str): 캐시 키
            
        Returns:
            Optional[Any]: 캐시된 값 또는 None
                - 캐시가 없거나 만료된 경우 None 반환
                - 캐시 파일이 손상된 경우 None 반환
        """
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # TTL 체크
            if time.time() - cache_data["timestamp"] > self.ttl:
                cache_path.unlink()
                return None
            
            return cache_data["value"]
        except Exception:
            if cache_path.exists():
                cache_path.unlink()
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        값을 캐시에 저장합니다.
        
        Args:
            key (str): 캐시 키
            value (Any): 저장할 값
            
        Returns:
            bool: 저장 성공 여부
        """
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                "key": key,
                "value": value,
                "timestamp": time.time()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
    
    def clear(self, max_age: Optional[int] = None):
        """
        캐시를 정리합니다.
        
        Args:
            max_age (Optional[int]): 최대 캐시 유효 시간 (초)
                - None: 모든 캐시 삭제
                - 정수: 지정된 시간보다 오래된 캐시만 삭제
        """
        now = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                if max_age is not None:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    if now - cache_data["timestamp"] > max_age:
                        cache_file.unlink()
                else:
                    cache_file.unlink()
            except Exception:
                if cache_file.exists():
                    cache_file.unlink()

"""
이 파일의 주요 역할:
1. 디스크 기반 캐시 시스템 구현
2. TTL 기반 캐시 만료 관리
3. 검색 결과 및 임베딩 캐싱
4. 캐시 정리 및 관리

주요 기능:
- 캐시 저장 및 조회
- TTL 기반 만료 처리
- 캐시 키 해싱
- 캐시 정리

사용된 주요 기술:
- JSON 파일 기반 저장
- MD5 해싱
- TTL 기반 만료
- 예외 처리
""" 