"""
검색 결과 및 임베딩 캐싱 기능
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Optional
from ..config.settings import CACHE_DIR, CACHE_TTL

class CacheManager:
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl: int = CACHE_TTL):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        """캐시 키 생성"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """캐시 파일 경로 생성"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """캐시된 값 조회"""
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
        """값을 캐시에 저장"""
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
        """캐시 정리"""
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