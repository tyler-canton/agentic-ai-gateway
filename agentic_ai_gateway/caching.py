"""
Request Caching Module
======================

Cache LLM responses to save costs on repeated queries.

Features:
- In-memory caching with TTL
- Cache key generation from prompts
- Cache statistics and hit rates
- Optional semantic similarity matching

Author: Tyler Canton
License: MIT
"""

import hashlib
import logging
import time
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from collections import OrderedDict

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Entry
# ============================================================================

@dataclass
class CacheEntry:
    """A cached response."""
    key: str
    prompt: str
    response: str
    model_id: str
    input_tokens: int
    output_tokens: int
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at
    
    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()


# ============================================================================
# Request Cache
# ============================================================================

class RequestCache:
    """
    Cache for LLM responses.
    
    Uses exact prompt matching by default. Caches responses with TTL
    and provides statistics on cache performance.
    
    Example:
        cache = RequestCache(ttl_seconds=3600, max_size=1000)
        
        # Check cache
        cached = cache.get(prompt, model_id="anthropic.claude-4-sonnet")
        if cached:
            return cached.response
        
        # Call model...
        response = gateway.invoke(prompt)
        
        # Store in cache
        cache.set(
            prompt=prompt,
            response=response.content,
            model_id=response.model_used,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens
        )
    """
    
    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        enabled: bool = True
    ):
        """
        Initialize request cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            max_size: Maximum number of entries (LRU eviction)
            enabled: Whether caching is enabled
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.enabled = enabled
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._tokens_saved = 0
        self._cost_saved = 0.0
    
    def _generate_key(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate cache key from prompt and parameters."""
        # Include relevant parameters in key
        key_parts = [prompt]
        if model_id:
            key_parts.append(f"model:{model_id}")
        
        # Include any other kwargs that affect output
        for k, v in sorted(kwargs.items()):
            if k in ("temperature", "max_tokens", "system"):
                key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def get(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Optional[CacheEntry]:
        """
        Get cached response for a prompt.
        
        Args:
            prompt: The prompt to look up
            model_id: Optional model filter
            **kwargs: Additional parameters used in key generation
            
        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        if not self.enabled:
            return None
        
        key = self._generate_key(prompt, model_id, **kwargs)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired:
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                return None
            
            # Cache hit
            self._hits += 1
            entry.hit_count += 1
            self._tokens_saved += entry.input_tokens + entry.output_tokens
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            logger.info(
                f"[Cache] HIT: {key[:8]}... "
                f"(age: {entry.age_seconds:.0f}s, hits: {entry.hit_count})"
            )
            
            return entry
    
    def set(
        self,
        prompt: str,
        response: str,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> CacheEntry:
        """
        Cache a response.
        
        Args:
            prompt: The prompt
            response: The response to cache
            model_id: The model used
            input_tokens: Input token count
            output_tokens: Output token count
            ttl_seconds: Override default TTL
            metadata: Additional metadata
            **kwargs: Additional parameters for key generation
            
        Returns:
            The created CacheEntry
        """
        if not self.enabled:
            return None
        
        key = self._generate_key(prompt, model_id, **kwargs)
        ttl = ttl_seconds or self.ttl_seconds
        
        entry = CacheEntry(
            key=key,
            prompt=prompt,
            response=response,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=ttl),
            metadata=metadata or {}
        )
        
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(f"[Cache] Evicted: {oldest_key[:8]}...")
            
            self._cache[key] = entry
            
        logger.info(
            f"[Cache] SET: {key[:8]}... "
            f"(tokens: {input_tokens}+{output_tokens}, ttl: {ttl}s)"
        )
        
        return entry
    
    def invalidate(self, prompt: str, model_id: Optional[str] = None, **kwargs) -> bool:
        """
        Invalidate a specific cache entry.
        
        Returns:
            True if entry was found and removed
        """
        key = self._generate_key(prompt, model_id, **kwargs)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"[Cache] Cleared {count} entries")
            return count
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"[Cache] Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "evictions": self._evictions,
            "tokens_saved": self._tokens_saved,
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
        }


# ============================================================================
# Cached Gateway Wrapper
# ============================================================================

class CachedGateway:
    """
    Wrapper that adds caching to any AIGateway.
    
    Example:
        from agentic_ai_gateway import create_bedrock_gateway
        from agentic_ai_gateway.caching import CachedGateway
        
        gateway = create_bedrock_gateway(...)
        cached_gateway = CachedGateway(gateway, ttl_seconds=3600)
        
        # First call hits the model
        response1 = cached_gateway.invoke("What is Python?")
        
        # Second identical call returns cached response
        response2 = cached_gateway.invoke("What is Python?")
        print(response2.metadata["cache_hit"])  # True
    """
    
    def __init__(
        self,
        gateway,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        enabled: bool = True
    ):
        self.gateway = gateway
        self.cache = RequestCache(
            ttl_seconds=ttl_seconds,
            max_size=max_size,
            enabled=enabled
        )
    
    def invoke(self, prompt: str, **kwargs):
        """Invoke with caching."""
        # Check cache
        cached = self.cache.get(prompt, **kwargs)
        if cached:
            # Return cached response with metadata
            from .gateway import AIGatewayResponse
            return AIGatewayResponse(
                content=cached.response,
                model_used=cached.model_id,
                latency_ms=0,
                fallback_used=False,
                canary_used=False,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                metadata={"cache_hit": True, "cache_age_seconds": cached.age_seconds}
            )
        
        # Call gateway
        response = self.gateway.invoke(prompt, **kwargs)
        
        # Cache response
        self.cache.set(
            prompt=prompt,
            response=response.content,
            model_id=response.model_used,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            **kwargs
        )
        
        # Add cache metadata
        response.metadata["cache_hit"] = False
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
