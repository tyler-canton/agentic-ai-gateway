"""
Redis Caching Module
====================

Production-ready distributed caching using Redis.
Shares cache across multiple instances behind a load balancer.

Features:
- Redis-backed cache (survives restarts)
- Shared across all server instances
- Same interface as RequestCache
- Optional JSON serialization for complex responses

Author: Tyler Canton
License: MIT
"""

import hashlib
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Redis Cache Entry
# ============================================================================

@dataclass
class RedisCacheEntry:
    """A cached response stored in Redis."""
    key: str
    prompt: str
    response: str
    model_id: str
    input_tokens: int
    output_tokens: int
    created_at: str  # ISO format string for JSON serialization
    hit_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def age_seconds(self) -> float:
        created = datetime.fromisoformat(self.created_at)
        return (datetime.now() - created).total_seconds()

    def to_json(self) -> str:
        """Serialize to JSON for Redis storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "RedisCacheEntry":
        """Deserialize from Redis JSON."""
        return cls(**json.loads(data))


# ============================================================================
# Redis Request Cache
# ============================================================================

class RedisRequestCache:
    """
    Redis-backed cache for LLM responses.

    Production-ready distributed caching that works across
    multiple server instances behind a load balancer.

    Example:
        import redis
        from agentic_ai_gateway.redis_caching import RedisRequestCache

        redis_client = redis.Redis.from_url("redis://localhost:6379")
        cache = RedisRequestCache(redis_client, ttl_seconds=3600)

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

    Requirements:
        pip install redis
    """

    def __init__(
        self,
        redis_client,
        ttl_seconds: int = 3600,
        key_prefix: str = "llm_cache:",
        enabled: bool = True
    ):
        """
        Initialize Redis cache.

        Args:
            redis_client: Redis client instance (from redis-py)
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            key_prefix: Prefix for all Redis keys (namespace isolation)
            enabled: Whether caching is enabled
        """
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.enabled = enabled

        # Statistics (stored in Redis for shared metrics)
        self._stats_key = f"{key_prefix}stats"

    def _generate_key(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate cache key from prompt and parameters."""
        key_parts = [prompt]
        if model_id:
            key_parts.append(f"model:{model_id}")

        for k, v in sorted(kwargs.items()):
            if k in ("temperature", "max_tokens", "system"):
                key_parts.append(f"{k}:{v}")

        key_string = "|".join(key_parts)
        hash_key = hashlib.sha256(key_string.encode()).hexdigest()[:32]
        return f"{self.key_prefix}{hash_key}"

    def _increment_stat(self, stat: str, amount: int = 1) -> None:
        """Increment a statistic in Redis."""
        try:
            self.redis.hincrby(self._stats_key, stat, amount)
        except Exception as e:
            logger.warning(f"[RedisCache] Failed to increment stat: {e}")

    def get(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Optional[RedisCacheEntry]:
        """
        Get cached response for a prompt.

        Args:
            prompt: The prompt to look up
            model_id: Optional model filter
            **kwargs: Additional parameters used in key generation

        Returns:
            RedisCacheEntry if found, None otherwise
        """
        if not self.enabled:
            return None

        key = self._generate_key(prompt, model_id, **kwargs)

        try:
            data = self.redis.get(key)

            if data is None:
                self._increment_stat("misses")
                return None

            # Deserialize entry
            entry = RedisCacheEntry.from_json(data.decode("utf-8"))

            # Update hit count
            entry.hit_count += 1
            self.redis.set(key, entry.to_json(), ex=self.ttl_seconds)

            # Update stats
            self._increment_stat("hits")
            self._increment_stat("tokens_saved", entry.input_tokens + entry.output_tokens)

            logger.info(
                f"[RedisCache] HIT: {key[-8:]}... "
                f"(age: {entry.age_seconds:.0f}s, hits: {entry.hit_count})"
            )

            return entry

        except Exception as e:
            logger.error(f"[RedisCache] GET error: {e}")
            self._increment_stat("misses")
            return None

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
    ) -> Optional[RedisCacheEntry]:
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
            The created RedisCacheEntry, or None on error
        """
        if not self.enabled:
            return None

        key = self._generate_key(prompt, model_id, **kwargs)
        ttl = ttl_seconds or self.ttl_seconds

        entry = RedisCacheEntry(
            key=key,
            prompt=prompt,
            response=response,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        try:
            self.redis.set(key, entry.to_json(), ex=ttl)

            logger.info(
                f"[RedisCache] SET: {key[-8:]}... "
                f"(tokens: {input_tokens}+{output_tokens}, ttl: {ttl}s)"
            )

            return entry

        except Exception as e:
            logger.error(f"[RedisCache] SET error: {e}")
            return None

    def invalidate(self, prompt: str, model_id: Optional[str] = None, **kwargs) -> bool:
        """
        Invalidate a specific cache entry.

        Returns:
            True if entry was found and removed
        """
        key = self._generate_key(prompt, model_id, **kwargs)

        try:
            result = self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"[RedisCache] DELETE error: {e}")
            return False

    def clear(self, confirm: bool = False) -> int:
        """
        Clear all cache entries with this prefix.

        Args:
            confirm: Must be True to actually clear (safety check)

        Returns:
            Number of entries cleared
        """
        if not confirm:
            logger.warning("[RedisCache] clear() called without confirm=True, skipping")
            return 0

        try:
            # Find all keys with our prefix
            pattern = f"{self.key_prefix}*"
            keys = list(self.redis.scan_iter(match=pattern))

            if keys:
                count = self.redis.delete(*keys)
                logger.info(f"[RedisCache] Cleared {count} entries")
                return count
            return 0

        except Exception as e:
            logger.error(f"[RedisCache] CLEAR error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics from Redis."""
        try:
            stats = self.redis.hgetall(self._stats_key)

            hits = int(stats.get(b"hits", 0))
            misses = int(stats.get(b"misses", 0))
            total = hits + misses

            return {
                "backend": "redis",
                "hits": hits,
                "misses": misses,
                "hit_rate": (hits / total * 100) if total > 0 else 0.0,
                "tokens_saved": int(stats.get(b"tokens_saved", 0)),
                "enabled": self.enabled,
                "ttl_seconds": self.ttl_seconds,
                "key_prefix": self.key_prefix,
            }

        except Exception as e:
            logger.error(f"[RedisCache] STATS error: {e}")
            return {"backend": "redis", "error": str(e)}

    def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            self.redis.ping()
            return True
        except Exception:
            return False


# ============================================================================
# Redis Cached Gateway Wrapper
# ============================================================================

class RedisCachedGateway:
    """
    Wrapper that adds Redis caching to any AIGateway.

    Production-ready distributed caching for load-balanced deployments.

    Example:
        import redis
        from agentic_ai_gateway import create_bedrock_gateway
        from agentic_ai_gateway.redis_caching import RedisCachedGateway

        gateway = create_bedrock_gateway(...)

        redis_client = redis.Redis.from_url("redis://localhost:6379")
        cached_gateway = RedisCachedGateway(
            gateway,
            redis_client,
            ttl_seconds=3600
        )

        # First call hits the model
        response1 = cached_gateway.invoke("What is Python?")

        # Second identical call returns cached response (even from different server)
        response2 = cached_gateway.invoke("What is Python?")
        print(response2.metadata["cache_hit"])  # True

    Requirements:
        pip install redis
    """

    def __init__(
        self,
        gateway,
        redis_client,
        ttl_seconds: int = 3600,
        key_prefix: str = "llm_cache:",
        enabled: bool = True
    ):
        """
        Initialize Redis-cached gateway.

        Args:
            gateway: The underlying AIGateway
            redis_client: Redis client instance
            ttl_seconds: Cache TTL (default: 1 hour)
            key_prefix: Redis key namespace
            enabled: Whether caching is enabled
        """
        self.gateway = gateway
        self.cache = RedisRequestCache(
            redis_client=redis_client,
            ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
            enabled=enabled
        )

    def invoke(self, prompt: str, **kwargs):
        """Invoke with Redis caching."""
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
                metadata={
                    "cache_hit": True,
                    "cache_backend": "redis",
                    "cache_age_seconds": cached.age_seconds
                }
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
        response.metadata["cache_backend"] = "redis"

        return response

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def health_check(self) -> bool:
        """Check Redis connection health."""
        return self.cache.health_check()
