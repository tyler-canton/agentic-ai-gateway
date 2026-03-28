"""Tests for Request Caching module."""

import pytest
import time
from datetime import datetime, timedelta
from agentic_ai_gateway.caching import (
    RequestCache,
    CacheEntry,
    CachedGateway,
)


class TestRequestCache:
    """Tests for RequestCache."""
    
    def test_basic_set_get(self):
        """Test basic cache set and get."""
        cache = RequestCache(ttl_seconds=3600)
        
        cache.set(
            prompt="What is Python?",
            response="Python is a programming language.",
            model_id="anthropic.claude-4-sonnet",
            input_tokens=10,
            output_tokens=20
        )
        
        entry = cache.get("What is Python?")
        
        assert entry is not None
        assert entry.response == "Python is a programming language."
        assert entry.model_id == "anthropic.claude-4-sonnet"
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = RequestCache()
        
        entry = cache.get("Unknown prompt")
        
        assert entry is None
    
    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = RequestCache(ttl_seconds=1)  # 1 second TTL
        
        cache.set(
            prompt="Test prompt",
            response="Test response",
            model_id="test-model"
        )
        
        # Should be cached
        assert cache.get("Test prompt") is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("Test prompt") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = RequestCache(max_size=2)
        
        cache.set("prompt1", "response1", "model")
        cache.set("prompt2", "response2", "model")
        
        # Access prompt1 to make it most recently used
        cache.get("prompt1")
        
        # Add third entry, should evict prompt2
        cache.set("prompt3", "response3", "model")
        
        assert cache.get("prompt1") is not None
        assert cache.get("prompt2") is None  # Evicted
        assert cache.get("prompt3") is not None
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = RequestCache()
        
        cache.set("test", "response", "model")
        
        cache.get("test")  # Hit
        cache.get("test")  # Hit
        cache.get("missing")  # Miss
        
        assert cache.hit_rate == pytest.approx(66.67, rel=0.1)
    
    def test_disabled_cache(self):
        """Test cache when disabled."""
        cache = RequestCache(enabled=False)
        
        result = cache.set("prompt", "response", "model")
        assert result is None
        
        entry = cache.get("prompt")
        assert entry is None
    
    def test_invalidate(self):
        """Test cache invalidation."""
        cache = RequestCache()
        
        cache.set("test", "response", "model")
        assert cache.get("test") is not None
        
        result = cache.invalidate("test")
        assert result is True
        
        assert cache.get("test") is None
    
    def test_clear(self):
        """Test clearing all entries."""
        cache = RequestCache()
        
        cache.set("prompt1", "response1", "model")
        cache.set("prompt2", "response2", "model")
        
        count = cache.clear()
        
        assert count == 2
        assert cache.size == 0
    
    def test_key_includes_model(self):
        """Test that different models have different cache keys."""
        cache = RequestCache()
        
        cache.set("same prompt", "response1", "model-a")
        cache.set("same prompt", "response2", "model-b")
        
        entry_a = cache.get("same prompt", model_id="model-a")
        entry_b = cache.get("same prompt", model_id="model-b")
        
        assert entry_a.response == "response1"
        assert entry_b.response == "response2"
    
    def test_stats(self):
        """Test get_stats."""
        cache = RequestCache(max_size=100, ttl_seconds=3600)
        
        cache.set("test", "response", "model")
        cache.get("test")
        
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 100.0
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = RequestCache(ttl_seconds=1)
        
        cache.set("prompt1", "response1", "model")
        cache.set("prompt2", "response2", "model")
        
        time.sleep(1.1)
        
        removed = cache.cleanup_expired()
        
        assert removed == 2
        assert cache.size == 0


class TestCacheEntry:
    """Tests for CacheEntry."""
    
    def test_is_expired(self):
        """Test expiration check."""
        entry = CacheEntry(
            key="test",
            prompt="test",
            response="test",
            model_id="test",
            input_tokens=0,
            output_tokens=0,
            created_at=datetime.now(),
            expires_at=datetime.now() - timedelta(seconds=1)  # Already expired
        )
        
        assert entry.is_expired is True
    
    def test_age_seconds(self):
        """Test age calculation."""
        entry = CacheEntry(
            key="test",
            prompt="test",
            response="test",
            model_id="test",
            input_tokens=0,
            output_tokens=0,
            created_at=datetime.now() - timedelta(seconds=5),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        assert entry.age_seconds >= 5
