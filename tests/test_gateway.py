"""Tests for core AI Gateway."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agentic_ai_gateway.gateway import (
    AIGateway,
    AIGatewayConfig,
    AIGatewayResponse,
    InMemoryMetrics,
)


class TestAIGatewayConfig:
    """Tests for AIGatewayConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AIGatewayConfig(primary_model="test-model")
        
        assert config.primary_model == "test-model"
        assert config.canary_model is None
        assert config.canary_percentage == 0
        assert config.fallback_models == []
    
    def test_full_config(self):
        """Test full configuration."""
        config = AIGatewayConfig(
            primary_model="primary",
            canary_model="canary",
            canary_percentage=10,
            fallback_models=["fallback1", "fallback2"]
        )
        
        assert config.canary_model == "canary"
        assert config.canary_percentage == 10
        assert len(config.fallback_models) == 2


class TestInMemoryMetrics:
    """Tests for InMemoryMetrics."""
    
    def test_record_success(self):
        """Test recording successful invocation."""
        metrics = InMemoryMetrics()
        
        metrics.record(
            model_id="test-model",
            latency_ms=100,
            success=True,
            is_canary=False,
            is_fallback=False
        )
        
        stats = metrics.get_stats()
        assert stats["total_invocations"] == 1
        assert stats["total_errors"] == 0
    
    def test_record_error(self):
        """Test recording failed invocation."""
        metrics = InMemoryMetrics()
        
        metrics.record(
            model_id="test-model",
            latency_ms=50,
            success=False,
            is_canary=False,
            is_fallback=False,
            error="Test error"
        )
        
        stats = metrics.get_stats()
        assert stats["total_errors"] == 1
        assert stats["error_rate"] == 1.0
    
    def test_fallback_tracking(self):
        """Test fallback tracking."""
        metrics = InMemoryMetrics()
        
        metrics.record("model1", 100, True, False, False)
        metrics.record("model2", 100, True, False, True)  # Fallback
        
        stats = metrics.get_stats()
        assert stats["fallback_rate"] == 0.5
    
    def test_per_model_stats(self):
        """Test per-model statistics."""
        metrics = InMemoryMetrics()
        
        metrics.record("model-a", 100, True, False, False)
        metrics.record("model-b", 200, True, False, False)
        metrics.record("model-a", 150, True, False, False)
        
        stats = metrics.get_stats()
        
        assert stats["by_model"]["model-a"]["invocations"] == 2
        assert stats["by_model"]["model-b"]["invocations"] == 1


class TestAIGateway:
    """Tests for AIGateway."""
    
    def test_model_selection_no_canary(self):
        """Test model selection without canary."""
        config = AIGatewayConfig(primary_model="primary")
        gateway = AIGateway(config)
        
        model, is_canary = gateway._select_model()
        
        assert model == "primary"
        assert is_canary is False
    
    def test_build_model_chain(self):
        """Test model chain building."""
        config = AIGatewayConfig(
            primary_model="primary",
            fallback_models=["fallback1", "fallback2"]
        )
        gateway = AIGateway(config)
        
        chain = gateway._build_model_chain()
        
        assert chain[0] == "primary"
        assert "fallback1" in chain
        assert "fallback2" in chain
    
    def test_force_model(self):
        """Test forcing specific model."""
        config = AIGatewayConfig(
            primary_model="primary",
            fallback_models=["fallback"]
        )
        gateway = AIGateway(config)
        
        chain = gateway._build_model_chain(force_model="forced")
        
        assert chain == ["forced"]
    
    def test_update_config(self):
        """Test runtime config update."""
        config = AIGatewayConfig(primary_model="primary")
        gateway = AIGateway(config)
        
        gateway.update_config(
            canary_model="new-canary",
            canary_percentage=25
        )
        
        assert gateway.config.canary_model == "new-canary"
        assert gateway.config.canary_percentage == 25


class TestAIGatewayResponse:
    """Tests for AIGatewayResponse."""
    
    def test_response_fields(self):
        """Test response field access."""
        response = AIGatewayResponse(
            content="Test response",
            model_used="test-model",
            latency_ms=100,
            fallback_used=False,
            canary_used=True,
            input_tokens=10,
            output_tokens=20
        )
        
        assert response.content == "Test response"
        assert response.model_used == "test-model"
        assert response.latency_ms == 100
        assert response.input_tokens == 10
        assert response.output_tokens == 20
