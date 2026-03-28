"""Tests for Cost Tracking module."""

import pytest
from datetime import datetime, timedelta
from agentic_ai_gateway.costs import (
    CostTracker,
    CostRecord,
    BudgetAlert,
    BEDROCK_PRICING,
    OPENAI_PRICING,
)


class TestCostTracker:
    """Tests for CostTracker."""
    
    def test_basic_recording(self):
        """Test basic cost recording."""
        tracker = CostTracker()
        
        record = tracker.record(
            model_id="anthropic.claude-4-sonnet",
            input_tokens=1000,
            output_tokens=500
        )
        
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.total_cost > 0
        assert tracker.total_cost == record.total_cost
    
    def test_cost_calculation(self):
        """Test cost calculation accuracy."""
        tracker = CostTracker()
        
        # Claude 4 Sonnet: $3/1M input, $15/1M output
        record = tracker.record(
            model_id="anthropic.claude-4-sonnet",
            input_tokens=1_000_000,  # 1M tokens
            output_tokens=1_000_000
        )
        
        expected_input_cost = 3.00
        expected_output_cost = 15.00
        
        assert abs(record.input_cost - expected_input_cost) < 0.01
        assert abs(record.output_cost - expected_output_cost) < 0.01
    
    def test_budget_alerts(self):
        """Test budget alert triggering."""
        alerts_received = []
        
        tracker = CostTracker(
            budget_limit=10.00,
            alert_thresholds=[50, 100]
        )
        tracker.on_alert(lambda a: alerts_received.append(a))
        
        # Use enough tokens to hit 50% budget
        # Claude 4 Sonnet: $3/1M input = need ~1.67M tokens for $5
        tracker.record(
            model_id="anthropic.claude-4-sonnet",
            input_tokens=1_700_000,
            output_tokens=0
        )
        
        assert len(alerts_received) == 1
        assert alerts_received[0].threshold_percent == 50
    
    def test_budget_remaining(self):
        """Test budget remaining calculation."""
        tracker = CostTracker(budget_limit=100.00)
        
        tracker.record(
            model_id="anthropic.claude-4-haiku",
            input_tokens=1_000_000,
            output_tokens=0
        )
        
        # Haiku: $0.25/1M input = $0.25 spent
        assert tracker.budget_remaining is not None
        assert tracker.budget_remaining < 100.00
        assert tracker.budget_remaining > 99.00
    
    def test_cost_by_model(self):
        """Test cost breakdown by model."""
        tracker = CostTracker()
        
        tracker.record("anthropic.claude-4-sonnet", 1000, 500)
        tracker.record("anthropic.claude-4-haiku", 2000, 1000)
        tracker.record("anthropic.claude-4-sonnet", 500, 250)
        
        by_model = tracker.cost_by_model()
        
        assert "anthropic.claude-4-sonnet" in by_model
        assert "anthropic.claude-4-haiku" in by_model
    
    def test_tokens_by_model(self):
        """Test token breakdown by model."""
        tracker = CostTracker()
        
        tracker.record("anthropic.claude-4-sonnet", 1000, 500)
        tracker.record("anthropic.claude-4-sonnet", 500, 250)
        
        tokens = tracker.tokens_by_model()
        
        assert tokens["anthropic.claude-4-sonnet"]["input"] == 1500
        assert tokens["anthropic.claude-4-sonnet"]["output"] == 750
    
    def test_custom_pricing(self):
        """Test custom pricing override."""
        custom_pricing = {
            "my-custom-model": {"input": 1.00, "output": 2.00}
        }
        
        tracker = CostTracker(custom_pricing=custom_pricing)
        
        record = tracker.record(
            model_id="my-custom-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000
        )
        
        assert abs(record.input_cost - 1.00) < 0.01
        assert abs(record.output_cost - 2.00) < 0.01
    
    def test_reset(self):
        """Test manual reset."""
        tracker = CostTracker()
        
        tracker.record("anthropic.claude-4-sonnet", 1000, 500)
        assert tracker.total_cost > 0
        
        tracker.reset()
        
        assert tracker.total_cost == 0
    
    def test_summary(self):
        """Test get_summary."""
        tracker = CostTracker(budget_limit=100.00)
        
        tracker.record("anthropic.claude-4-sonnet", 1000, 500)
        
        summary = tracker.get_summary()
        
        assert "total_cost" in summary
        assert "total_requests" in summary
        assert "budget_limit" in summary
        assert "cost_by_model" in summary
        assert summary["total_requests"] == 1


class TestPricing:
    """Tests for pricing data."""
    
    def test_bedrock_pricing_exists(self):
        """Test that Bedrock pricing data exists."""
        assert len(BEDROCK_PRICING) > 0
        assert "anthropic.claude-4-sonnet" in BEDROCK_PRICING
    
    def test_openai_pricing_exists(self):
        """Test that OpenAI pricing data exists."""
        assert len(OPENAI_PRICING) > 0
        assert "gpt-4o" in OPENAI_PRICING
    
    def test_pricing_has_required_fields(self):
        """Test pricing entries have input/output."""
        for model, pricing in BEDROCK_PRICING.items():
            assert "input" in pricing, f"{model} missing input pricing"
            assert "output" in pricing, f"{model} missing output pricing"
