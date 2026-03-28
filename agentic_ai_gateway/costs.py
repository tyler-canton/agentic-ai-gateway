"""
Cost Tracking Module
====================

Track token costs per model, set budgets, and get alerts when spending exceeds thresholds.

Features:
- Per-model cost tracking with accurate pricing
- Budget limits with alerts
- Cost breakdown by model, time period, and request type
- Support for custom pricing overrides

Author: Tyler Canton
License: MIT
"""

import logging
import time
from typing import Dict, Optional, Callable, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


# ============================================================================
# Pricing Data (per 1M tokens, as of 2026)
# ============================================================================

# Bedrock pricing (per 1M tokens)
BEDROCK_PRICING = {
    # Claude 4 models
    "anthropic.claude-4-opus": {"input": 15.00, "output": 75.00},
    "anthropic.claude-4-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic.claude-4-haiku": {"input": 0.25, "output": 1.25},
    # Claude 3.x models
    "anthropic.claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic.claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic.claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic.claude-3-haiku": {"input": 0.25, "output": 1.25},
    "anthropic.claude-3-opus": {"input": 15.00, "output": 75.00},
    # Meta Llama models
    "meta.llama3-70b": {"input": 2.65, "output": 3.50},
    "meta.llama3-8b": {"input": 0.30, "output": 0.60},
    "meta.llama3-1-405b": {"input": 5.32, "output": 16.00},
    # Mistral models
    "mistral.mixtral-8x7b": {"input": 0.45, "output": 0.70},
    "mistral.mistral-large": {"input": 4.00, "output": 12.00},
}

# OpenAI pricing (per 1M tokens)
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
}


# ============================================================================
# Cost Tracking
# ============================================================================

@dataclass
class CostRecord:
    """Record of a single request's cost."""
    timestamp: datetime
    model_id: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetAlert:
    """Alert triggered when budget threshold is reached."""
    timestamp: datetime
    budget_name: str
    threshold_percent: int
    current_spend: float
    budget_limit: float
    message: str


class CostTracker:
    """
    Track costs across LLM invocations.
    
    Features:
    - Accurate per-model pricing
    - Budget limits with alerts
    - Cost breakdown by model and time period
    - Thread-safe for concurrent access
    
    Example:
        tracker = CostTracker(budget_limit=100.00)  # $100 budget
        tracker.on_alert(lambda alert: print(f"Alert: {alert.message}"))
        
        # Track a request
        tracker.record(
            model_id="anthropic.claude-4-sonnet",
            input_tokens=1000,
            output_tokens=500
        )
        
        # Check costs
        print(f"Total spend: ${tracker.total_cost:.4f}")
        print(f"By model: {tracker.cost_by_model()}")
    """
    
    def __init__(
        self,
        budget_limit: Optional[float] = None,
        alert_thresholds: List[int] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        reset_period: Optional[timedelta] = None
    ):
        """
        Initialize cost tracker.
        
        Args:
            budget_limit: Maximum spend limit in USD (None = unlimited)
            alert_thresholds: Percentages to trigger alerts (default: [50, 75, 90, 100])
            custom_pricing: Override default pricing for specific models
            reset_period: Auto-reset costs after this period (None = never)
        """
        self.budget_limit = budget_limit
        self.alert_thresholds = alert_thresholds or [50, 75, 90, 100]
        self.reset_period = reset_period
        
        # Merge pricing tables with custom overrides
        self._pricing = {**BEDROCK_PRICING, **OPENAI_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)
        
        # State
        self._records: List[CostRecord] = []
        self._total_cost: float = 0.0
        self._triggered_thresholds: set = set()
        self._alert_callbacks: List[Callable[[BudgetAlert], None]] = []
        self._lock = Lock()
        self._reset_time: Optional[datetime] = None
        
        if reset_period:
            self._reset_time = datetime.now() + reset_period
    
    def on_alert(self, callback: Callable[[BudgetAlert], None]) -> None:
        """Register a callback for budget alerts."""
        self._alert_callbacks.append(callback)
    
    def record(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostRecord:
        """
        Record the cost of a request.
        
        Args:
            model_id: The model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_id: Optional request identifier
            metadata: Optional additional metadata
            
        Returns:
            CostRecord with calculated costs
        """
        with self._lock:
            # Check for reset
            self._check_reset()
            
            # Calculate costs
            pricing = self._get_pricing(model_id)
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost
            
            # Create record
            record = CostRecord(
                timestamp=datetime.now(),
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                request_id=request_id,
                metadata=metadata or {}
            )
            
            # Update state
            self._records.append(record)
            self._total_cost += total_cost
            
            # Log
            logger.info(
                f"[CostTracker] {model_id}: {input_tokens} in + {output_tokens} out = "
                f"${total_cost:.6f} (total: ${self._total_cost:.4f})"
            )
            
            # Check budget alerts
            self._check_alerts()
            
            return record
    
    def _get_pricing(self, model_id: str) -> Dict[str, float]:
        """Get pricing for a model, with fallback matching."""
        # Direct match
        if model_id in self._pricing:
            return self._pricing[model_id]
        
        # Partial match (e.g., "anthropic.claude-4-sonnet-20250812-v1:0")
        for key, pricing in self._pricing.items():
            if key in model_id.lower():
                return pricing
        
        # Default fallback (Sonnet pricing)
        logger.warning(f"[CostTracker] No pricing for {model_id}, using default")
        return {"input": 3.00, "output": 15.00}
    
    def _check_reset(self) -> None:
        """Check if costs should be reset."""
        if self._reset_time and datetime.now() >= self._reset_time:
            logger.info("[CostTracker] Resetting costs for new period")
            self._records = []
            self._total_cost = 0.0
            self._triggered_thresholds = set()
            self._reset_time = datetime.now() + self.reset_period
    
    def _check_alerts(self) -> None:
        """Check and trigger budget alerts."""
        if not self.budget_limit:
            return
        
        percent_used = (self._total_cost / self.budget_limit) * 100
        
        for threshold in self.alert_thresholds:
            if percent_used >= threshold and threshold not in self._triggered_thresholds:
                self._triggered_thresholds.add(threshold)
                
                alert = BudgetAlert(
                    timestamp=datetime.now(),
                    budget_name="default",
                    threshold_percent=threshold,
                    current_spend=self._total_cost,
                    budget_limit=self.budget_limit,
                    message=f"Budget alert: {threshold}% of ${self.budget_limit:.2f} "
                            f"limit reached (${self._total_cost:.4f} spent)"
                )
                
                logger.warning(f"[CostTracker] {alert.message}")
                
                for callback in self._alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"[CostTracker] Alert callback failed: {e}")
    
    @property
    def total_cost(self) -> float:
        """Get total cost so far."""
        with self._lock:
            return self._total_cost
    
    @property
    def budget_remaining(self) -> Optional[float]:
        """Get remaining budget, or None if no limit."""
        if not self.budget_limit:
            return None
        return max(0, self.budget_limit - self._total_cost)
    
    @property
    def budget_percent_used(self) -> Optional[float]:
        """Get percentage of budget used."""
        if not self.budget_limit:
            return None
        return (self._total_cost / self.budget_limit) * 100
    
    def cost_by_model(self) -> Dict[str, float]:
        """Get cost breakdown by model."""
        with self._lock:
            costs = defaultdict(float)
            for record in self._records:
                costs[record.model_id] += record.total_cost
            return dict(costs)
    
    def cost_since(self, since: datetime) -> float:
        """Get cost since a specific time."""
        with self._lock:
            return sum(
                r.total_cost for r in self._records
                if r.timestamp >= since
            )
    
    def tokens_by_model(self) -> Dict[str, Dict[str, int]]:
        """Get token breakdown by model."""
        with self._lock:
            tokens = defaultdict(lambda: {"input": 0, "output": 0})
            for record in self._records:
                tokens[record.model_id]["input"] += record.input_tokens
                tokens[record.model_id]["output"] += record.output_tokens
            return dict(tokens)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of cost tracking."""
        with self._lock:
            return {
                "total_cost": self._total_cost,
                "total_requests": len(self._records),
                "budget_limit": self.budget_limit,
                "budget_remaining": self.budget_remaining,
                "budget_percent_used": self.budget_percent_used,
                "cost_by_model": self.cost_by_model(),
                "tokens_by_model": self.tokens_by_model(),
            }
    
    def reset(self) -> None:
        """Manually reset all cost tracking."""
        with self._lock:
            self._records = []
            self._total_cost = 0.0
            self._triggered_thresholds = set()
            logger.info("[CostTracker] Costs manually reset")
