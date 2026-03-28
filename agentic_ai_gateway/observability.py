"""
Observability Module
====================

CloudWatch integration for automatic metrics publishing.

Features:
- Automatic metric publishing to CloudWatch
- Custom metric dimensions
- Batch metric publishing
- Dashboard-ready metrics

Author: Tyler Canton
License: MIT
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock, Thread
from collections import defaultdict
from queue import Queue, Empty

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Types
# ============================================================================

@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# CloudWatch Metrics Collector
# ============================================================================

class CloudWatchMetrics:
    """
    Publish metrics to AWS CloudWatch.
    
    Automatically batches metrics and publishes in background thread
    to avoid blocking LLM calls.
    
    Example:
        metrics = CloudWatchMetrics(
            namespace="AgenticAIGateway",
            region="us-east-1"
        )
        
        # Use as metrics collector for gateway
        gateway = AIGateway(config, metrics=metrics)
        
        # Metrics automatically published to CloudWatch
        response = gateway.invoke("Hello")
        
        # Manual metric publishing
        metrics.put_metric(
            name="CustomMetric",
            value=42,
            unit="Count",
            dimensions={"Environment": "Production"}
        )
    """
    
    def __init__(
        self,
        namespace: str = "AgenticAIGateway",
        region: str = "us-east-1",
        publish_interval_seconds: float = 60.0,
        batch_size: int = 20,
        enabled: bool = True,
        default_dimensions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize CloudWatch metrics.
        
        Args:
            namespace: CloudWatch namespace for metrics
            region: AWS region
            publish_interval_seconds: How often to publish batched metrics
            batch_size: Max metrics per PutMetricData call
            enabled: Whether to actually publish (False for testing)
            default_dimensions: Dimensions added to all metrics
        """
        self.namespace = namespace
        self.region = region
        self.publish_interval = publish_interval_seconds
        self.batch_size = batch_size
        self.enabled = enabled
        self.default_dimensions = default_dimensions or {}
        
        self._client = None
        self._queue: Queue = Queue()
        self._lock = Lock()
        self._running = False
        self._thread: Optional[Thread] = None
        
        # Local aggregation for high-volume metrics
        self._aggregated: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics
        self._metrics_published = 0
        self._publish_errors = 0
        
        # Start background publisher
        if enabled:
            self._start_publisher()
    
    @property
    def client(self):
        """Lazy-load CloudWatch client."""
        if self._client is None:
            import boto3
            self._client = boto3.client("cloudwatch", region_name=self.region)
        return self._client
    
    def _start_publisher(self) -> None:
        """Start background publishing thread."""
        self._running = True
        self._thread = Thread(target=self._publisher_loop, daemon=True)
        self._thread.start()
        logger.info(f"[CloudWatch] Started publisher (interval: {self.publish_interval}s)")
    
    def _publisher_loop(self) -> None:
        """Background loop to publish metrics."""
        while self._running:
            try:
                self._publish_batch()
                time.sleep(self.publish_interval)
            except Exception as e:
                logger.error(f"[CloudWatch] Publisher error: {e}")
                self._publish_errors += 1
    
    def _publish_batch(self) -> None:
        """Publish queued metrics to CloudWatch."""
        metrics = []
        
        # Drain queue
        while not self._queue.empty() and len(metrics) < self.batch_size:
            try:
                metric = self._queue.get_nowait()
                metrics.append(metric)
            except Empty:
                break
        
        if not metrics:
            return
        
        # Convert to CloudWatch format
        metric_data = []
        for m in metrics:
            dimensions = [
                {"Name": k, "Value": v}
                for k, v in {**self.default_dimensions, **m.dimensions}.items()
            ]
            
            metric_data.append({
                "MetricName": m.name,
                "Value": m.value,
                "Unit": m.unit,
                "Timestamp": m.timestamp,
                "Dimensions": dimensions
            })
        
        # Publish
        try:
            self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data
            )
            self._metrics_published += len(metric_data)
            logger.info(f"[CloudWatch] Published {len(metric_data)} metrics")
        except Exception as e:
            logger.error(f"[CloudWatch] Failed to publish: {e}")
            self._publish_errors += 1
    
    def put_metric(
        self,
        name: str,
        value: float,
        unit: str = "Count",
        dimensions: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Queue a metric for publishing.
        
        Args:
            name: Metric name
            value: Metric value
            unit: CloudWatch unit (Count, Milliseconds, Bytes, etc.)
            dimensions: Additional dimensions
        """
        if not self.enabled:
            return
        
        metric = MetricPoint(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            dimensions=dimensions or {}
        )
        
        self._queue.put(metric)
    
    def record(
        self,
        model_id: str,
        latency_ms: int,
        success: bool,
        is_canary: bool,
        is_fallback: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Record gateway metrics (implements MetricsCollector protocol).
        
        This method is called automatically by AIGateway after each invocation.
        """
        # Common dimensions
        dims = {
            "ModelId": model_id,
            "IsCanary": str(is_canary),
            "IsFallback": str(is_fallback),
        }
        
        # Latency
        self.put_metric("Latency", latency_ms, "Milliseconds", dims)
        
        # Invocation count
        self.put_metric("Invocations", 1, "Count", dims)
        
        # Success/Error
        if success:
            self.put_metric("SuccessfulInvocations", 1, "Count", dims)
        else:
            self.put_metric("Errors", 1, "Count", dims)
            self.put_metric(
                "ErrorsByType",
                1,
                "Count",
                {**dims, "ErrorType": error[:50] if error else "Unknown"}
            )
        
        # Fallback metrics
        if is_fallback:
            self.put_metric("FallbackInvocations", 1, "Count", dims)
        
        # Canary metrics
        if is_canary:
            self.put_metric("CanaryInvocations", 1, "Count", dims)
    
    def record_tokens(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """Record token usage metrics."""
        dims = {"ModelId": model_id}
        
        self.put_metric("InputTokens", input_tokens, "Count", dims)
        self.put_metric("OutputTokens", output_tokens, "Count", dims)
        self.put_metric("TotalTokens", input_tokens + output_tokens, "Count", dims)
    
    def record_cost(
        self,
        model_id: str,
        cost_usd: float
    ) -> None:
        """Record cost metrics."""
        dims = {"ModelId": model_id}
        self.put_metric("Cost", cost_usd, "None", dims)  # No currency unit in CloudWatch
    
    def record_cache(
        self,
        hit: bool,
        tokens_saved: int = 0
    ) -> None:
        """Record cache metrics."""
        if hit:
            self.put_metric("CacheHits", 1, "Count")
            self.put_metric("TokensSavedByCache", tokens_saved, "Count")
        else:
            self.put_metric("CacheMisses", 1, "Count")
    
    def flush(self) -> None:
        """Force publish all queued metrics."""
        while not self._queue.empty():
            self._publish_batch()
    
    def stop(self) -> None:
        """Stop the background publisher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get publishing statistics."""
        return {
            "metrics_published": self._metrics_published,
            "publish_errors": self._publish_errors,
            "queue_size": self._queue.qsize(),
            "enabled": self.enabled,
            "namespace": self.namespace,
        }


# ============================================================================
# Dashboard Template
# ============================================================================

def generate_dashboard_json(namespace: str = "AgenticAIGateway") -> Dict[str, Any]:
    """
    Generate a CloudWatch dashboard template.
    
    Returns JSON that can be used with `aws cloudwatch put-dashboard`.
    
    Example:
        import json
        dashboard = generate_dashboard_json()
        print(json.dumps(dashboard, indent=2))
    """
    return {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Invocations by Model",
                    "metrics": [
                        [namespace, "Invocations", {"stat": "Sum"}]
                    ],
                    "period": 60,
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Latency (p50, p90, p99)",
                    "metrics": [
                        [namespace, "Latency", {"stat": "p50"}],
                        [namespace, "Latency", {"stat": "p90"}],
                        [namespace, "Latency", {"stat": "p99"}]
                    ],
                    "period": 60,
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 8,
                "height": 6,
                "properties": {
                    "title": "Error Rate",
                    "metrics": [
                        [namespace, "Errors", {"stat": "Sum"}],
                        [namespace, "SuccessfulInvocations", {"stat": "Sum"}]
                    ],
                    "period": 60,
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 8,
                "y": 6,
                "width": 8,
                "height": 6,
                "properties": {
                    "title": "Fallback Rate",
                    "metrics": [
                        [namespace, "FallbackInvocations", {"stat": "Sum"}],
                        [namespace, "Invocations", {"stat": "Sum"}]
                    ],
                    "period": 60,
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 16,
                "y": 6,
                "width": 8,
                "height": 6,
                "properties": {
                    "title": "Token Usage",
                    "metrics": [
                        [namespace, "InputTokens", {"stat": "Sum"}],
                        [namespace, "OutputTokens", {"stat": "Sum"}]
                    ],
                    "period": 60,
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 12,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Cache Performance",
                    "metrics": [
                        [namespace, "CacheHits", {"stat": "Sum"}],
                        [namespace, "CacheMisses", {"stat": "Sum"}]
                    ],
                    "period": 60,
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 12,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Cost",
                    "metrics": [
                        [namespace, "Cost", {"stat": "Sum"}]
                    ],
                    "period": 3600,
                    "view": "timeSeries"
                }
            }
        ]
    }
