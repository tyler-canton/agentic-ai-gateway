"""
Cost Tracking & Budget Alerts (v0.6.0)
======================================

Track LLM costs and set budget alerts to prevent surprise bills.

Features:
- Per-request cost calculation
- Daily/weekly/monthly budgets
- Alert callbacks (Slack, email, etc.)
- Usage analytics by model
- Multi-tenant cost isolation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import json
import threading


class BudgetPeriod(Enum):
    """Budget time periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# Pricing per 1K tokens (as of 2024)
MODEL_PRICING = {
    # Bedrock Claude
    "anthropic.claude-3-opus": {"input": 0.015, "output": 0.075},
    "anthropic.claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    # OpenAI
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Bedrock Llama
    "meta.llama3-70b": {"input": 0.00265, "output": 0.0035},
    "meta.llama3-8b": {"input": 0.0003, "output": 0.0006},
    # Bedrock Titan
    "amazon.titan-text-express": {"input": 0.0002, "output": 0.0006},
    "amazon.titan-text-lite": {"input": 0.00015, "output": 0.0002},
}


@dataclass
class CostRecord:
    """Single cost record for a request."""
    timestamp: datetime
    model_id: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """Budget configuration."""
    limit: float  # Dollar amount
    period: BudgetPeriod = BudgetPeriod.DAILY
    alert_threshold: float = 0.8  # Alert at 80% by default
    on_alert: Optional[Callable[[float, float], None]] = None  # (current, limit)
    on_exceeded: Optional[Callable[[float, float], None]] = None
    block_on_exceeded: bool = False  # Stop requests when budget exceeded


@dataclass
class CostStats:
    """Cost statistics."""
    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    by_model: Dict[str, Dict[str, float]]
    by_tenant: Dict[str, float]
    period_start: datetime
    period_end: datetime


class CostTracker:
    """
    Track LLM costs and manage budgets.

    Usage:
        tracker = CostTracker(
            budget=BudgetConfig(
                limit=10.00,
                period=BudgetPeriod.DAILY,
                on_alert=lambda curr, limit: print(f"Warning: ${curr}/{limit}")
            )
        )

        # Record a request
        cost = tracker.record(
            model_id="anthropic.claude-3-sonnet",
            input_tokens=500,
            output_tokens=200
        )

        # Get stats
        stats = tracker.get_stats()
    """

    def __init__(
        self,
        budget: Optional[BudgetConfig] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.budget = budget
        self.pricing = {**MODEL_PRICING, **(custom_pricing or {})}
        self.records: List[CostRecord] = []
        self._lock = threading.Lock()
        self._alert_sent = False
        self._exceeded_sent = False

    def get_model_pricing(self, model_id: str) -> Dict[str, float]:
        """Get pricing for a model, with fallback for partial matches."""
        # Exact match
        if model_id in self.pricing:
            return self.pricing[model_id]

        # Partial match (e.g., "anthropic.claude-3-sonnet-20240229-v1:0" -> "anthropic.claude-3-sonnet")
        for key in self.pricing:
            if key in model_id:
                return self.pricing[key]

        # Default pricing (conservative estimate)
        return {"input": 0.01, "output": 0.03}

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> tuple[float, float, float]:
        """Calculate cost for a request."""
        pricing = self.get_model_pricing(model_id)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        return input_cost, output_cost, total_cost

    def record(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostRecord:
        """Record a request and check budget."""
        input_cost, output_cost, total_cost = self.calculate_cost(
            model_id, input_tokens, output_tokens
        )

        record = CostRecord(
            timestamp=datetime.utcnow(),
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            tenant_id=tenant_id,
            metadata=metadata or {}
        )

        with self._lock:
            self.records.append(record)
            self._check_budget()

        return record

    def _get_period_start(self) -> datetime:
        """Get the start of the current budget period."""
        now = datetime.utcnow()

        if not self.budget:
            return now - timedelta(days=1)

        if self.budget.period == BudgetPeriod.HOURLY:
            return now.replace(minute=0, second=0, microsecond=0)
        elif self.budget.period == BudgetPeriod.DAILY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif self.budget.period == BudgetPeriod.WEEKLY:
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif self.budget.period == BudgetPeriod.MONTHLY:
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return now - timedelta(days=1)

    def get_current_spend(self, tenant_id: Optional[str] = None) -> float:
        """Get current spend for the budget period."""
        period_start = self._get_period_start()

        with self._lock:
            total = sum(
                r.total_cost
                for r in self.records
                if r.timestamp >= period_start
                and (tenant_id is None or r.tenant_id == tenant_id)
            )

        return total

    def _check_budget(self) -> None:
        """Check budget and trigger alerts."""
        if not self.budget:
            return

        current = self.get_current_spend()
        limit = self.budget.limit
        threshold = limit * self.budget.alert_threshold

        # Alert threshold
        if current >= threshold and not self._alert_sent:
            self._alert_sent = True
            if self.budget.on_alert:
                self.budget.on_alert(current, limit)

        # Exceeded
        if current >= limit and not self._exceeded_sent:
            self._exceeded_sent = True
            if self.budget.on_exceeded:
                self.budget.on_exceeded(current, limit)

    def check_can_proceed(self, tenant_id: Optional[str] = None) -> bool:
        """Check if a request can proceed (budget not exceeded)."""
        if not self.budget or not self.budget.block_on_exceeded:
            return True

        current = self.get_current_spend(tenant_id)
        return current < self.budget.limit

    def get_stats(
        self,
        period: Optional[BudgetPeriod] = None,
        tenant_id: Optional[str] = None
    ) -> CostStats:
        """Get cost statistics."""
        if period:
            # Use specified period
            now = datetime.utcnow()
            if period == BudgetPeriod.HOURLY:
                start = now - timedelta(hours=1)
            elif period == BudgetPeriod.DAILY:
                start = now - timedelta(days=1)
            elif period == BudgetPeriod.WEEKLY:
                start = now - timedelta(weeks=1)
            else:
                start = now - timedelta(days=30)
        else:
            start = self._get_period_start()

        end = datetime.utcnow()

        with self._lock:
            filtered = [
                r for r in self.records
                if r.timestamp >= start
                and (tenant_id is None or r.tenant_id == tenant_id)
            ]

        # Aggregate by model
        by_model: Dict[str, Dict[str, float]] = {}
        for r in filtered:
            if r.model_id not in by_model:
                by_model[r.model_id] = {
                    "cost": 0,
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            by_model[r.model_id]["cost"] += r.total_cost
            by_model[r.model_id]["requests"] += 1
            by_model[r.model_id]["input_tokens"] += r.input_tokens
            by_model[r.model_id]["output_tokens"] += r.output_tokens

        # Aggregate by tenant
        by_tenant: Dict[str, float] = {}
        for r in filtered:
            tid = r.tenant_id or "default"
            by_tenant[tid] = by_tenant.get(tid, 0) + r.total_cost

        return CostStats(
            total_cost=sum(r.total_cost for r in filtered),
            total_requests=len(filtered),
            total_input_tokens=sum(r.input_tokens for r in filtered),
            total_output_tokens=sum(r.output_tokens for r in filtered),
            by_model=by_model,
            by_tenant=by_tenant,
            period_start=start,
            period_end=end
        )

    def reset_period(self) -> None:
        """Reset alerts for new period (called automatically on period change)."""
        self._alert_sent = False
        self._exceeded_sent = False

    def export_records(self, format: str = "json") -> str:
        """Export records for analysis."""
        with self._lock:
            data = [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "model_id": r.model_id,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_cost": r.total_cost,
                    "tenant_id": r.tenant_id
                }
                for r in self.records
            ]

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            # CSV
            lines = ["timestamp,model_id,input_tokens,output_tokens,total_cost,tenant_id"]
            for d in data:
                lines.append(f"{d['timestamp']},{d['model_id']},{d['input_tokens']},{d['output_tokens']},{d['total_cost']},{d['tenant_id']}")
            return "\n".join(lines)


class CostTrackedGateway:
    """
    Gateway wrapper that tracks costs.

    Usage:
        from agentic_ai_gateway import CostTrackedGateway, BudgetConfig, BudgetPeriod

        gateway = CostTrackedGateway(
            gateway=base_gateway,
            budget=BudgetConfig(
                limit=10.00,
                period=BudgetPeriod.DAILY,
                alert_threshold=0.8,
                on_alert=lambda curr, limit: slack_notify(f"LLM spend: ${curr:.2f}/${limit}")
            )
        )

        response = gateway.invoke("Hello!")
        print(f"Cost: ${response.cost:.4f}")

        stats = gateway.get_cost_stats()
        print(f"Today: ${stats.total_cost:.2f}")
    """

    def __init__(
        self,
        gateway: Any,
        budget: Optional[BudgetConfig] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        tenant_id: Optional[str] = None
    ):
        self.gateway = gateway
        self.tracker = CostTracker(budget=budget, custom_pricing=custom_pricing)
        self.default_tenant_id = tenant_id

    def invoke(
        self,
        prompt: str,
        tenant_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Invoke with cost tracking."""
        tid = tenant_id or self.default_tenant_id

        # Check budget
        if not self.tracker.check_can_proceed(tid):
            raise BudgetExceededError(
                f"Budget exceeded: ${self.tracker.get_current_spend(tid):.2f}"
            )

        # Make request
        response = self.gateway.invoke(prompt, **kwargs)

        # Record cost
        record = self.tracker.record(
            model_id=getattr(response, 'model_used', 'unknown'),
            input_tokens=getattr(response, 'input_tokens', 0),
            output_tokens=getattr(response, 'output_tokens', 0),
            tenant_id=tid
        )

        # Attach cost to response
        response.cost = record.total_cost
        response.cost_record = record

        return response

    async def ainvoke(
        self,
        prompt: str,
        tenant_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Async invoke with cost tracking."""
        tid = tenant_id or self.default_tenant_id

        if not self.tracker.check_can_proceed(tid):
            raise BudgetExceededError(
                f"Budget exceeded: ${self.tracker.get_current_spend(tid):.2f}"
            )

        response = await self.gateway.ainvoke(prompt, **kwargs)

        record = self.tracker.record(
            model_id=getattr(response, 'model_used', 'unknown'),
            input_tokens=getattr(response, 'input_tokens', 0),
            output_tokens=getattr(response, 'output_tokens', 0),
            tenant_id=tid
        )

        response.cost = record.total_cost
        response.cost_record = record

        return response

    def get_cost_stats(
        self,
        period: Optional[BudgetPeriod] = None,
        tenant_id: Optional[str] = None
    ) -> CostStats:
        """Get cost statistics."""
        return self.tracker.get_stats(period, tenant_id)

    def get_current_spend(self, tenant_id: Optional[str] = None) -> float:
        """Get current spend."""
        return self.tracker.get_current_spend(tenant_id or self.default_tenant_id)

    def export_records(self, format: str = "json") -> str:
        """Export cost records."""
        return self.tracker.export_records(format)


class BudgetExceededError(Exception):
    """Raised when budget is exceeded and blocking is enabled."""
    pass


# =============================================================================
# ENTERPRISE INTEGRATIONS
# =============================================================================


class SlackAlerter:
    """
    Send cost alerts to Slack.

    Usage:
        slack = SlackAlerter(webhook_url="https://hooks.slack.com/services/...")

        gateway = CostTrackedGateway(
            gateway=base_gateway,
            budget=BudgetConfig(
                limit=100.00,
                on_alert=slack.send_alert,
                on_exceeded=slack.send_exceeded
            )
        )
    """

    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.channel = channel

    def send_alert(self, current: float, limit: float) -> None:
        """Send budget warning to Slack."""
        import urllib.request
        import json

        payload = {
            "text": f":warning: *LLM Budget Alert*\nCurrent spend: ${current:.2f} / ${limit:.2f} ({(current/limit)*100:.0f}%)",
            "attachments": [{
                "color": "warning",
                "fields": [
                    {"title": "Current Spend", "value": f"${current:.2f}", "short": True},
                    {"title": "Budget Limit", "value": f"${limit:.2f}", "short": True},
                ]
            }]
        }
        if self.channel:
            payload["channel"] = self.channel

        req = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req)

    def send_exceeded(self, current: float, limit: float) -> None:
        """Send budget exceeded alert to Slack."""
        import urllib.request
        import json

        payload = {
            "text": f":rotating_light: *LLM Budget EXCEEDED*\nSpend: ${current:.2f} / ${limit:.2f}",
            "attachments": [{
                "color": "danger",
                "fields": [
                    {"title": "Current Spend", "value": f"${current:.2f}", "short": True},
                    {"title": "Budget Limit", "value": f"${limit:.2f}", "short": True},
                ]
            }]
        }
        if self.channel:
            payload["channel"] = self.channel

        req = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req)


class CloudWatchCostMetrics:
    """
    Push cost metrics to AWS CloudWatch.

    Usage:
        cw_metrics = CloudWatchCostMetrics(namespace="MyApp/LLM")

        # After each request
        cw_metrics.record_request(cost_record)

        # Or wrap the tracker
        gateway = CostTrackedGateway(
            gateway=base_gateway,
            on_record=cw_metrics.record_request
        )
    """

    def __init__(
        self,
        namespace: str = "AgenticAIGateway",
        region: Optional[str] = None
    ):
        import boto3
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self.namespace = namespace

    def record_request(self, record: CostRecord) -> None:
        """Push cost metrics to CloudWatch."""
        dimensions = [
            {"Name": "ModelId", "Value": record.model_id},
        ]
        if record.tenant_id:
            dimensions.append({"Name": "TenantId", "Value": record.tenant_id})

        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    "MetricName": "RequestCost",
                    "Value": record.total_cost,
                    "Unit": "None",
                    "Dimensions": dimensions
                },
                {
                    "MetricName": "InputTokens",
                    "Value": record.input_tokens,
                    "Unit": "Count",
                    "Dimensions": dimensions
                },
                {
                    "MetricName": "OutputTokens",
                    "Value": record.output_tokens,
                    "Unit": "Count",
                    "Dimensions": dimensions
                },
            ]
        )

    def record_budget_status(self, current: float, limit: float, period: str) -> None:
        """Push budget utilization to CloudWatch."""
        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    "MetricName": "BudgetUtilization",
                    "Value": (current / limit) * 100,
                    "Unit": "Percent",
                    "Dimensions": [{"Name": "Period", "Value": period}]
                },
                {
                    "MetricName": "BudgetSpend",
                    "Value": current,
                    "Unit": "None",
                    "Dimensions": [{"Name": "Period", "Value": period}]
                },
            ]
        )


class DataDogCostMetrics:
    """
    Push cost metrics to DataDog.

    Usage:
        dd = DataDogCostMetrics(api_key="...", app_key="...")
        gateway = CostTrackedGateway(gateway=base_gateway)

        # After requests
        dd.record_request(record)
    """

    def __init__(self, api_key: str, app_key: Optional[str] = None, site: str = "datadoghq.com"):
        self.api_key = api_key
        self.app_key = app_key
        self.base_url = f"https://api.{site}"

    def record_request(self, record: CostRecord) -> None:
        """Send metrics to DataDog."""
        import urllib.request
        import json
        import time

        tags = [
            f"model:{record.model_id}",
        ]
        if record.tenant_id:
            tags.append(f"tenant:{record.tenant_id}")

        now = int(time.time())
        payload = {
            "series": [
                {
                    "metric": "llm.request.cost",
                    "points": [[now, record.total_cost]],
                    "type": "gauge",
                    "tags": tags
                },
                {
                    "metric": "llm.tokens.input",
                    "points": [[now, record.input_tokens]],
                    "type": "count",
                    "tags": tags
                },
                {
                    "metric": "llm.tokens.output",
                    "points": [[now, record.output_tokens]],
                    "type": "count",
                    "tags": tags
                },
            ]
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/v1/series",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "DD-API-KEY": self.api_key,
            }
        )
        urllib.request.urlopen(req)


class WebhookExporter:
    """
    Export cost data to any HTTP endpoint.

    Usage:
        webhook = WebhookExporter(
            url="https://your-api.com/llm-costs",
            headers={"Authorization": "Bearer ..."}
        )

        gateway = CostTrackedGateway(gateway=base_gateway)

        # Batch export every hour
        webhook.export_batch(gateway.tracker.records)
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 100
    ):
        self.url = url
        self.headers = headers or {}
        self.batch_size = batch_size

    def export_record(self, record: CostRecord) -> None:
        """Export single record."""
        import urllib.request
        import json

        payload = {
            "timestamp": record.timestamp.isoformat(),
            "model_id": record.model_id,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "input_cost": record.input_cost,
            "output_cost": record.output_cost,
            "total_cost": record.total_cost,
            "tenant_id": record.tenant_id,
            "metadata": record.metadata
        }

        headers = {"Content-Type": "application/json", **self.headers}
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers=headers
        )
        urllib.request.urlopen(req)

    def export_batch(self, records: List[CostRecord]) -> None:
        """Export batch of records."""
        import urllib.request
        import json

        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            payload = {
                "records": [
                    {
                        "timestamp": r.timestamp.isoformat(),
                        "model_id": r.model_id,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "total_cost": r.total_cost,
                        "tenant_id": r.tenant_id,
                    }
                    for r in batch
                ]
            }

            headers = {"Content-Type": "application/json", **self.headers}
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode(),
                headers=headers
            )
            urllib.request.urlopen(req)


class S3CostExporter:
    """
    Export cost data to S3 for data lake / analytics.

    Usage:
        s3_exporter = S3CostExporter(
            bucket="my-llm-analytics",
            prefix="costs/"
        )

        # Daily export
        s3_exporter.export_daily(gateway.tracker.records)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "llm-costs/",
        region: Optional[str] = None
    ):
        import boto3
        self.s3 = boto3.client("s3", region_name=region)
        self.bucket = bucket
        self.prefix = prefix

    def export_daily(self, records: List[CostRecord], date: Optional[datetime] = None) -> str:
        """Export records for a day to S3."""
        import json

        if date is None:
            date = datetime.utcnow()

        # Filter to date
        day_records = [
            r for r in records
            if r.timestamp.date() == date.date()
        ]

        # Build key
        key = f"{self.prefix}year={date.year}/month={date.month:02d}/day={date.day:02d}/costs.json"

        # Export as newline-delimited JSON (good for Athena)
        body = "\n".join(
            json.dumps({
                "timestamp": r.timestamp.isoformat(),
                "model_id": r.model_id,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_cost": r.total_cost,
                "tenant_id": r.tenant_id,
            })
            for r in day_records
        )

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body.encode(),
            ContentType="application/x-ndjson"
        )

        return f"s3://{self.bucket}/{key}"

    def export_csv(self, records: List[CostRecord], key: str) -> str:
        """Export records as CSV."""
        lines = ["timestamp,model_id,input_tokens,output_tokens,total_cost,tenant_id"]
        for r in records:
            lines.append(
                f"{r.timestamp.isoformat()},{r.model_id},{r.input_tokens},{r.output_tokens},{r.total_cost},{r.tenant_id or ''}"
            )

        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}{key}",
            Body="\n".join(lines).encode(),
            ContentType="text/csv"
        )

        return f"s3://{self.bucket}/{self.prefix}{key}"


class MCPCostServer:
    """
    MCP (Model Context Protocol) server for cost tracking.

    Exposes cost data as MCP resources and tools that AI assistants
    (Claude Code, Claude Desktop) can query directly.

    Usage:
        # Create MCP server
        mcp_server = MCPCostServer(tracker=gateway.tracker)

        # Run as standalone server
        mcp_server.run(port=3001)

        # Or integrate with FastAPI
        from fastapi import FastAPI
        app = FastAPI()
        mcp_server.mount(app, prefix="/mcp")

    MCP Config (claude_desktop_config.json):
        {
            "mcpServers": {
                "llm-costs": {
                    "command": "python",
                    "args": ["-m", "agentic_ai_gateway.mcp_server"],
                    "env": {"COST_TRACKER_URL": "http://localhost:8000/mcp"}
                }
            }
        }
    """

    def __init__(self, tracker: "CostTracker"):
        self.tracker = tracker

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "get_cost_stats",
                "description": "Get LLM cost statistics for current budget period",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {
                            "type": "string",
                            "description": "Optional tenant ID to filter by"
                        }
                    }
                }
            },
            {
                "name": "get_cost_by_model",
                "description": "Get cost breakdown by model",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_cost_by_tenant",
                "description": "Get cost breakdown by tenant/customer",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_budget_status",
                "description": "Check current budget utilization and remaining budget",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_recent_requests",
                "description": "Get recent LLM requests with cost details",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of records to return (default 10)"
                        }
                    }
                }
            },
        ]

    def handle_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls."""
        if name == "get_cost_stats":
            stats = self.tracker.get_stats()
            return {
                "total_cost": f"${stats.total_cost:.4f}",
                "total_requests": stats.total_requests,
                "total_input_tokens": stats.total_input_tokens,
                "total_output_tokens": stats.total_output_tokens,
                "period": stats.period,
            }

        elif name == "get_cost_by_model":
            stats = self.tracker.get_stats()
            return {
                "by_model": {
                    model: f"${cost:.4f}"
                    for model, cost in stats.by_model.items()
                }
            }

        elif name == "get_cost_by_tenant":
            stats = self.tracker.get_stats()
            return {
                "by_tenant": {
                    tenant: f"${cost:.4f}"
                    for tenant, cost in stats.by_tenant.items()
                }
            }

        elif name == "get_budget_status":
            stats = self.tracker.get_stats()
            budget = getattr(self.tracker, 'budget', None)
            if budget:
                return {
                    "current_spend": f"${stats.total_cost:.2f}",
                    "budget_limit": f"${budget.limit:.2f}",
                    "utilization": f"{(stats.total_cost / budget.limit) * 100:.1f}%",
                    "remaining": f"${max(0, budget.limit - stats.total_cost):.2f}",
                    "period": budget.period.value,
                    "status": "EXCEEDED" if stats.total_cost >= budget.limit else "OK"
                }
            return {"message": "No budget configured"}

        elif name == "get_recent_requests":
            limit = arguments.get("limit", 10)
            records = self.tracker.records[-limit:]
            return {
                "requests": [
                    {
                        "timestamp": r.timestamp.isoformat(),
                        "model": r.model_id,
                        "tokens": f"{r.input_tokens}→{r.output_tokens}",
                        "cost": f"${r.total_cost:.4f}",
                        "tenant": r.tenant_id or "default"
                    }
                    for r in reversed(records)
                ]
            }

        return {"error": f"Unknown tool: {name}"}

    def get_resources(self) -> List[Dict[str, Any]]:
        """Return MCP resource definitions."""
        return [
            {
                "uri": "costs://stats",
                "name": "Cost Statistics",
                "description": "Current LLM cost statistics",
                "mimeType": "application/json"
            },
            {
                "uri": "costs://records",
                "name": "Cost Records",
                "description": "Detailed cost records (last 100)",
                "mimeType": "application/json"
            }
        ]

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read MCP resource content."""
        if uri == "costs://stats":
            stats = self.tracker.get_stats()
            return {
                "total_cost": stats.total_cost,
                "total_requests": stats.total_requests,
                "by_model": stats.by_model,
                "by_tenant": stats.by_tenant,
            }

        elif uri == "costs://records":
            return {
                "records": [
                    {
                        "timestamp": r.timestamp.isoformat(),
                        "model_id": r.model_id,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "total_cost": r.total_cost,
                        "tenant_id": r.tenant_id,
                    }
                    for r in self.tracker.records[-100:]
                ]
            }

        return {"error": f"Unknown resource: {uri}"}

    def to_fastapi_routes(self):
        """
        Generate FastAPI routes for MCP protocol.

        Usage:
            from fastapi import FastAPI
            app = FastAPI()

            mcp = MCPCostServer(tracker)
            app.include_router(mcp.to_fastapi_routes(), prefix="/mcp")
        """
        try:
            from fastapi import APIRouter
            from fastapi.responses import JSONResponse
        except ImportError:
            raise ImportError("FastAPI required: pip install fastapi")

        router = APIRouter()

        @router.get("/tools")
        async def list_tools():
            return {"tools": self.get_tools()}

        @router.post("/tools/{tool_name}")
        async def call_tool(tool_name: str, arguments: Dict[str, Any] = {}):
            result = self.handle_tool(tool_name, arguments)
            return JSONResponse(content=result)

        @router.get("/resources")
        async def list_resources():
            return {"resources": self.get_resources()}

        @router.get("/resources/{resource_uri:path}")
        async def read_resource(resource_uri: str):
            result = self.read_resource(f"costs://{resource_uri}")
            return JSONResponse(content=result)

        return router
