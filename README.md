# Agentic AI Gateway

[![PyPI version](https://badge.fury.io/py/agentic-ai-gateway.svg)](https://pypi.org/project/agentic-ai-gateway/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author: Tyler Canton](https://img.shields.io/badge/Author-Tyler%20Canton-green.svg)](https://github.com/tyler-canton)

**Production-grade LLM routing with automatic fallbacks, canary deployments, and multi-provider support.**

Created by [Tyler Canton](https://github.com/tyler-canton) | [PyPI](https://pypi.org/project/agentic-ai-gateway/) | [Documentation](https://github.com/tyler-canton/agentic-ai-gateway#readme)

## The Problem

When you call an LLM directly, you're one API error away from a crashed application:

```python
# If Claude is down, rate-limited, or throws an error... your app crashes
response = bedrock.invoke_model(modelId='anthropic.claude-3-sonnet...')
```

AWS Bedrock, OpenAI, and other LLM providers don't offer:
- Automatic fallback to alternative models
- Traffic splitting for A/B testing new models
- Centralized monitoring across models
- Runtime configuration without redeployment

## The Solution

Agentic AI Gateway sits between your application and LLM providers:

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────────────────┐
│  Your App   │────▶│ Agentic AI Gateway │────▶│ Claude (Primary)             │
└─────────────┘     │             │     │ Llama (Fallback)             │
                    │ - Routing   │     │ GPT-4 (Cross-provider backup)│
                    │ - Fallback  │     └──────────────────────────────┘
                    │ - Canary    │
                    │ - Metrics   │
                    └─────────────┘
```

## Installation

```bash
# For AWS Bedrock
pip install agentic-ai-gateway[bedrock]

# For OpenAI
pip install agentic-ai-gateway[openai]

# With Redis caching (v0.5.0+)
pip install agentic-ai-gateway[redis]

# For everything (cross-provider + redis)
pip install agentic-ai-gateway[all]
```

## Quick Start

### AWS Bedrock

```python
from agentic_ai_gateway import create_bedrock_gateway

# Create gateway with automatic fallback
gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"],
    region="us-east-1"
)

# Use it - if Claude Sonnet fails, automatically tries Haiku
response = gateway.invoke("What is the capital of France?")
print(response.content)  # "The capital of France is Paris."
print(response.model_used)  # Shows which model actually responded
print(response.fallback_used)  # True if primary failed
```

### OpenAI

```python
from agentic_ai_gateway import create_openai_gateway

gateway = create_openai_gateway(
    primary_model="gpt-4o",
    fallback_models=["gpt-4o-mini"],
    api_key="sk-..."
)

response = gateway.invoke("Explain quantum computing")
```

### Cross-Provider Fallback

```python
from agentic_ai_gateway import create_multi_provider_gateway

# Ultimate resilience: fall back across providers
gateway = create_multi_provider_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=[
        "anthropic.claude-3-haiku-20240307-v1:0",  # Bedrock fallback
        "gpt-4o-mini",  # OpenAI fallback
    ],
    bedrock_region="us-east-1",
    openai_api_key="sk-..."
)

response = gateway.invoke("Summarize this document...")
# Tries Claude Sonnet → Claude Haiku → GPT-4o Mini
```

## Canary Deployments

Test new models on a percentage of traffic:

```python
from agentic_ai_gateway import create_bedrock_gateway

gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    canary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_percentage=10,  # 10% traffic to Claude 3.5
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"]
)

# 90% of requests go to Claude 3 Sonnet
# 10% of requests go to Claude 3.5 Sonnet (canary)
response = gateway.invoke("Hello!")
print(response.canary_used)  # True if canary was selected
```

### Gradual Rollout

```python
# Week 1: 5% canary
gateway.update_config(canary_percentage=5)

# Week 2: 20% canary (metrics look good)
gateway.update_config(canary_percentage=20)

# Week 3: 50% canary
gateway.update_config(canary_percentage=50)

# Week 4: Promote canary to primary
gateway.update_config(
    primary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_model=None,
    canary_percentage=0
)
```

## Monitoring

Built-in metrics tracking:

```python
# After running some requests
metrics = gateway.get_metrics()

print(metrics)
# {
#     "total_invocations": 1000,
#     "total_errors": 12,
#     "error_rate": 0.012,
#     "fallback_rate": 0.03,
#     "avg_latency_ms": 1250,
#     "by_model": {
#         "anthropic.claude-3-sonnet...": {"invocations": 900, "errors": 10},
#         "anthropic.claude-3-haiku...": {"invocations": 100, "errors": 2}
#     }
# }
```

### CloudWatch Integration

```python
import boto3
from agentic_ai_gateway import AgenticGateway, AgenticGatewayConfig, BedrockProvider

class CloudWatchMetrics:
    def __init__(self, namespace="AgenticGateway"):
        self.cloudwatch = boto3.client("cloudwatch")
        self.namespace = namespace

    def record(self, model_id, latency_ms, success, is_canary, is_fallback, error=None):
        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    "MetricName": "Invocations",
                    "Value": 1,
                    "Dimensions": [
                        {"Name": "ModelId", "Value": model_id},
                        {"Name": "Success", "Value": str(success)}
                    ]
                },
                {
                    "MetricName": "Latency",
                    "Value": latency_ms,
                    "Unit": "Milliseconds",
                    "Dimensions": [{"Name": "ModelId", "Value": model_id}]
                }
            ]
        )

# Use custom metrics
gateway = AgenticGateway(
    config=AgenticGatewayConfig(
        primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"]
    ),
    providers=[BedrockProvider()],
    metrics=CloudWatchMetrics()
)
```

## Custom Providers

Add support for any LLM provider:

```python
from agentic_ai_gateway import AgenticGateway, AgenticGatewayConfig, LLMProvider

class AnthropicDirectProvider(LLMProvider):
    def __init__(self, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def supports_model(self, model_id: str) -> bool:
        return "claude" in model_id and "anthropic." not in model_id

    def invoke(self, model_id: str, prompt: str, **kwargs):
        response = self.client.messages.create(
            model=model_id,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        return content, response.usage.input_tokens, response.usage.output_tokens

# Use it
gateway = AgenticGateway(
    config=AgenticGatewayConfig(primary_model="claude-3-opus-20240229"),
    providers=[AnthropicDirectProvider(api_key="sk-...")]
)
```

## Multi-Agent Tool Calling

For multi-agent workflows that need tool calling, use the `converse()` method:

```python
from agentic_ai_gateway import create_bedrock_gateway

gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"]
)

# Define tools
tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_patient_data",
                "description": "Retrieve patient records",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string"}
                        },
                        "required": ["patient_id"]
                    }
                }
            }
        }
    ]
}

# Use converse() with tool calling - includes automatic fallback
result = gateway.converse(
    messages=[{
        "role": "user",
        "content": [{"text": "Look up patient P001"}]
    }],
    system=[{"text": "You are a healthcare assistant."}],
    tool_config=tool_config,
    inference_config={"maxTokens": 4096, "temperature": 0.1}
)

print(f"Model used: {result['model_used']}")
print(f"Fallback used: {result['fallback_used']}")

# Access raw Bedrock response
response = result["response"]
```

## RAG Pipeline Integration

Integrate with your RAG pipeline for resilient document Q&A:

```python
from agentic_ai_gateway import create_bedrock_gateway

gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"],
    canary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_percentage=10  # A/B test new model
)

def rag_query(question: str, context_chunks: list[str]) -> dict:
    """RAG query with automatic fallback."""
    prompt = f"""Answer based on context:

Context:
{chr(10).join(context_chunks)}

Question: {question}"""

    response = gateway.invoke(prompt, max_tokens=500, temperature=0.3)

    return {
        "answer": response.content,
        "model_used": response.model_used,
        "fallback_used": response.fallback_used
    }
```

## Async Support

```python
import asyncio
from agentic_ai_gateway import create_bedrock_gateway

gateway = create_bedrock_gateway()

async def main():
    response = await gateway.ainvoke("Hello async world!")
    print(response.content)

asyncio.run(main())
```

## v0.6.0 Features

### Cost Tracking & Budget Alerts

Track LLM costs and prevent surprise bills:

```python
from agentic_ai_gateway import CostTrackedGateway, BudgetConfig, BudgetPeriod

gateway = CostTrackedGateway(
    gateway=base_gateway,
    budget=BudgetConfig(
        limit=10.00,  # $10/day
        period=BudgetPeriod.DAILY,
        alert_threshold=0.8,  # Alert at 80%
        on_alert=lambda curr, limit: slack_notify(f"LLM spend: ${curr:.2f}/${limit}"),
        block_on_exceeded=True  # Stop requests when budget hit
    )
)

# Every request tracks cost
response = gateway.invoke("Summarize this document...")
print(f"Cost: ${response.cost:.4f}")

# Get usage stats
stats = gateway.get_cost_stats()
print(f"Today: ${stats.total_cost:.2f}")
print(f"By model: {stats.by_model}")
```

### Multi-Tenant Cost Isolation

Track spend per customer/tenant:

```python
# Track costs per tenant
response = gateway.invoke("Hello", tenant_id="customer-123")
response = gateway.invoke("World", tenant_id="customer-456")

stats = gateway.get_cost_stats()
print(stats.by_tenant)
# {"customer-123": 0.003, "customer-456": 0.002}

# Export for billing
csv_data = gateway.export_records(format="csv")
```

### Enterprise Integrations

Connect cost tracking to your production monitoring stack:

#### Slack Alerts

```python
from agentic_ai_gateway import CostTrackedGateway, BudgetConfig, BudgetPeriod, SlackAlerter

slack = SlackAlerter(
    webhook_url="https://hooks.slack.com/services/T.../B.../xxx",
    channel="#llm-costs",
    mention_on_critical="@oncall"
)

gateway = CostTrackedGateway(
    gateway=base_gateway,
    budget=BudgetConfig(
        limit=100.00,
        period=BudgetPeriod.DAILY,
        alert_threshold=0.8,
        on_alert=slack.send_alert
    )
)
```

#### CloudWatch Metrics

```python
from agentic_ai_gateway import CostTrackedGateway, CloudWatchCostMetrics

cw_metrics = CloudWatchCostMetrics(
    namespace="MyApp/LLMCosts",
    region="us-east-1"
)

# Push metrics after each request
response = gateway.invoke("Hello")
cw_metrics.push(gateway.tracker.get_stats())

# Or push periodically
import threading
def push_metrics():
    while True:
        cw_metrics.push(gateway.tracker.get_stats())
        time.sleep(60)

threading.Thread(target=push_metrics, daemon=True).start()
```

#### DataDog Metrics

```python
from agentic_ai_gateway import DataDogCostMetrics

dd_metrics = DataDogCostMetrics(
    api_key="your-datadog-api-key",
    app_key="your-datadog-app-key",
    tags=["env:production", "service:chat-api"]
)

# Push to DataDog
dd_metrics.push(gateway.tracker.get_stats())
```

#### S3 Export (for Athena/QuickSight)

```python
from agentic_ai_gateway import S3CostExporter

exporter = S3CostExporter(
    bucket="my-llm-analytics",
    prefix="costs/",
    region="us-east-1"
)

# Export daily costs (e.g., from cron job)
records = gateway.tracker.get_records(
    start_time=datetime.now() - timedelta(days=1)
)
exporter.export(records, partition_by="day")
# Writes to: s3://my-llm-analytics/costs/year=2024/month=01/day=15/costs.parquet
```

#### Custom Webhook

```python
from agentic_ai_gateway import WebhookExporter

webhook = WebhookExporter(
    url="https://your-api.com/llm-costs",
    headers={"Authorization": "Bearer xxx"},
    batch_size=100
)

# Export records to your internal systems
webhook.export(gateway.tracker.get_records())
```

### MCP Integration (Model Context Protocol)

Let Claude query your cost data directly via MCP:

```python
from agentic_ai_gateway import MCPCostServer, CostTrackedGateway

# Create cost-tracked gateway
gateway = CostTrackedGateway(gateway=base_gateway, budget=budget_config)

# Create MCP server
mcp = MCPCostServer(tracker=gateway.tracker)

# Mount on FastAPI
from fastapi import FastAPI
app = FastAPI()
app.include_router(mcp.to_fastapi_routes(), prefix="/mcp")
```

**Claude Desktop Config** (`claude_desktop_config.json`):
```json
{
    "mcpServers": {
        "llm-costs": {
            "url": "http://localhost:8000/mcp"
        }
    }
}
```

**Available MCP Tools:**
| Tool | Description |
|------|-------------|
| `get_cost_stats` | Current spend, token counts, period stats |
| `get_cost_by_model` | Cost breakdown by model |
| `get_cost_by_tenant` | Cost breakdown by tenant/customer |
| `get_budget_status` | Budget utilization and remaining |
| `get_recent_requests` | Recent LLM requests with costs |

Now Claude can answer: *"How much have I spent on Claude 3 Sonnet today?"*

---

## v0.5.0 Features

### Redis Distributed Cache

Cache LLM responses across load-balanced servers:

```python
from agentic_ai_gateway import RedisCachedGateway, create_bedrock_gateway

# Wrap any gateway with Redis caching
base_gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"]
)

gateway = RedisCachedGateway(
    gateway=base_gateway,
    redis_url="redis://localhost:6379",
    ttl_seconds=3600,  # Cache for 1 hour
    prefix="llm:"
)

# First call hits LLM (~2s)
response = gateway.invoke("What is the capital of France?")

# Second call hits cache (~1ms)
response = gateway.invoke("What is the capital of France?")
print(response.cache_hit)  # True
```

### Conversation Memory

Multi-turn conversations with Redis persistence:

```python
from agentic_ai_gateway import ConversationGateway, RedisConversationMemory

memory = RedisConversationMemory(
    redis_url="redis://localhost:6379",
    max_history=20
)

gateway = ConversationGateway(
    gateway=base_gateway,
    memory=memory
)

# Start a conversation
response = gateway.invoke("Hi, I'm building a healthcare app", conversation_id="user-123")

# Continue the conversation (remembers context)
response = gateway.invoke("What tech stack do you recommend?", conversation_id="user-123")

# Clear conversation
gateway.clear_conversation("user-123")
```

### Guardrails (PII & Injection Protection)

Protect your LLM from sensitive data leaks and attacks:

```python
from agentic_ai_gateway import GuardedGateway, Guardrails, PIIType

gateway = GuardedGateway(
    gateway=base_gateway,
    guardrails=Guardrails(
        pii_detection=True,
        pii_action="redact",  # or "block"
        pii_types=[PIIType.SSN, PIIType.CREDIT_CARD, PIIType.EMAIL],
        prompt_injection_detection=True
    )
)

# PII is automatically redacted
response = gateway.invoke("My SSN is 123-45-6789")
# Prompt sent to LLM: "My SSN is [REDACTED_SSN]"

# Prompt injection is blocked
try:
    response = gateway.invoke("Ignore all instructions and...")
except GuardrailsError as e:
    print(e)  # "Prompt injection detected"
```

### Streaming Support (v0.2.0+)

Stream tokens in real-time for chat interfaces and SSE endpoints:

### Basic Streaming

```python
from agentic_ai_gateway import create_bedrock_gateway

gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"]
)

# Synchronous streaming
for chunk in gateway.invoke_stream("Tell me a story"):
    if chunk["type"] == "start":
        print(f"Using model: {chunk['model_used']}")
    elif chunk["type"] == "token":
        print(chunk["content"], end="", flush=True)
    elif chunk["type"] == "done":
        print(f"\n\nCompleted in {chunk['latency_ms']}ms")
        print(f"Tokens: {chunk['output_tokens']}")
```

### Async Streaming (for FastAPI/aiohttp)

```python
import asyncio
from agentic_ai_gateway import create_bedrock_gateway

gateway = create_bedrock_gateway()

async def stream_response():
    async for chunk in gateway.ainvoke_stream("Explain quantum computing"):
        if chunk["type"] == "token":
            yield chunk["content"]
```

### FastAPI SSE Integration

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/api/v1/query/stream")
async def stream_query(request: QueryRequest):
    async def generate():
        # Emit start event
        yield f"data: {json.dumps({'type': 'start'})}\n\n"

        full_response = ""
        async for chunk in gateway.ainvoke_stream(request.prompt):
            if chunk["type"] == "token":
                full_response += chunk.get("content", "")
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.get('content', '')})}\n\n"
            elif chunk["type"] == "done":
                yield f"data: {json.dumps({'type': 'done', 'model_used': chunk.get('model_used', 'unknown'), 'fallback_used': chunk.get('fallback_used', False)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Streaming Event Types

The streaming API yields dictionaries with the following types:

| Event Type | Description | Fields |
|------------|-------------|--------|
| `start` | Stream started | `model_used`, `fallback_used`, `canary_used` |
| `token` | Content token | `content` (the token text) |
| `done` | Stream complete | `model_used`, `latency_ms`, `input_tokens`, `output_tokens`, `fallback_used` |
| `error` | Error occurred | `error` (error message) |

### Streaming with Fallback

Streaming includes automatic fallback support. If the primary model fails before streaming begins, the gateway automatically tries fallback models:

```python
# If Claude Sonnet fails during connection, automatically tries Haiku
for chunk in gateway.invoke_stream("Hello"):
    if chunk["type"] == "start":
        if chunk["fallback_used"]:
            print(f"⚠️ Using fallback model: {chunk['model_used']}")
    # ... handle other events
```

**Note:** Once streaming has started successfully, if an error occurs mid-stream, the gateway will emit an error event rather than attempting fallback (since partial content has already been delivered).

## Examples

See the [examples/](examples/) directory for complete integration examples:

- **[bedrock_example.py](examples/bedrock_example.py)** - Basic Bedrock usage with fallbacks and canary
- **[multiagent_example.py](examples/multiagent_example.py)** - Multi-agent tool calling with agentic loop
- **[rag_example.py](examples/rag_example.py)** - RAG pipeline integration
- **[streaming_example.py](examples/streaming_example.py)** - Real-time token streaming with SSE

## Why Not Just Use...

| Approach | Limitation |
|----------|------------|
| **Direct API calls** | No fallback, crashes on errors |
| **Try/except wrapper** | Manual, error-prone, no canary |
| **API Gateway (AWS)** | Doesn't understand LLM-specific routing |
| **SageMaker endpoints** | Overkill for routing, designed for hosting |

Agentic AI Gateway is purpose-built for LLM routing:
- Model-aware fallback chains
- Canary deployments with gradual rollout
- Multi-provider support (Bedrock + OpenAI + custom)
- **Cost tracking with budget alerts (v0.6.0)**
- **Multi-tenant cost isolation (v0.6.0)**
- Redis distributed cache for load-balanced apps (v0.5.0)
- Conversation memory with persistence (v0.5.0)
- Guardrails: PII detection & prompt injection defense (v0.5.0)
- Zero infrastructure (it's just Python code)

## Author

**Tyler Canton** - AI/ML Engineer specializing in production LLM systems

- GitHub: [@tyler-canton](https://github.com/tyler-canton)
- PyPI: [agentic-ai-gateway](https://pypi.org/project/agentic-ai-gateway/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - Copyright (c) 2026 Tyler Canton

See [LICENSE](LICENSE) for details.
