# Agentic AI Gateway

Production-grade LLM routing with automatic fallbacks, canary deployments, and multi-provider support.

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

# For both (cross-provider fallback)
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

## Streaming Support (v0.2.0+)

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
- Zero infrastructure (it's just Python code)

## License

MIT
