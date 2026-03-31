# LinkedIn Article: Agentic AI Gateway v0.4.0

---

## Announcing Agentic AI Gateway v0.4.0: Enterprise-Grade LLM Operations

**TL;DR:** I just released v0.4.0 of my open-source LLM gateway. This release adds the features production teams actually need: response caching, cost tracking with budget alerts, semantic routing, circuit breakers, and CloudWatch integration.

---

If you're running LLMs in production, you've probably dealt with:

- Surprise bills when traffic spikes
- Repeated calls burning tokens on identical queries
- One-size-fits-all model selection wasting money
- No visibility into what's happening across providers

**v0.4.0 solves all of these.**

---

### What's New in v0.4.0

**1. Response Caching**

Stop paying for the same answer twice.

```python
from agentic_ai_gateway import create_bedrock_gateway
from agentic_ai_gateway.caching import CachedGateway

gateway = create_bedrock_gateway(...)
cached = CachedGateway(gateway, ttl_seconds=3600, max_size=1000)

# First call hits the model
response1 = cached.invoke("What is Python?")

# Second call returns instantly from cache
response2 = cached.invoke("What is Python?")
print(response2.metadata["cache_hit"])  # True
```

LRU eviction, configurable TTL, and token savings tracking built in.

**2. Cost Tracking with Budget Alerts**

Know exactly what you're spending — and stop before it's too late.

```python
from agentic_ai_gateway.costs import CostTracker

tracker = CostTracker(budget_limit=100.00)  # $100 limit
tracker.on_alert(lambda a: slack.send(f"Budget alert: {a.message}"))

# Tracks every invocation
tracker.record(
    model_id="anthropic.claude-4-sonnet",
    input_tokens=1000,
    output_tokens=500
)

print(f"Total: ${tracker.total_cost:.4f}")
print(f"Remaining: ${tracker.budget_remaining:.2f}")
```

Accurate per-model pricing for Bedrock and OpenAI. Alerts at 50%, 75%, 90%, 100%.

**3. Semantic Routing**

Route prompts to the right model automatically.

```python
from agentic_ai_gateway.routing import RoutedGateway, PromptIntent

routed = RoutedGateway(
    gateway,
    model_mapping={
        PromptIntent.CODE: "anthropic.claude-4-sonnet",
        PromptIntent.CHAT: "anthropic.claude-4-haiku",      # Cheap for simple Q&A
        PromptIntent.ANALYSIS: "anthropic.claude-4-opus",   # Power for complex tasks
    }
)

# Automatically routes to Sonnet
response = routed.invoke("Write a Python function to sort a list")

# Automatically routes to Haiku
response = routed.invoke("What's the capital of France?")
```

Intent classification, complexity estimation, and custom routing rules.

**4. Circuit Breaker + Retry**

Production-grade resilience patterns.

```python
from agentic_ai_gateway.resilience import RetryWithBackoff, CircuitBreaker

# Exponential backoff with jitter
retry = RetryWithBackoff(max_retries=3, base_delay_seconds=1.0)
result = retry.execute(lambda: gateway.invoke(prompt))

# Circuit breaker for failing endpoints
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=30)
if breaker.is_open:
    use_fallback()
```

**5. CloudWatch Integration**

One-line observability.

```python
from agentic_ai_gateway.observability import CloudWatchMetrics

metrics = CloudWatchMetrics(namespace="MyApp/LLM", region="us-east-1")
gateway = AIGateway(config, metrics=metrics)

# Automatic metrics: Latency, Invocations, Errors, Tokens, Costs
# Dashboard template included
```

---

### Why This Matters

Last month I talked to a team that got a $40K surprise bill because one endpoint started looping. Another team was sending simple FAQ questions to GPT-4 when GPT-3.5 would've been fine.

v0.4.0 is about building the operational layer that production LLM systems need:

- **Cache** repeated queries
- **Track** every dollar spent
- **Route** to the right model for the job
- **Recover** gracefully from failures
- **Monitor** everything in one place

---

### Quick Start

```bash
pip install agentic-ai-gateway
```

```python
from agentic_ai_gateway import AgenticGateway, AgenticGatewayConfig
from agentic_ai_gateway.caching import CachedGateway
from agentic_ai_gateway.costs import CostTracker
from agentic_ai_gateway.routing import RoutedGateway, PromptIntent
from agentic_ai_gateway.observability import CloudWatchMetrics

# Full production setup
config = AgenticGatewayConfig(
    primary_model="anthropic.claude-4-sonnet",
    fallback_models=["anthropic.claude-4-haiku"],
)

gateway = AgenticGateway(config)
gateway = CachedGateway(gateway, ttl_seconds=3600)
gateway = RoutedGateway(gateway, model_mapping={
    PromptIntent.CHAT: "anthropic.claude-4-haiku",
    PromptIntent.CODE: "anthropic.claude-4-sonnet",
})

response = gateway.invoke("Hello, world!")
```

---

### What's Next (v0.5.0)

- **LLM-based classification** — Use a cheap model to classify intent before routing
- **Persistent caching** — Redis/DynamoDB backends
- **Multi-region routing** — Automatic failover across AWS regions

---

### Try It

PyPI: `pip install agentic-ai-gateway`
GitHub: github.com/tyler-canton/agentic-ai-gateway
Docs: Full examples in the README

What operational challenges are you hitting with LLMs in production? Drop a comment — I'm building based on real pain points.

---

#AI #LLM #MachineLearning #Python #AWS #Bedrock #OpenSource #MLOps #LLMOps #CloudWatch #DevOps

---

## Post Options

### Short Version (for regular post)

Just shipped Agentic AI Gateway v0.4.0

New features for production LLM systems:

- Response caching (stop paying for repeated queries)
- Cost tracking with budget alerts ($100 limit? Get notified at 50/75/90%)
- Semantic routing (simple questions → cheap model, complex → powerful)
- Circuit breaker + exponential backoff
- CloudWatch metrics out of the box

```python
# Route to the right model automatically
routed = RoutedGateway(gateway, model_mapping={
    PromptIntent.CHAT: "claude-4-haiku",     # $0.25/1M tokens
    PromptIntent.CODE: "claude-4-sonnet",    # $3/1M tokens
    PromptIntent.ANALYSIS: "claude-4-opus",  # $15/1M tokens
})
```

`pip install agentic-ai-gateway`

What operational challenges are you hitting with LLMs? Building based on real pain points.

#AI #LLM #Python #MLOps #OpenSource

---

### Image Suggestions

1. Architecture diagram: Gateway → Cache → Router → Models with CloudWatch metrics flowing out
2. Cost dashboard mockup showing budget alerts and per-model breakdown
3. Code comparison: Manual model selection vs. semantic routing
4. Terminal screenshot showing cache hit rates and cost savings

