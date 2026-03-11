# Why Your Production LLM App Needs an Agentic AI Gateway

*And how to build one in 200 lines of Python*

---

## The 3 AM Wake-Up Call

Picture this: Your AI-powered healthcare application is in production. Doctors rely on it to summarize patient records. Then at 3 AM, your phone buzzes. Anthropic's Claude is experiencing elevated error rates. Your app? Completely down.

```python
# Your current code
response = bedrock.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({'messages': [{'role': 'user', 'content': prompt}]})
)
# If this fails, everything fails.
```

This is the reality of production LLM applications in 2024. We've built sophisticated AI systems, but we're still calling APIs with the reliability architecture of a prototype.

---

## What Bedrock (and OpenAI) Don't Give You

I spent months building a healthcare AI system on AWS Bedrock. Great service. But here's what surprised me:

**No automatic fallbacks.** If Claude is down, Bedrock doesn't automatically route to Llama. It just throws an exception.

**No traffic splitting.** Want to test Claude 3.5 on 10% of traffic before full rollout? Manual work.

**No cross-model monitoring.** CloudWatch shows you Bedrock metrics, but comparing Claude vs. Llama performance? Build it yourself.

**No runtime configuration.** Want to change your model without redeploying? Good luck.

Bedrock is a **model hosting service**, not an orchestration layer. The same is true for OpenAI's API, Azure OpenAI, and others.

---

## The Agentic AI Gateway Pattern

The solution is an infrastructure layer between your application and LLM providers. I call it an **Agentic AI Gateway**.

```
┌─────────────┐     ┌─────────────┐     ┌────────────────────────────┐
│  Your App   │────▶│ Agentic AI Gateway │────▶│ Claude 3 Sonnet (Primary)  │
└─────────────┘     │             │     │ Claude 3.5 (Canary 10%)    │
                    │             │     │ Claude Haiku (Fallback)    │
                    │ • Routing   │     │ Llama 3 (Last resort)      │
                    │ • Fallback  │     └────────────────────────────┘
                    │ • Canary    │
                    │ • Metrics   │
                    └─────────────┘
```

The Agentic AI Gateway handles:

1. **Model Selection** - Which model should serve this request?
2. **Automatic Fallback** - If primary fails, try fallbacks in order
3. **Canary Deployments** - Route X% of traffic to a new model
4. **Metrics Collection** - Track latency, errors, and usage per model

---

## Building an Agentic AI Gateway

Here's the core implementation. It's surprisingly simple:

```python
class AgenticGateway:
    def __init__(self, config):
        self.config = config
        self.bedrock = boto3.client('bedrock-runtime')

    def _select_model(self):
        """Route to canary based on configured percentage."""
        if (
            self.config.canary_model
            and random.randint(1, 100) <= self.config.canary_percentage
        ):
            return self.config.canary_model, True
        return self.config.primary_model, False

    def invoke(self, prompt, **kwargs):
        """Invoke with automatic fallback."""
        # Build model chain: selected → fallbacks
        selected, is_canary = self._select_model()
        model_chain = [selected] + self.config.fallback_models

        for model_id in model_chain:
            try:
                start = time.time()
                response = self.bedrock.invoke_model(
                    modelId=model_id,
                    body=self._format_request(model_id, prompt, **kwargs)
                )

                return {
                    'content': self._parse_response(model_id, response),
                    'model_used': model_id,
                    'latency_ms': int((time.time() - start) * 1000),
                    'fallback_used': model_id != selected,
                    'canary_used': is_canary
                }

            except Exception as e:
                logger.warning(f"{model_id} failed: {e}, trying next...")
                continue

        raise Exception("All models failed")
```

That's the core pattern. About 50 lines of Python, and your app is now resilient to single-model failures.

---

## Canary Deployments for LLMs

Traditional software canary deployments are straightforward: route 5% of traffic to the new version, monitor, increase.

LLM canaries are trickier. You're not just checking "does it work?" You're checking:

- **Latency**: Is the new model slower?
- **Cost**: Does it use more tokens?
- **Quality**: Are the responses as good?

Here's how I structure LLM canary rollouts:

```
Week 1: Claude 3 Sonnet (100%) → Baseline metrics
        Start canary at 5%

Week 2: Claude 3 Sonnet (95%) + Claude 3.5 (5%)
        Monitor: latency within 10%, error rate < 1%

Week 3: Claude 3 Sonnet (80%) + Claude 3.5 (20%)
        Run evaluation suite on both

Week 4: Claude 3.5 (100%)
        or rollback if metrics degraded
```

The Agentic AI Gateway makes this a config change:

```python
# Start canary
gateway.update_config(
    canary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_percentage=5
)

# Increase after week 1
gateway.update_config(canary_percentage=20)

# Full rollout
gateway.update_config(
    primary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_model=None,
    canary_percentage=0
)
```

No redeployment. No code changes. Just configuration.

---

## Multi-Provider Fallback

Here's where it gets interesting. What if your entire provider goes down?

In December 2024, both OpenAI and Anthropic had outages in the same week. Companies with single-provider dependencies were scrambling.

The Agentic AI Gateway pattern supports cross-provider fallback:

```python
gateway = AgenticGateway(
    config=AgenticGatewayConfig(
        primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_models=[
            "anthropic.claude-3-haiku-20240307-v1:0",  # Same provider
            "meta.llama3-70b-instruct-v1:0",           # Different provider
            "gpt-4o-mini",                             # Different ecosystem
        ]
    )
)
```

The fallback chain tries:
1. Claude Sonnet (primary)
2. Claude Haiku (faster, cheaper, same provider)
3. Llama 3 70B (different provider on Bedrock)
4. GPT-4o Mini (different ecosystem entirely)

Your app stays up even if an entire provider goes down.

---

## The Cost of Not Having This

Let me be concrete about what this prevents:

| Scenario | Without Agentic AI Gateway | With Agentic AI Gateway |
|----------|-------------------|-----------------|
| Claude rate-limited | App errors, users see failures | Silent fallback to Haiku |
| Testing new model | Code change + deployment | Config change, 5% canary |
| Model comparison | Manual A/B test setup | Built-in metrics per model |
| Provider outage | Complete downtime | Automatic cross-provider fallback |
| Cost optimization | Hard to measure | Per-model cost tracking |

---

## What About AWS API Gateway?

You might think: "AWS API Gateway already does routing and fallbacks."

It does, but not for LLMs. API Gateway doesn't understand:

- Different request formats per model (Claude vs. Llama vs. Titan)
- Token-based cost tracking
- Quality-based routing decisions
- Model-specific parsing of responses

The Agentic AI Gateway is purpose-built for LLM routing. It's application-layer logic, not infrastructure.

---

## What About SageMaker?

SageMaker is for **hosting** models, not routing to them. You'd use SageMaker if you're deploying custom fine-tuned models. You'd use an Agentic AI Gateway to route between:

- Bedrock models (Claude, Llama)
- SageMaker endpoints (your custom models)
- External APIs (OpenAI)

They're complementary, not competing.

---

## Implementation Checklist

If you're building production LLM apps, here's what to implement:

**Day 1: Basic Fallback**
```python
model_chain = [primary_model, fallback_model]
for model in model_chain:
    try:
        return invoke(model, prompt)
    except:
        continue
```

**Week 1: Add Metrics**
- Log model used, latency, success/failure
- CloudWatch dashboard showing model distribution

**Week 2: Canary Support**
- Random percentage routing to canary model
- Runtime config updates without redeployment

**Week 3: Multi-Provider**
- Add second provider (OpenAI if using Bedrock, vice versa)
- Cross-provider fallback chain

---

## The Open Source Version

I've extracted this pattern into a standalone library. It's about 300 lines of Python with no infrastructure dependencies:

```bash
pip install agentic-ai-gateway[bedrock]
```

```python
from agentic_ai_gateway import create_bedrock_gateway

gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"],
    canary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_percentage=10
)

response = gateway.invoke("What is the capital of France?")
```

GitHub: [github.com/tyler-canton/agentic-ai-gateway](https://github.com/tyler-canton/agentic-ai-gateway)

---

## Conclusion

Every production LLM application needs an Agentic AI Gateway. Not because it's complex—it's actually simple. But because the alternative is a fragile system that fails when a single API call fails.

The pattern is:
1. **Fallback chains** - Never depend on a single model
2. **Canary deployments** - Test new models safely
3. **Multi-provider** - Survive provider outages
4. **Metrics** - Know what's actually happening

Build it yourself in an afternoon. Or use the library. Either way, your 3 AM self will thank you.

---

*Tyler Canton builds AI systems. He's currently working on healthcare AI with LangGraph, Bedrock, and Pinecone. Follow him on [LinkedIn/Twitter] for more production AI patterns.*
