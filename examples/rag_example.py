"""
Agentic AI Gateway - RAG Pipeline Integration Example

Shows how to integrate the gateway into a RAG (Retrieval-Augmented Generation)
pipeline for resilient document Q&A with automatic fallbacks.
"""

from agentic_ai_gateway import create_bedrock_gateway, AgenticGatewayConfig, AgenticGateway, BedrockProvider

# Create gateway with fallbacks optimized for RAG
gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=[
        "anthropic.claude-3-haiku-20240307-v1:0",  # Faster, cheaper fallback
        "meta.llama3-70b-instruct-v1:0",           # Cross-provider fallback
    ],
    canary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_percentage=10,  # Test new model on 10% of traffic
    region="us-east-1"
)


def create_rag_prompt(question: str, context_chunks: list[str]) -> str:
    """Create a RAG prompt with retrieved context."""
    context = "\n\n---\n\n".join(context_chunks)
    return f"""Answer the question based on the context provided below.
If the answer cannot be found in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


# Example: Simple RAG query
def rag_query(question: str, retrieved_chunks: list[str]) -> dict:
    """Execute a RAG query with automatic fallback."""
    prompt = create_rag_prompt(question, retrieved_chunks)

    response = gateway.invoke(
        prompt=prompt,
        max_tokens=500,
        temperature=0.3  # Lower temperature for factual answers
    )

    return {
        "answer": response.content,
        "model_used": response.model_used,
        "latency_ms": response.latency_ms,
        "fallback_used": response.fallback_used,
        "canary_used": response.canary_used,
        "tokens": {
            "input": response.input_tokens,
            "output": response.output_tokens
        }
    }


# Simulated retrieved chunks (would come from Pinecone/vector DB)
sample_chunks = [
    """Patient John Doe (P001) was admitted on January 15, 2024 with
    symptoms of chest pain and shortness of breath. Initial diagnosis
    indicated possible cardiac involvement.""",

    """Lab results for patient P001 showed elevated troponin levels
    (0.15 ng/mL) and slightly elevated BNP (250 pg/mL). ECG showed
    ST-segment depression in leads V1-V4.""",

    """Treatment plan for P001 includes aspirin 81mg daily,
    metoprolol 25mg twice daily, and scheduled cardiac catheterization
    within 48 hours."""
]

# Run RAG query
result = rag_query(
    question="What medications was patient P001 prescribed?",
    retrieved_chunks=sample_chunks
)

print("RAG Query Result:")
print(f"Answer: {result['answer']}")
print(f"Model: {result['model_used']}")
print(f"Latency: {result['latency_ms']}ms")
print(f"Fallback used: {result['fallback_used']}")
print(f"Canary used: {result['canary_used']}")
print(f"Tokens: {result['tokens']}")


# Example: RAG with streaming (async)
async def rag_query_async(question: str, retrieved_chunks: list[str]) -> dict:
    """Async RAG query for better concurrency."""
    prompt = create_rag_prompt(question, retrieved_chunks)

    response = await gateway.ainvoke(
        prompt=prompt,
        max_tokens=500,
        temperature=0.3
    )

    return {
        "answer": response.content,
        "model_used": response.model_used,
        "latency_ms": response.latency_ms
    }


# Example: Integration with existing RAG pipeline class
class ResilientRAGPipeline:
    """RAG pipeline with resilient LLM routing via Agentic Gateway."""

    def __init__(
        self,
        vector_store,  # Your Pinecone/vector store client
        primary_model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_models: list[str] = None,
        region: str = "us-east-1"
    ):
        self.vector_store = vector_store
        self.gateway = create_bedrock_gateway(
            primary_model=primary_model,
            fallback_models=fallback_models or [
                "anthropic.claude-3-haiku-20240307-v1:0"
            ],
            region=region
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: dict = None
    ) -> dict:
        """
        Execute a RAG query with retrieval and generation.

        The gateway handles:
        - Automatic fallback if primary model fails
        - Canary deployments for testing new models
        - Latency and error tracking
        """
        # 1. Retrieve relevant chunks
        # chunks = self.vector_store.query(
        #     query=question,
        #     top_k=top_k,
        #     filter=metadata_filter
        # )
        chunks = sample_chunks  # Mock for example

        # 2. Generate answer with resilient LLM routing
        prompt = create_rag_prompt(question, chunks)

        response = self.gateway.invoke(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )

        return {
            "answer": response.content,
            "sources": chunks,
            "model_used": response.model_used,
            "latency_ms": response.latency_ms,
            "fallback_used": response.fallback_used
        }

    def update_routing(
        self,
        canary_model: str = None,
        canary_percentage: int = None
    ):
        """Update model routing without redeployment."""
        self.gateway.update_config(
            canary_model=canary_model,
            canary_percentage=canary_percentage
        )

    def get_metrics(self) -> dict:
        """Get LLM routing metrics."""
        return self.gateway.get_metrics()


# Usage
print("\n" + "=" * 50)
print("Using ResilientRAGPipeline:")
print("=" * 50)

# Initialize (vector_store would be your real Pinecone client)
pipeline = ResilientRAGPipeline(
    vector_store=None,  # Mock
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"]
)

# Query
result = pipeline.query("What is patient P001's diagnosis?")
print(f"Answer: {result['answer']}")
print(f"Model: {result['model_used']}")

# Update canary for A/B testing new model
pipeline.update_routing(
    canary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_percentage=20
)

# Run more queries - 20% will use canary
for i in range(5):
    result = pipeline.query("What medications were prescribed?")
    canary_tag = " (CANARY)" if result.get("canary_used") else ""
    print(f"Query {i+1}: {result['model_used']}{canary_tag}")

# Check metrics
print("\nPipeline Metrics:")
print(pipeline.get_metrics())
