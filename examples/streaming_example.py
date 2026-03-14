"""
Streaming Example
=================

Demonstrates real-time token streaming with the Agentic AI Gateway.

Features shown:
- Synchronous streaming with invoke_stream()
- Async streaming with ainvoke_stream()
- FastAPI SSE integration pattern
"""

import asyncio
from agentic_ai_gateway import create_bedrock_gateway


def sync_streaming_example():
    """Basic synchronous streaming example."""
    print("=== Synchronous Streaming ===\n")

    gateway = create_bedrock_gateway(
        primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"],
        region_name="us-east-1"
    )

    print("Prompt: Tell me a short story about a robot.\n")
    print("Response: ", end="")

    for chunk in gateway.invoke_stream("Tell me a short story about a robot in 2-3 sentences."):
        if chunk["type"] == "start":
            print(f"\n[Using model: {chunk['model_used']}]")
            if chunk.get("fallback_used"):
                print("[Fallback activated]")
        elif chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "done":
            print(f"\n\n[Completed in {chunk['latency_ms']}ms]")
            print(f"[Output tokens: {chunk['output_tokens']}]")
        elif chunk["type"] == "error":
            print(f"\n[Error: {chunk['error']}]")


async def async_streaming_example():
    """Async streaming example for use with FastAPI/aiohttp."""
    print("\n=== Async Streaming ===\n")

    gateway = create_bedrock_gateway(
        primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"],
        region_name="us-east-1"
    )

    print("Prompt: Explain quantum computing briefly.\n")
    print("Response: ", end="")

    async for chunk in gateway.ainvoke_stream("Explain quantum computing in 2-3 sentences."):
        if chunk["type"] == "start":
            print(f"\n[Using model: {chunk['model_used']}]")
        elif chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "done":
            print(f"\n\n[Completed in {chunk['latency_ms']}ms]")


def fastapi_sse_pattern():
    """
    FastAPI SSE integration pattern (code example, not runnable).

    This shows how to integrate streaming with FastAPI's StreamingResponse
    for Server-Sent Events (SSE).
    """
    print("\n=== FastAPI SSE Pattern ===\n")

    code_example = '''
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"]
)

@app.post("/api/v1/query/stream")
async def stream_query(prompt: str):
    async def generate():
        async for chunk in gateway.ainvoke_stream(prompt):
            if chunk["type"] == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': chunk['content']})}\\n\\n"
            elif chunk["type"] == "done":
                yield f"data: {json.dumps({'type': 'done', 'model_used': chunk['model_used']})}\\n\\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
'''
    print(code_example)


if __name__ == "__main__":
    # Run sync example
    sync_streaming_example()

    # Run async example
    asyncio.run(async_streaming_example())

    # Show FastAPI pattern
    fastapi_sse_pattern()
