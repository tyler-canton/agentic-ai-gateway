"""
Streaming Response Module
=========================

Real-time streaming responses for production LLM applications.
No more blank screens while waiting for responses.

Features:
- Async streaming with proper backpressure
- Token-by-token or chunk-by-chunk delivery
- Fallback support (switches mid-stream if primary fails)
- Metrics collection for streaming sessions

Author: Tyler Canton
License: MIT
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Iterator, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Streaming Types
# ============================================================================

class StreamEventType(str, Enum):
    """Types of streaming events."""
    START = "start"              # Stream started
    CONTENT = "content"          # Content chunk
    TOOL_USE = "tool_use"        # Tool/function call
    METADATA = "metadata"        # Metadata update
    ERROR = "error"              # Error occurred
    END = "end"                  # Stream completed


@dataclass
class StreamChunk:
    """A single chunk in a stream."""
    event_type: StreamEventType
    content: str = ""
    index: int = 0
    model_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type.value,
            "content": self.content,
            "index": self.index,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        import json
        data = json.dumps(self.to_dict())
        return f"event: {self.event_type.value}\ndata: {data}\n\n"


@dataclass
class StreamingResponse:
    """Complete streaming response with metrics."""
    content: str                          # Full accumulated content
    model_used: str
    chunks_count: int
    start_time: datetime
    end_time: datetime
    time_to_first_chunk_ms: float         # TTFC - critical UX metric
    total_duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    fallback_used: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Calculate output speed."""
        if self.total_duration_ms == 0:
            return 0
        return (self.output_tokens / self.total_duration_ms) * 1000


# ============================================================================
# Bedrock Streaming
# ============================================================================

class BedrockStreamHandler:
    """
    Handle streaming responses from AWS Bedrock.

    Supports:
    - Claude models (Anthropic)
    - Titan models (Amazon)
    - Llama models (Meta)
    """

    def __init__(self, bedrock_client, model_id: str):
        self.client = bedrock_client
        self.model_id = model_id

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream response from Bedrock.

        Yields StreamChunk objects as they arrive.
        """
        import json

        start_time = time.time()
        first_chunk_time = None
        chunk_index = 0

        # Build request body based on model
        if "anthropic" in self.model_id.lower():
            body = self._build_anthropic_body(
                prompt, system, messages, max_tokens, temperature, **kwargs
            )
        else:
            body = self._build_generic_body(
                prompt, max_tokens, temperature, **kwargs
            )

        # Yield start event
        yield StreamChunk(
            event_type=StreamEventType.START,
            model_id=self.model_id,
            metadata={"prompt_length": len(prompt)}
        )

        try:
            # Call Bedrock with streaming
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            # Process stream
            for event in response.get("body", []):
                if "chunk" in event:
                    chunk_data = json.loads(event["chunk"]["bytes"].decode())

                    # Extract content based on model type
                    content = self._extract_content(chunk_data)

                    if content:
                        if first_chunk_time is None:
                            first_chunk_time = time.time()

                        yield StreamChunk(
                            event_type=StreamEventType.CONTENT,
                            content=content,
                            index=chunk_index,
                            model_id=self.model_id,
                            metadata={"raw": chunk_data}
                        )
                        chunk_index += 1

            # Yield end event
            end_time = time.time()
            ttfc = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0

            yield StreamChunk(
                event_type=StreamEventType.END,
                model_id=self.model_id,
                metadata={
                    "chunks_count": chunk_index,
                    "time_to_first_chunk_ms": ttfc,
                    "total_duration_ms": (end_time - start_time) * 1000,
                }
            )

        except Exception as e:
            logger.error(f"[Streaming] Error: {e}")
            yield StreamChunk(
                event_type=StreamEventType.ERROR,
                content=str(e),
                model_id=self.model_id,
            )

    def _build_anthropic_body(
        self,
        prompt: str,
        system: Optional[str],
        messages: Optional[list],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request body for Anthropic Claude."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            body["system"] = system

        if messages:
            body["messages"] = messages
        else:
            body["messages"] = [{"role": "user", "content": prompt}]

        return body

    def _build_generic_body(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Build generic request body."""
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _extract_content(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text content from chunk based on model type."""
        # Anthropic Claude format
        if "delta" in chunk_data:
            delta = chunk_data.get("delta", {})
            if "text" in delta:
                return delta["text"]

        # Content block format
        if "content_block" in chunk_data:
            block = chunk_data.get("content_block", {})
            if "text" in block:
                return block["text"]

        # Generic text format
        if "text" in chunk_data:
            return chunk_data["text"]

        # Completion format
        if "completion" in chunk_data:
            return chunk_data["completion"]

        return ""


# ============================================================================
# Streaming Gateway
# ============================================================================

class StreamingGateway:
    """
    Gateway with streaming support.

    Real-time token-by-token responses for production apps.
    Supports fallbacks, metrics, and callbacks.

    Example (Async):
        from agentic_ai_gateway.streaming import StreamingGateway
        import boto3

        bedrock = boto3.client("bedrock-runtime")
        gateway = StreamingGateway(
            bedrock_client=bedrock,
            primary_model="anthropic.claude-3-sonnet-20240229-v1:0"
        )

        # Stream response
        async for chunk in gateway.stream("Write a poem about AI"):
            print(chunk.content, end="", flush=True)

    Example (FastAPI):
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse

        @app.post("/chat/stream")
        async def chat_stream(prompt: str):
            async def generate():
                async for chunk in gateway.stream(prompt):
                    yield chunk.to_sse()

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )

    Example (Sync wrapper):
        # For non-async code
        full_response = gateway.invoke_streaming("Write a poem")
        print(full_response.content)
    """

    def __init__(
        self,
        bedrock_client=None,
        primary_model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_models: Optional[list] = None,
        on_chunk: Optional[Callable[[StreamChunk], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize streaming gateway.

        Args:
            bedrock_client: boto3 bedrock-runtime client
            primary_model: Primary model ID
            fallback_models: List of fallback model IDs
            on_chunk: Callback for each chunk (for metrics/logging)
            on_error: Callback for errors
        """
        self.bedrock_client = bedrock_client
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.on_chunk = on_chunk
        self.on_error = on_error

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream response asynchronously.

        Yields StreamChunk objects as they arrive.
        Falls back to secondary models on failure.
        """
        models_to_try = [self.primary_model] + self.fallback_models

        for model_id in models_to_try:
            try:
                handler = BedrockStreamHandler(self.bedrock_client, model_id)

                async for chunk in handler.stream(
                    prompt=prompt,
                    system=system,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                ):
                    # Call chunk callback if provided
                    if self.on_chunk:
                        self.on_chunk(chunk)

                    yield chunk

                    # Check for error and try fallback
                    if chunk.event_type == StreamEventType.ERROR:
                        raise Exception(chunk.content)

                # Success - don't try fallbacks
                return

            except Exception as e:
                logger.warning(f"[Streaming] {model_id} failed: {e}, trying fallback...")

                if self.on_error:
                    self.on_error(e)

                # Continue to next model
                continue

        # All models failed
        yield StreamChunk(
            event_type=StreamEventType.ERROR,
            content="All models failed",
        )

    def invoke_streaming(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs
    ) -> StreamingResponse:
        """
        Synchronous wrapper that collects full streaming response.

        Returns complete StreamingResponse with metrics.
        """
        import asyncio

        async def collect():
            chunks = []
            content_parts = []
            start_time = datetime.now()
            first_chunk_time = None
            model_used = self.primary_model
            error = None

            async for chunk in self.stream(prompt, system, messages, **kwargs):
                chunks.append(chunk)

                if chunk.event_type == StreamEventType.CONTENT:
                    content_parts.append(chunk.content)
                    if first_chunk_time is None:
                        first_chunk_time = datetime.now()

                if chunk.model_id:
                    model_used = chunk.model_id

                if chunk.event_type == StreamEventType.ERROR:
                    error = chunk.content

            end_time = datetime.now()
            ttfc = 0
            if first_chunk_time:
                ttfc = (first_chunk_time - start_time).total_seconds() * 1000

            return StreamingResponse(
                content="".join(content_parts),
                model_used=model_used,
                chunks_count=len([c for c in chunks if c.event_type == StreamEventType.CONTENT]),
                start_time=start_time,
                end_time=end_time,
                time_to_first_chunk_ms=ttfc,
                total_duration_ms=(end_time - start_time).total_seconds() * 1000,
                error=error,
            )

        # Run async in sync context
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(collect())
        finally:
            loop.close()

    def stream_to_string(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Simple helper: stream and return full string.

        For when you just want the content.
        """
        response = self.invoke_streaming(prompt, **kwargs)
        return response.content


# ============================================================================
# Utility: SSE Formatter for FastAPI
# ============================================================================

def format_sse(data: str, event: Optional[str] = None) -> str:
    """Format data as Server-Sent Event."""
    msg = ""
    if event:
        msg += f"event: {event}\n"
    msg += f"data: {data}\n\n"
    return msg


async def stream_to_sse(
    gateway: StreamingGateway,
    prompt: str,
    **kwargs
) -> AsyncIterator[str]:
    """
    Convert gateway stream to SSE format.

    Use with FastAPI StreamingResponse:

        @app.post("/stream")
        async def stream_endpoint(prompt: str):
            return StreamingResponse(
                stream_to_sse(gateway, prompt),
                media_type="text/event-stream"
            )
    """
    async for chunk in gateway.stream(prompt, **kwargs):
        yield chunk.to_sse()
