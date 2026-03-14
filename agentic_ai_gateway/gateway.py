"""
Agentic AI Gateway
===================

Production-grade LLM routing with automatic fallbacks, canary deployments,
and multi-provider support.

This module provides the core gateway infrastructure for routing LLM requests
with automatic fallback, A/B testing via canary deployments, and unified
metrics collection across providers.

Author: Tyler Canton
GitHub: https://github.com/tyler-canton
Package: https://pypi.org/project/agentic-ai-gateway/
License: MIT

Copyright (c) 2026 Tyler Canton. All rights reserved.
"""

import json
import random
import time
import logging
from typing import Optional, Dict, Any, List, Protocol, Callable, Generator, AsyncGenerator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AIGatewayConfig:
    """AI Gateway configuration."""
    primary_model: str
    canary_model: Optional[str] = None
    canary_percentage: int = 0  # 0-100
    fallback_models: List[str] = field(default_factory=list)
    max_retries: int = 2
    timeout_seconds: int = 30


@dataclass
class AIGatewayResponse:
    """Response from AI Gateway invocation."""
    content: str
    model_used: str
    latency_ms: int
    fallback_used: bool
    canary_used: bool
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Provider Abstraction
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def invoke(self, model_id: str, prompt: str, **kwargs) -> tuple[str, int, int]:
        """
        Invoke the model and return (content, input_tokens, output_tokens).
        """
        pass

    @abstractmethod
    def supports_model(self, model_id: str) -> bool:
        """Check if this provider supports the given model ID."""
        pass


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider."""

    def __init__(self, region_name: str = "us-east-1"):
        import boto3
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def supports_model(self, model_id: str) -> bool:
        return any(p in model_id for p in ["anthropic", "meta", "amazon", "cohere", "ai21"])

    def invoke(self, model_id: str, prompt: str, **kwargs) -> tuple[str, int, int]:
        body = self._format_request(model_id, prompt, **kwargs)
        response = self.client.invoke_model(modelId=model_id, body=body)
        return self._parse_response(model_id, response)

    def _format_request(self, model_id: str, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", 1024)
        temperature = kwargs.get("temperature", 0.7)

        if "anthropic" in model_id:
            return json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            })
        elif "meta" in model_id:
            return json.dumps({
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_gen_len": max_tokens,
                "temperature": temperature
            })
        elif "amazon" in model_id:
            return json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature
                }
            })
        else:
            # Default to Anthropic format
            return json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            })

    def _parse_response(self, model_id: str, response: Dict) -> tuple[str, int, int]:
        body = json.loads(response["body"].read())

        if "anthropic" in model_id:
            content = body.get("content", [{}])[0].get("text", "")
            usage = body.get("usage", {})
            return content, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        elif "meta" in model_id:
            return body.get("generation", ""), 0, 0
        elif "amazon" in model_id:
            results = body.get("results", [{}])
            content = results[0].get("outputText", "") if results else ""
            return content, 0, 0
        else:
            return str(body), 0, 0

    def converse(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        inference_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call Bedrock Converse API with tool support.

        This is used for multi-agent workflows where tool calling is needed.
        The Converse API provides a unified interface across models.

        Args:
            model_id: The model to use
            messages: Conversation messages
            system: System prompts
            tool_config: Tool definitions for tool calling
            inference_config: Max tokens, temperature, etc.

        Returns:
            Raw Bedrock converse response
        """
        kwargs = {
            "modelId": model_id,
            "messages": messages
        }

        if system:
            kwargs["system"] = system
        if tool_config:
            kwargs["toolConfig"] = tool_config
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        return self.client.converse(**kwargs)

    def invoke_stream(
        self,
        model_id: str,
        prompt: str,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tokens from Bedrock using invoke_model_with_response_stream.

        This method yields individual tokens as they arrive from the model,
        enabling real-time streaming in chat interfaces.

        Args:
            model_id: The Bedrock model ID (e.g., "anthropic.claude-3-sonnet...")
            prompt: The user prompt to send to the model
            **kwargs: Additional parameters like max_tokens, temperature

        Yields:
            Dict with keys:
                - type: "token" for content, "done" for completion, "error" for errors
                - content: The token text (for type="token")
                - input_tokens: Token count (for type="done")
                - output_tokens: Token count (for type="done")

        Example:
            for chunk in provider.invoke_stream(model_id, prompt):
                if chunk["type"] == "token":
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "done":
                    print(f"\\nTokens: {chunk['output_tokens']}")
        """
        # Step 1: Format the request body for the specific model type
        # Different models (Anthropic, Meta, Amazon) have different request formats
        body = self._format_request(model_id, prompt, **kwargs)

        # Step 2: Call Bedrock's streaming API
        # invoke_model_with_response_stream returns an EventStream
        response = self.client.invoke_model_with_response_stream(
            modelId=model_id,
            body=body
        )

        # Step 3: Track token counts for metrics
        input_tokens = 0
        output_tokens = 0

        # Step 4: Iterate through the event stream
        # Each event contains a chunk of the response
        for event in response.get("body"):
            # Step 5: Parse the chunk bytes as JSON
            chunk = json.loads(event["chunk"]["bytes"])

            # Step 6: Handle different event types based on model provider
            if "anthropic" in model_id:
                # Anthropic Claude models use specific event types
                if chunk["type"] == "content_block_delta":
                    # content_block_delta contains the actual token text
                    token = chunk["delta"].get("text", "")
                    yield {"type": "token", "content": token}

                elif chunk["type"] == "message_delta":
                    # message_delta contains usage statistics
                    usage = chunk.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)

                elif chunk["type"] == "message_start":
                    # message_start contains input token count
                    usage = chunk.get("message", {}).get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)

                elif chunk["type"] == "message_stop":
                    # message_stop signals completion
                    yield {
                        "type": "done",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
                    return

            elif "meta" in model_id:
                # Meta Llama models have different structure
                if "generation" in chunk:
                    yield {"type": "token", "content": chunk["generation"]}

            elif "amazon" in model_id:
                # Amazon Titan models
                if "outputText" in chunk:
                    yield {"type": "token", "content": chunk["outputText"]}

            else:
                # Generic fallback for unknown models
                if isinstance(chunk, dict):
                    text = chunk.get("text", chunk.get("content", str(chunk)))
                    yield {"type": "token", "content": text}

        # Step 7: Final done event if not already sent
        yield {"type": "done", "input_tokens": input_tokens, "output_tokens": output_tokens}


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""

    def __init__(self, api_key: Optional[str] = None):
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def supports_model(self, model_id: str) -> bool:
        return any(p in model_id for p in ["gpt-4", "gpt-3.5", "o1", "o3"])

    def invoke(self, model_id: str, prompt: str, **kwargs) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7)
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        return (
            content,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0
        )


# ============================================================================
# Metrics
# ============================================================================

class MetricsCollector(Protocol):
    """Protocol for metrics collection."""

    def record(
        self,
        model_id: str,
        latency_ms: int,
        success: bool,
        is_canary: bool,
        is_fallback: bool,
        error: Optional[str] = None
    ) -> None:
        ...


class InMemoryMetrics:
    """Simple in-memory metrics collector."""

    def __init__(self):
        self._data = {
            "invocations": 0,
            "fallbacks": 0,
            "errors": 0,
            "total_latency_ms": 0,
            "by_model": {}
        }

    def record(
        self,
        model_id: str,
        latency_ms: int,
        success: bool,
        is_canary: bool,
        is_fallback: bool,
        error: Optional[str] = None
    ) -> None:
        self._data["invocations"] += 1
        self._data["total_latency_ms"] += latency_ms

        if is_fallback:
            self._data["fallbacks"] += 1
        if not success:
            self._data["errors"] += 1

        if model_id not in self._data["by_model"]:
            self._data["by_model"][model_id] = {
                "invocations": 0, "errors": 0, "total_latency_ms": 0
            }

        self._data["by_model"][model_id]["invocations"] += 1
        self._data["by_model"][model_id]["total_latency_ms"] += latency_ms
        if not success:
            self._data["by_model"][model_id]["errors"] += 1

        # Log
        status = "SUCCESS" if success else f"FAILED: {error}"
        tags = []
        if is_canary:
            tags.append("CANARY")
        if is_fallback:
            tags.append("FALLBACK")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        logger.info(f"[AIGateway] {model_id}{tag_str} - {latency_ms}ms - {status}")

    def get_stats(self) -> Dict[str, Any]:
        total = self._data["invocations"]
        return {
            "total_invocations": total,
            "total_errors": self._data["errors"],
            "error_rate": self._data["errors"] / total if total > 0 else 0,
            "fallback_rate": self._data["fallbacks"] / total if total > 0 else 0,
            "avg_latency_ms": self._data["total_latency_ms"] / total if total > 0 else 0,
            "by_model": self._data["by_model"]
        }


# ============================================================================
# AI Gateway
# ============================================================================

class AIGateway:
    """
    AI Gateway for routing LLM requests.

    Sits between your application and LLM providers.
    Handles model selection, fallbacks, and monitoring.
    """

    def __init__(
        self,
        config: AIGatewayConfig,
        providers: Optional[List[LLMProvider]] = None,
        metrics: Optional[MetricsCollector] = None
    ):
        self.config = config
        self.providers = providers or []
        self.metrics = metrics or InMemoryMetrics()

    def add_provider(self, provider: LLMProvider) -> "AIGateway":
        """Add a provider (fluent interface)."""
        self.providers.append(provider)
        return self

    def _get_provider(self, model_id: str) -> Optional[LLMProvider]:
        """Find a provider that supports the given model."""
        for provider in self.providers:
            if provider.supports_model(model_id):
                return provider
        return None

    def _select_model(self) -> tuple[str, bool]:
        """Select model based on canary configuration."""
        if (
            self.config.canary_model
            and self.config.canary_percentage > 0
            and random.randint(1, 100) <= self.config.canary_percentage
        ):
            return self.config.canary_model, True
        return self.config.primary_model, False

    def _build_model_chain(self, force_model: Optional[str] = None) -> List[str]:
        """Build ordered list of models to try."""
        if force_model:
            return [force_model]

        selected, _ = self._select_model()
        chain = [selected]

        for fallback in self.config.fallback_models:
            if fallback != selected:
                chain.append(fallback)

        return chain

    def invoke(
        self,
        prompt: str,
        force_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AIGatewayResponse:
        """
        Invoke model with automatic fallback.

        Args:
            prompt: The prompt to send
            force_model: Force specific model (bypass canary/routing)
            metadata: Optional metadata for tracking
            **kwargs: Additional args (max_tokens, temperature, etc.)

        Returns:
            AIGatewayResponse with content, model used, latency, etc.

        Raises:
            Exception: If all models fail
        """
        model_chain = self._build_model_chain(force_model)
        selected_model, is_canary = self._select_model()

        last_error = None

        for i, model_id in enumerate(model_chain):
            is_fallback = (i > 0)
            start_time = time.time()

            try:
                provider = self._get_provider(model_id)
                if not provider:
                    raise ValueError(f"No provider found for model: {model_id}")

                content, input_tokens, output_tokens = provider.invoke(
                    model_id, prompt, **kwargs
                )

                latency_ms = int((time.time() - start_time) * 1000)

                self.metrics.record(
                    model_id=model_id,
                    latency_ms=latency_ms,
                    success=True,
                    is_canary=(model_id == selected_model and is_canary),
                    is_fallback=is_fallback
                )

                return AIGatewayResponse(
                    content=content,
                    model_used=model_id,
                    latency_ms=latency_ms,
                    fallback_used=is_fallback,
                    canary_used=(model_id == selected_model and is_canary),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata=metadata or {}
                )

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                last_error = str(e)

                self.metrics.record(
                    model_id=model_id,
                    latency_ms=latency_ms,
                    success=False,
                    is_canary=(model_id == selected_model and is_canary),
                    is_fallback=is_fallback,
                    error=last_error
                )

                logger.warning(f"[AIGateway] {model_id} failed: {last_error}, trying next...")
                continue

        raise Exception(f"[AIGateway] All models failed. Last error: {last_error}")

    async def ainvoke(
        self,
        prompt: str,
        force_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AIGatewayResponse:
        """Async version of invoke."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.invoke(prompt, force_model, metadata, **kwargs)
        )

    def invoke_stream(
        self,
        prompt: str,
        force_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tokens with automatic fallback support.

        This method provides real-time token streaming while maintaining
        the gateway's fallback capabilities. If the primary model fails
        during streaming, it will NOT automatically fallback (streaming
        must start fresh). However, if the initial connection fails,
        fallback models will be attempted.

        Args:
            prompt: The prompt to send to the model
            force_model: Force specific model (bypass canary/routing)
            metadata: Optional metadata for tracking
            **kwargs: Additional args (max_tokens, temperature, etc.)

        Yields:
            Dict with keys:
                - type: "start", "token", "done", or "error"
                - content: Token text (for type="token")
                - model_used: Which model is streaming (for type="start")
                - fallback_used: Whether fallback was used (for type="start")
                - input_tokens: Token count (for type="done")
                - output_tokens: Token count (for type="done")
                - latency_ms: Total request time (for type="done")
                - error: Error message (for type="error")

        Example:
            for chunk in gateway.invoke_stream("Tell me a story"):
                if chunk["type"] == "start":
                    print(f"Using model: {chunk['model_used']}")
                elif chunk["type"] == "token":
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "done":
                    print(f"\\nCompleted in {chunk['latency_ms']}ms")
        """
        # Step 1: Build the model chain (primary → fallbacks)
        # This determines the order of models to try
        model_chain = self._build_model_chain(force_model)
        selected_model, is_canary = self._select_model()

        last_error = None

        # Step 2: Try each model in the chain until one succeeds
        for i, model_id in enumerate(model_chain):
            is_fallback = (i > 0)
            start_time = time.time()

            try:
                # Step 3: Find a provider that supports this model
                provider = self._get_provider(model_id)
                if not provider:
                    raise ValueError(f"No provider found for model: {model_id}")

                # Step 4: Verify the provider supports streaming
                if not hasattr(provider, 'invoke_stream'):
                    raise ValueError(f"Provider for {model_id} does not support streaming")

                # Step 5: Emit start event with model info
                # This tells the consumer which model is being used
                yield {
                    "type": "start",
                    "model_used": model_id,
                    "fallback_used": is_fallback,
                    "canary_used": (model_id == selected_model and is_canary),
                    "message": "Streaming started..."
                }

                # Step 6: Stream tokens from the provider
                # Each token is yielded as it arrives from the model
                input_tokens = 0
                output_tokens = 0

                for chunk in provider.invoke_stream(model_id, prompt, **kwargs):
                    if chunk["type"] == "token":
                        # Pass through token events directly
                        yield chunk
                    elif chunk["type"] == "done":
                        # Capture token counts from the done event
                        input_tokens = chunk.get("input_tokens", 0)
                        output_tokens = chunk.get("output_tokens", 0)

                # Step 7: Calculate latency and record metrics
                latency_ms = int((time.time() - start_time) * 1000)

                self.metrics.record(
                    model_id=model_id,
                    latency_ms=latency_ms,
                    success=True,
                    is_canary=(model_id == selected_model and is_canary),
                    is_fallback=is_fallback
                )

                # Step 8: Emit done event with final statistics
                yield {
                    "type": "done",
                    "model_used": model_id,
                    "latency_ms": latency_ms,
                    "fallback_used": is_fallback,
                    "canary_used": (model_id == selected_model and is_canary),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "metadata": metadata or {}
                }

                # Success! Exit the retry loop
                return

            except Exception as e:
                # Step 9: Handle errors and try next model in chain
                latency_ms = int((time.time() - start_time) * 1000)
                last_error = str(e)

                self.metrics.record(
                    model_id=model_id,
                    latency_ms=latency_ms,
                    success=False,
                    is_canary=(model_id == selected_model and is_canary),
                    is_fallback=is_fallback,
                    error=last_error
                )

                logger.warning(f"[AIGateway] {model_id} stream failed: {last_error}, trying next...")
                continue

        # Step 10: All models failed, emit error event
        yield {
            "type": "error",
            "error": f"All models failed. Last error: {last_error}"
        }

    async def ainvoke_stream(
        self,
        prompt: str,
        force_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async streaming with automatic fallback support.

        This is the async version of invoke_stream, suitable for use in
        async frameworks like FastAPI, aiohttp, or asyncio applications.

        Args:
            prompt: The prompt to send to the model
            force_model: Force specific model (bypass canary/routing)
            metadata: Optional metadata for tracking
            **kwargs: Additional args (max_tokens, temperature, etc.)

        Yields:
            Same as invoke_stream - dicts with type, content, etc.

        Example:
            async for chunk in gateway.ainvoke_stream("Tell me a story"):
                if chunk["type"] == "token":
                    await websocket.send(chunk["content"])
        """
        import asyncio

        # Run the sync generator in a thread pool to avoid blocking
        # the event loop while waiting for Bedrock responses
        loop = asyncio.get_event_loop()

        # Create the sync generator
        sync_gen = self.invoke_stream(prompt, force_model, metadata, **kwargs)

        # Wrap in async generator
        def get_next():
            try:
                return next(sync_gen)
            except StopIteration:
                return None

        while True:
            # Run next() in executor to not block the event loop
            chunk = await loop.run_in_executor(None, get_next)
            if chunk is None:
                break
            yield chunk

    def update_config(
        self,
        canary_model: Optional[str] = None,
        canary_percentage: Optional[int] = None,
        fallback_models: Optional[List[str]] = None
    ) -> None:
        """Update configuration at runtime."""
        if canary_model is not None:
            self.config.canary_model = canary_model
        if canary_percentage is not None:
            self.config.canary_percentage = canary_percentage
        if fallback_models is not None:
            self.config.fallback_models = fallback_models

        logger.info(
            f"[AIGateway] Config updated: canary={self.config.canary_model} "
            f"({self.config.canary_percentage}%), "
            f"fallbacks={self.config.fallback_models}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if hasattr(self.metrics, "get_stats"):
            return self.metrics.get_stats()
        return {}

    def converse(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        force_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke model using Bedrock Converse API with automatic fallback.

        Used for multi-agent workflows where tool calling is needed.
        The Converse API provides a unified interface across models.

        Args:
            messages: Conversation messages
            system: System prompts
            tool_config: Tool definitions for tool calling
            inference_config: Max tokens, temperature, etc.
            force_model: Force specific model (bypass canary/routing)
            metadata: Optional metadata for tracking

        Returns:
            Dict containing:
                - response: Raw Bedrock converse response
                - model_used: Which model handled the request
                - latency_ms: Request latency
                - fallback_used: Whether a fallback model was used
                - canary_used: Whether canary model was used

        Raises:
            Exception: If all models fail
        """
        model_chain = self._build_model_chain(force_model)
        selected_model, is_canary = self._select_model()

        last_error = None

        for i, model_id in enumerate(model_chain):
            is_fallback = (i > 0)
            start_time = time.time()

            try:
                provider = self._get_provider(model_id)
                if not provider:
                    raise ValueError(f"No provider found for model: {model_id}")

                if not hasattr(provider, 'converse'):
                    raise ValueError(f"Provider for {model_id} does not support converse()")

                response = provider.converse(
                    model_id=model_id,
                    messages=messages,
                    system=system,
                    tool_config=tool_config,
                    inference_config=inference_config
                )

                latency_ms = int((time.time() - start_time) * 1000)

                # Extract token usage if available
                usage = response.get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)

                self.metrics.record(
                    model_id=model_id,
                    latency_ms=latency_ms,
                    success=True,
                    is_canary=(model_id == selected_model and is_canary),
                    is_fallback=is_fallback
                )

                return {
                    "response": response,
                    "model_used": model_id,
                    "latency_ms": latency_ms,
                    "fallback_used": is_fallback,
                    "canary_used": (model_id == selected_model and is_canary),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "metadata": metadata or {}
                }

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                last_error = str(e)

                self.metrics.record(
                    model_id=model_id,
                    latency_ms=latency_ms,
                    success=False,
                    is_canary=(model_id == selected_model and is_canary),
                    is_fallback=is_fallback,
                    error=last_error
                )

                logger.warning(f"[AIGateway] {model_id} converse failed: {last_error}, trying next...")
                continue

        raise Exception(f"[AIGateway] All models failed for converse. Last error: {last_error}")


# ============================================================================
# Factory Functions
# ============================================================================

def create_bedrock_gateway(
    primary_model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models: Optional[List[str]] = None,
    canary_model: Optional[str] = None,
    canary_percentage: int = 0,
    region: str = "us-east-1"
) -> AIGateway:
    """Create an AI Gateway configured for AWS Bedrock."""
    config = AIGatewayConfig(
        primary_model=primary_model,
        fallback_models=fallback_models or ["anthropic.claude-3-haiku-20240307-v1:0"],
        canary_model=canary_model,
        canary_percentage=canary_percentage
    )

    return AIGateway(config).add_provider(BedrockProvider(region_name=region))


def create_openai_gateway(
    primary_model: str = "gpt-4o",
    fallback_models: Optional[List[str]] = None,
    canary_model: Optional[str] = None,
    canary_percentage: int = 0,
    api_key: Optional[str] = None
) -> AIGateway:
    """Create an AI Gateway configured for OpenAI."""
    config = AIGatewayConfig(
        primary_model=primary_model,
        fallback_models=fallback_models or ["gpt-4o-mini"],
        canary_model=canary_model,
        canary_percentage=canary_percentage
    )

    return AIGateway(config).add_provider(OpenAIProvider(api_key=api_key))


def create_multi_provider_gateway(
    primary_model: str,
    fallback_models: List[str],
    canary_model: Optional[str] = None,
    canary_percentage: int = 0,
    bedrock_region: str = "us-east-1",
    openai_api_key: Optional[str] = None
) -> AIGateway:
    """Create an AI Gateway with multiple providers for cross-provider fallback."""
    config = AIGatewayConfig(
        primary_model=primary_model,
        fallback_models=fallback_models,
        canary_model=canary_model,
        canary_percentage=canary_percentage
    )

    gateway = AIGateway(config)
    gateway.add_provider(BedrockProvider(region_name=bedrock_region))

    if openai_api_key:
        gateway.add_provider(OpenAIProvider(api_key=openai_api_key))

    return gateway
