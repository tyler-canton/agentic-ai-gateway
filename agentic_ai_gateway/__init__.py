"""
Agentic AI Gateway - Production-grade LLM routing with fallbacks and canary deployments.
"""

from .gateway import (
    AIGateway as AgenticGateway,
    AIGatewayConfig as AgenticGatewayConfig,
    AIGatewayResponse as AgenticGatewayResponse,
    LLMProvider,
    BedrockProvider,
    OpenAIProvider,
    InMemoryMetrics,
    MetricsCollector,
    create_bedrock_gateway,
    create_openai_gateway,
    create_multi_provider_gateway,
)

__version__ = "0.1.0"
__all__ = [
    "AgenticGateway",
    "AgenticGatewayConfig",
    "AgenticGatewayResponse",
    "LLMProvider",
    "BedrockProvider",
    "OpenAIProvider",
    "InMemoryMetrics",
    "MetricsCollector",
    "create_bedrock_gateway",
    "create_openai_gateway",
    "create_multi_provider_gateway",
]
