"""
Agentic AI Gateway
==================

Production-grade LLM routing with automatic fallbacks, canary deployments,
and multi-provider support.

Author: Tyler Canton
GitHub: https://github.com/tyler-canton
PyPI: https://pypi.org/project/agentic-ai-gateway/

Copyright (c) 2026 Tyler Canton. All rights reserved.
Licensed under the MIT License.
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

__version__ = "0.2.0"
__author__ = "Tyler Canton"
__author_email__ = "tylercanton808@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/tyler-canton/agentic-ai-gateway"
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
