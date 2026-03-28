"""
Agentic AI Gateway
==================

Production-grade LLM routing with automatic fallbacks, canary deployments,
and multi-provider support.

v0.4.0 Features:
- Cost tracking with budget alerts
- Request caching to reduce costs
- Retry with exponential backoff
- Semantic routing based on prompt content
- CloudWatch metrics integration
- Circuit breaker pattern

v0.3.0 Features:
- Model family definitions with default agent types
- Cross-provider model mappings (same intent across providers)
- Bedrock API auto-discovery for latest models
- Support for "latest:haiku" style model resolution
- create_gateway_for_type() factory function

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
    create_gateway_for_type,
)

from .models import (
    AgentType,
    Provider,
    ModelFamily,
    ModelMapping,
    get_model_for_type,
    get_agent_type_for_model,
    resolve_model_alias,
    # Bedrock model constants
    CLAUDE_4_OPUS,
    CLAUDE_4_SONNET,
    CLAUDE_4_HAIKU,
    CLAUDE_3_7_SONNET,
    US_CLAUDE_4_SONNET,
    BEDROCK_MODELS,
)

from .discovery import (
    BedrockDiscovery,
    DiscoveredModel,
    DiscoveryResult,
    discover_models,
    get_latest_model,
    get_models_for_type,
    get_cross_region_profile,
)

from .costs import (
    CostTracker,
    CostRecord,
    BudgetAlert,
    BEDROCK_PRICING,
    OPENAI_PRICING,
)

from .caching import (
    RequestCache,
    CacheEntry,
    CachedGateway,
)

from .resilience import (
    RetryWithBackoff,
    RetryConfig,
    RetryStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    with_retry,
)

from .routing import (
    SemanticRouter,
    RoutedGateway,
    RoutingRule,
    RoutingDecision,
    PromptIntent,
    PromptComplexity,
)

from .observability import (
    CloudWatchMetrics,
    MetricPoint,
    generate_dashboard_json,
)

__version__ = "0.4.0"
__author__ = "Tyler Canton"
__author_email__ = "tylercanton808@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/tyler-canton/agentic-ai-gateway"
__all__ = [
    # Core Gateway
    "AgenticGateway",
    "AgenticGatewayConfig",
    "AgenticGatewayResponse",
    "LLMProvider",
    "BedrockProvider",
    "OpenAIProvider",
    "InMemoryMetrics",
    "MetricsCollector",

    # Factory Functions
    "create_bedrock_gateway",
    "create_openai_gateway",
    "create_multi_provider_gateway",
    "create_gateway_for_type",

    # Model Types & Mappings
    "AgentType",
    "Provider",
    "ModelFamily",
    "ModelMapping",
    "get_model_for_type",
    "get_agent_type_for_model",
    "resolve_model_alias",

    # Model Constants
    "CLAUDE_4_OPUS",
    "CLAUDE_4_SONNET",
    "CLAUDE_4_HAIKU",
    "CLAUDE_3_7_SONNET",
    "US_CLAUDE_4_SONNET",
    "BEDROCK_MODELS",

    # Discovery
    "BedrockDiscovery",
    "DiscoveredModel",
    "DiscoveryResult",
    "discover_models",
    "get_latest_model",
    "get_models_for_type",
    "get_cross_region_profile",

    # Cost Tracking (v0.4.0)
    "CostTracker",
    "CostRecord",
    "BudgetAlert",
    "BEDROCK_PRICING",
    "OPENAI_PRICING",

    # Caching (v0.4.0)
    "RequestCache",
    "CacheEntry",
    "CachedGateway",

    # Resilience (v0.4.0)
    "RetryWithBackoff",
    "RetryConfig",
    "RetryStrategy",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    "with_retry",

    # Semantic Routing (v0.4.0)
    "SemanticRouter",
    "RoutedGateway",
    "RoutingRule",
    "RoutingDecision",
    "PromptIntent",
    "PromptComplexity",

    # Observability (v0.4.0)
    "CloudWatchMetrics",
    "MetricPoint",
    "generate_dashboard_json",
]
