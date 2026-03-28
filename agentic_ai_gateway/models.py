"""
Model Definitions and Cross-Provider Mappings
==============================================

This module provides:
1. Model family definitions with default agent types
2. Cross-provider model mappings (same intent across providers)
3. Model resolution utilities (e.g., "latest:haiku" → actual model ID)

Agent Types:
- fast: Speed-optimized, structured extraction (Haiku-class)
- balanced: General purpose, summarization (Sonnet-class)
- code: Tool-calling, agentic workflows (Sonnet 3.7-class)
- reasoning: Complex multi-step reasoning (Opus-class)
- high_throughput: Cross-region profiles for high SLA
- embedding: Text embeddings
- vision: Multimodal image understanding

Author: Tyler Canton
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum


# ============================================================================
# Agent Types
# ============================================================================

class AgentType(str, Enum):
    """
    Agent type classification based on task characteristics.

    Each type maps to model families optimized for that workload.
    """
    FAST = "fast"                    # Speed, structured extraction
    BALANCED = "balanced"            # General purpose
    CODE = "code"                    # Tool-calling, agentic workflows
    REASONING = "reasoning"          # Complex multi-step reasoning
    HIGH_THROUGHPUT = "high_throughput"  # Cross-region for SLA
    EMBEDDING = "embedding"          # Text embeddings
    VISION = "vision"                # Multimodal
    REALTIME = "realtime"            # Voice/realtime APIs
    BATCH = "batch"                  # Batch processing
    COMPLIANCE = "compliance"        # Regulated industries


# ============================================================================
# Provider Definitions
# ============================================================================

class Provider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    MISTRAL = "mistral"
    META = "meta"
    AMAZON = "amazon"
    COHERE = "cohere"


# ============================================================================
# Model Family Definitions
# ============================================================================

@dataclass
class ModelFamily:
    """
    A model family represents a class of models with similar characteristics.

    Example: "haiku" is fast/cheap, "opus" is powerful/expensive.
    """
    name: str                          # e.g., "haiku", "sonnet", "opus"
    provider: Provider                 # Which provider
    default_agent_type: AgentType      # Default classification
    description: str                   # Human-readable description
    characteristics: List[str] = field(default_factory=list)  # Key traits


# Anthropic Model Families
ANTHROPIC_FAMILIES = {
    "haiku": ModelFamily(
        name="haiku",
        provider=Provider.ANTHROPIC,
        default_agent_type=AgentType.FAST,
        description="Fast, cost-effective for structured extraction",
        characteristics=["speed", "low-cost", "structured-output"]
    ),
    "sonnet": ModelFamily(
        name="sonnet",
        provider=Provider.ANTHROPIC,
        default_agent_type=AgentType.BALANCED,
        description="Balanced performance for general tasks",
        characteristics=["balanced", "summarization", "general-purpose"]
    ),
    "sonnet-3.7": ModelFamily(
        name="sonnet-3.7",
        provider=Provider.ANTHROPIC,
        default_agent_type=AgentType.CODE,
        description="Agentic workhorse - optimized for tool-calling",
        characteristics=["tool-calling", "computer-use", "agentic"]
    ),
    "opus": ModelFamily(
        name="opus",
        provider=Provider.ANTHROPIC,
        default_agent_type=AgentType.REASONING,
        description="State of the art reasoning and analysis",
        characteristics=["reasoning", "analysis", "complex-tasks"]
    ),
}

# OpenAI Model Families
OPENAI_FAMILIES = {
    "gpt-4o-mini": ModelFamily(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        default_agent_type=AgentType.FAST,
        description="Fast, cost-effective multimodal",
        characteristics=["speed", "low-cost", "multimodal"]
    ),
    "gpt-4o": ModelFamily(
        name="gpt-4o",
        provider=Provider.OPENAI,
        default_agent_type=AgentType.BALANCED,
        description="Flagship multimodal model",
        characteristics=["balanced", "multimodal", "general-purpose"]
    ),
    "gpt-4-turbo": ModelFamily(
        name="gpt-4-turbo",
        provider=Provider.OPENAI,
        default_agent_type=AgentType.CODE,
        description="Optimized for code and function calling",
        characteristics=["tool-calling", "code", "128k-context"]
    ),
    "o1": ModelFamily(
        name="o1",
        provider=Provider.OPENAI,
        default_agent_type=AgentType.REASONING,
        description="Advanced reasoning model",
        characteristics=["reasoning", "chain-of-thought", "analysis"]
    ),
    "o3": ModelFamily(
        name="o3",
        provider=Provider.OPENAI,
        default_agent_type=AgentType.REASONING,
        description="Latest reasoning model",
        characteristics=["reasoning", "advanced", "state-of-the-art"]
    ),
}

# Google Model Families
GOOGLE_FAMILIES = {
    "gemini-flash": ModelFamily(
        name="gemini-flash",
        provider=Provider.GOOGLE,
        default_agent_type=AgentType.FAST,
        description="Fast, efficient multimodal",
        characteristics=["speed", "low-cost", "multimodal"]
    ),
    "gemini-pro": ModelFamily(
        name="gemini-pro",
        provider=Provider.GOOGLE,
        default_agent_type=AgentType.BALANCED,
        description="Balanced performance model",
        characteristics=["balanced", "multimodal", "general-purpose"]
    ),
    "gemini-ultra": ModelFamily(
        name="gemini-ultra",
        provider=Provider.GOOGLE,
        default_agent_type=AgentType.REASONING,
        description="Most capable Gemini model",
        characteristics=["reasoning", "analysis", "complex-tasks"]
    ),
}

# Mistral Model Families
MISTRAL_FAMILIES = {
    "mistral-7b": ModelFamily(
        name="mistral-7b",
        provider=Provider.MISTRAL,
        default_agent_type=AgentType.FAST,
        description="Fast, efficient base model",
        characteristics=["speed", "low-cost", "efficient"]
    ),
    "mixtral": ModelFamily(
        name="mixtral",
        provider=Provider.MISTRAL,
        default_agent_type=AgentType.BALANCED,
        description="Mixture of experts - balanced",
        characteristics=["balanced", "moe", "general-purpose"]
    ),
    "mistral-large": ModelFamily(
        name="mistral-large",
        provider=Provider.MISTRAL,
        default_agent_type=AgentType.REASONING,
        description="Most capable Mistral model",
        characteristics=["reasoning", "analysis", "multilingual"]
    ),
    "codestral": ModelFamily(
        name="codestral",
        provider=Provider.MISTRAL,
        default_agent_type=AgentType.CODE,
        description="Optimized for code generation",
        characteristics=["code", "tool-calling", "fill-in-middle"]
    ),
}

# Meta Model Families (via Bedrock)
META_FAMILIES = {
    "llama-3-8b": ModelFamily(
        name="llama-3-8b",
        provider=Provider.META,
        default_agent_type=AgentType.FAST,
        description="Fast, efficient Llama 3",
        characteristics=["speed", "open-source", "efficient"]
    ),
    "llama-3-70b": ModelFamily(
        name="llama-3-70b",
        provider=Provider.META,
        default_agent_type=AgentType.BALANCED,
        description="Balanced Llama 3 model",
        characteristics=["balanced", "open-source", "general-purpose"]
    ),
    "llama-3.1-405b": ModelFamily(
        name="llama-3.1-405b",
        provider=Provider.META,
        default_agent_type=AgentType.REASONING,
        description="Most capable Llama model",
        characteristics=["reasoning", "128k-context", "state-of-the-art"]
    ),
}

# All families combined
ALL_FAMILIES: Dict[str, ModelFamily] = {
    **ANTHROPIC_FAMILIES,
    **OPENAI_FAMILIES,
    **GOOGLE_FAMILIES,
    **MISTRAL_FAMILIES,
    **META_FAMILIES,
}


# ============================================================================
# Cross-Provider Model Mappings
# ============================================================================

@dataclass
class ModelMapping:
    """
    Maps an agent type to specific model IDs across providers.

    This enables: create_gateway("fast") → haiku for Bedrock, gpt-4o-mini for OpenAI
    """
    agent_type: AgentType
    models: Dict[Provider, str]  # Provider → model_id
    fallbacks: Dict[Provider, List[str]] = field(default_factory=dict)


# Default model mappings by agent type
# These are the recommended models for each workload type per provider
DEFAULT_MODELS: Dict[AgentType, ModelMapping] = {
    AgentType.FAST: ModelMapping(
        agent_type=AgentType.FAST,
        models={
            Provider.ANTHROPIC: "anthropic.claude-4-haiku-20260115-v1:0",
            Provider.OPENAI: "gpt-4o-mini",
            Provider.GOOGLE: "gemini-1.5-flash",
            Provider.MISTRAL: "mistral.mistral-7b-instruct-v0:2",
            Provider.META: "meta.llama3-8b-instruct-v1:0",
        },
        fallbacks={
            Provider.ANTHROPIC: [
                "anthropic.claude-3-7-sonnet-20250410-v1:0",
                "anthropic.claude-3-5-haiku-20241022-v1:0"
            ],
            Provider.OPENAI: ["gpt-4o"],
            Provider.GOOGLE: ["gemini-1.5-pro"],
            Provider.MISTRAL: ["mistral.mixtral-8x7b-instruct-v0:1"],
            Provider.META: ["meta.llama3-70b-instruct-v1:0"],
        }
    ),
    AgentType.BALANCED: ModelMapping(
        agent_type=AgentType.BALANCED,
        models={
            Provider.ANTHROPIC: "anthropic.claude-4-sonnet-20250812-v1:0",
            Provider.OPENAI: "gpt-4o",
            Provider.GOOGLE: "gemini-1.5-pro",
            Provider.MISTRAL: "mistral.mixtral-8x7b-instruct-v0:1",
            Provider.META: "meta.llama3-70b-instruct-v1:0",
        },
        fallbacks={
            Provider.ANTHROPIC: ["anthropic.claude-4-haiku-20260115-v1:0"],
            Provider.OPENAI: ["gpt-4o-mini"],
            Provider.GOOGLE: ["gemini-1.5-flash"],
            Provider.MISTRAL: ["mistral.mistral-7b-instruct-v0:2"],
            Provider.META: ["meta.llama3-8b-instruct-v1:0"],
        }
    ),
    AgentType.CODE: ModelMapping(
        agent_type=AgentType.CODE,
        models={
            Provider.ANTHROPIC: "anthropic.claude-3-7-sonnet-20250410-v1:0",
            Provider.OPENAI: "gpt-4-turbo",
            Provider.GOOGLE: "gemini-1.5-pro",
            Provider.MISTRAL: "mistral.mistral-large-2402-v1:0",
            Provider.META: "meta.llama3-70b-instruct-v1:0",
        },
        fallbacks={
            Provider.ANTHROPIC: [
                "anthropic.claude-4-sonnet-20250812-v1:0",
                "anthropic.claude-4-haiku-20260115-v1:0"
            ],
            Provider.OPENAI: ["gpt-4o", "gpt-4o-mini"],
            Provider.GOOGLE: ["gemini-1.5-flash"],
            Provider.MISTRAL: ["mistral.mixtral-8x7b-instruct-v0:1"],
            Provider.META: ["meta.llama3-8b-instruct-v1:0"],
        }
    ),
    AgentType.REASONING: ModelMapping(
        agent_type=AgentType.REASONING,
        models={
            Provider.ANTHROPIC: "anthropic.claude-4-opus-20251105-v1:0",
            Provider.OPENAI: "o1",
            Provider.GOOGLE: "gemini-1.5-pro",  # No ultra in API yet
            Provider.MISTRAL: "mistral.mistral-large-2402-v1:0",
            Provider.META: "meta.llama3-1-405b-instruct-v1:0",
        },
        fallbacks={
            Provider.ANTHROPIC: [
                "anthropic.claude-4-sonnet-20250812-v1:0",
                "anthropic.claude-3-7-sonnet-20250410-v1:0"
            ],
            Provider.OPENAI: ["gpt-4o", "gpt-4-turbo"],
            Provider.GOOGLE: ["gemini-1.5-pro", "gemini-1.5-flash"],
            Provider.MISTRAL: ["mistral.mixtral-8x7b-instruct-v0:1"],
            Provider.META: ["meta.llama3-70b-instruct-v1:0"],
        }
    ),
    AgentType.HIGH_THROUGHPUT: ModelMapping(
        agent_type=AgentType.HIGH_THROUGHPUT,
        models={
            Provider.ANTHROPIC: "us.anthropic.claude-4-sonnet-20250812-v1:0",  # Cross-region profile
            Provider.OPENAI: "gpt-4o",
            Provider.GOOGLE: "gemini-1.5-pro",
            Provider.MISTRAL: "mistral.mistral-large-2402-v1:0",
            Provider.META: "meta.llama3-70b-instruct-v1:0",
        },
        fallbacks={
            Provider.ANTHROPIC: [
                "anthropic.claude-4-sonnet-20250812-v1:0",
                "anthropic.claude-4-haiku-20260115-v1:0"
            ],
            Provider.OPENAI: ["gpt-4o-mini"],
            Provider.GOOGLE: ["gemini-1.5-flash"],
            Provider.MISTRAL: ["mistral.mixtral-8x7b-instruct-v0:1"],
            Provider.META: ["meta.llama3-8b-instruct-v1:0"],
        }
    ),
    AgentType.EMBEDDING: ModelMapping(
        agent_type=AgentType.EMBEDDING,
        models={
            Provider.ANTHROPIC: "amazon.titan-embed-text-v2:0",  # Via Bedrock
            Provider.OPENAI: "text-embedding-3-large",
            Provider.GOOGLE: "text-embedding-004",
            Provider.MISTRAL: "mistral-embed",
            Provider.COHERE: "cohere.embed-english-v3",
        },
        fallbacks={
            Provider.ANTHROPIC: ["amazon.titan-embed-text-v1"],
            Provider.OPENAI: ["text-embedding-3-small", "text-embedding-ada-002"],
            Provider.COHERE: ["cohere.embed-multilingual-v3"],
        }
    ),
    AgentType.VISION: ModelMapping(
        agent_type=AgentType.VISION,
        models={
            Provider.ANTHROPIC: "anthropic.claude-4-sonnet-20250812-v1:0",
            Provider.OPENAI: "gpt-4o",
            Provider.GOOGLE: "gemini-1.5-pro",
            Provider.META: "meta.llama3-2-90b-instruct-v1:0",
        },
        fallbacks={
            Provider.ANTHROPIC: ["anthropic.claude-3-7-sonnet-20250410-v1:0"],
            Provider.OPENAI: ["gpt-4o-mini"],
            Provider.GOOGLE: ["gemini-1.5-flash"],
        }
    ),
}


# ============================================================================
# Model Resolution Utilities
# ============================================================================

def get_model_for_type(
    agent_type: AgentType,
    provider: Provider = Provider.ANTHROPIC,
    with_fallbacks: bool = False
) -> str | tuple[str, List[str]]:
    """
    Get the recommended model ID for an agent type and provider.

    Args:
        agent_type: The type of agent (fast, balanced, code, reasoning)
        provider: Which provider to use
        with_fallbacks: If True, returns (primary, fallbacks) tuple

    Returns:
        Model ID string, or (primary, fallbacks) if with_fallbacks=True

    Example:
        >>> get_model_for_type(AgentType.FAST, Provider.ANTHROPIC)
        "anthropic.claude-4-haiku-20260115-v1:0"

        >>> get_model_for_type(AgentType.CODE, Provider.OPENAI, with_fallbacks=True)
        ("gpt-4-turbo", ["gpt-4o", "gpt-4o-mini"])
    """
    mapping = DEFAULT_MODELS.get(agent_type)
    if not mapping:
        raise ValueError(f"Unknown agent type: {agent_type}")

    primary = mapping.models.get(provider)
    if not primary:
        raise ValueError(f"No model mapping for {agent_type} on {provider}")

    if with_fallbacks:
        fallbacks = mapping.fallbacks.get(provider, [])
        return primary, fallbacks

    return primary


def get_agent_type_for_model(model_id: str) -> Optional[AgentType]:
    """
    Infer the agent type from a model ID based on known patterns.

    Args:
        model_id: The full model ID (e.g., "anthropic.claude-4-haiku-20260115-v1:0")

    Returns:
        AgentType if pattern matches, None otherwise

    Example:
        >>> get_agent_type_for_model("anthropic.claude-4-haiku-20260115-v1:0")
        AgentType.FAST

        >>> get_agent_type_for_model("gpt-4o")
        AgentType.BALANCED
    """
    model_lower = model_id.lower()

    # Anthropic patterns
    if "haiku" in model_lower:
        return AgentType.FAST
    if "3-7-sonnet" in model_lower or "3.7" in model_lower:
        return AgentType.CODE
    if "opus" in model_lower:
        return AgentType.REASONING
    if "sonnet" in model_lower:
        return AgentType.BALANCED

    # OpenAI patterns
    if "gpt-4o-mini" in model_lower:
        return AgentType.FAST
    if "o1" in model_lower or "o3" in model_lower:
        return AgentType.REASONING
    if "gpt-4-turbo" in model_lower:
        return AgentType.CODE
    if "gpt-4o" in model_lower or "gpt-4" in model_lower:
        return AgentType.BALANCED

    # Google patterns
    if "flash" in model_lower:
        return AgentType.FAST
    if "ultra" in model_lower:
        return AgentType.REASONING
    if "gemini" in model_lower:
        return AgentType.BALANCED

    # Mistral patterns
    if "7b" in model_lower:
        return AgentType.FAST
    if "codestral" in model_lower:
        return AgentType.CODE
    if "large" in model_lower:
        return AgentType.REASONING
    if "mixtral" in model_lower:
        return AgentType.BALANCED

    # Meta patterns
    if "8b" in model_lower:
        return AgentType.FAST
    if "405b" in model_lower:
        return AgentType.REASONING
    if "70b" in model_lower or "llama" in model_lower:
        return AgentType.BALANCED

    # Embedding patterns
    if "embed" in model_lower or "embedding" in model_lower:
        return AgentType.EMBEDDING

    return None


def resolve_model_alias(
    alias: str,
    provider: Provider = Provider.ANTHROPIC
) -> str:
    """
    Resolve a model alias to an actual model ID.

    Supports aliases like:
    - "latest:haiku" → latest Haiku model
    - "fast" → model for fast agent type
    - "reasoning" → model for reasoning type

    Args:
        alias: Model alias (e.g., "latest:haiku", "fast", "code")
        provider: Which provider to resolve for

    Returns:
        Resolved model ID

    Example:
        >>> resolve_model_alias("latest:haiku", Provider.ANTHROPIC)
        "anthropic.claude-4-haiku-20260115-v1:0"

        >>> resolve_model_alias("fast", Provider.OPENAI)
        "gpt-4o-mini"
    """
    alias_lower = alias.lower()

    # Handle "latest:family" format
    if alias_lower.startswith("latest:"):
        family = alias_lower.split(":")[1]
        return _resolve_latest_for_family(family, provider)

    # Handle agent type aliases
    try:
        agent_type = AgentType(alias_lower)
        return get_model_for_type(agent_type, provider)
    except ValueError:
        pass

    # Handle family name directly
    if alias_lower in ALL_FAMILIES:
        family = ALL_FAMILIES[alias_lower]
        return get_model_for_type(family.default_agent_type, provider)

    # Return as-is if not an alias
    return alias


def _resolve_latest_for_family(family: str, provider: Provider) -> str:
    """Resolve latest model for a family name."""
    family_lower = family.lower()

    # Map family names to agent types
    family_to_type = {
        "haiku": AgentType.FAST,
        "sonnet": AgentType.BALANCED,
        "sonnet-3.7": AgentType.CODE,
        "opus": AgentType.REASONING,
        "gpt-4o-mini": AgentType.FAST,
        "gpt-4o": AgentType.BALANCED,
        "gpt-4-turbo": AgentType.CODE,
        "o1": AgentType.REASONING,
        "gemini-flash": AgentType.FAST,
        "gemini-pro": AgentType.BALANCED,
    }

    agent_type = family_to_type.get(family_lower)
    if agent_type:
        return get_model_for_type(agent_type, provider)

    raise ValueError(f"Unknown model family: {family}")


# ============================================================================
# Bedrock Model Constants (2025-2026)
# ============================================================================

# Current production models as of 2026
BEDROCK_MODELS = {
    # Claude 4 Series (2025-2026)
    "claude-4-opus": "anthropic.claude-4-opus-20251105-v1:0",
    "claude-4-sonnet": "anthropic.claude-4-sonnet-20250812-v1:0",
    "claude-4-haiku": "anthropic.claude-4-haiku-20260115-v1:0",

    # Claude 3.7 (Agentic Workhorse)
    "claude-3.7-sonnet": "anthropic.claude-3-7-sonnet-20250410-v1:0",

    # Cross-Region Inference Profiles (bypass 429 throttling)
    "us-claude-4-sonnet": "us.anthropic.claude-4-sonnet-20250812-v1:0",

    # Legacy (2024)
    "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3.5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
}

# Convenience aliases
CLAUDE_4_OPUS = BEDROCK_MODELS["claude-4-opus"]
CLAUDE_4_SONNET = BEDROCK_MODELS["claude-4-sonnet"]
CLAUDE_4_HAIKU = BEDROCK_MODELS["claude-4-haiku"]
CLAUDE_3_7_SONNET = BEDROCK_MODELS["claude-3.7-sonnet"]
US_CLAUDE_4_SONNET = BEDROCK_MODELS["us-claude-4-sonnet"]
