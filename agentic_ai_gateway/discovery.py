"""
Model Discovery and Auto-Configuration
=======================================

This module provides automatic model discovery using cloud provider APIs:
- AWS Bedrock: list_foundation_models, list_inference_profiles
- Future: OpenAI model listing, Google AI model listing

Benefits:
- Always use latest available models
- Automatic fallback chain construction
- Cross-region inference profile discovery
- No hardcoded model IDs in application code

Author: Tyler Canton
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from functools import lru_cache

from .models import (
    AgentType,
    Provider,
    get_agent_type_for_model,
    BEDROCK_MODELS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Discovery Results
# ============================================================================

@dataclass
class DiscoveredModel:
    """A model discovered from a provider API."""
    model_id: str
    provider: Provider
    model_name: str
    agent_type: Optional[AgentType] = None
    is_inference_profile: bool = False
    regions: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    status: str = "ACTIVE"


@dataclass
class DiscoveryResult:
    """Results from model discovery."""
    models: List[DiscoveredModel]
    inference_profiles: List[DiscoveredModel]
    by_agent_type: Dict[AgentType, List[DiscoveredModel]] = field(default_factory=dict)
    latest: Dict[str, str] = field(default_factory=dict)  # family → model_id


# ============================================================================
# Bedrock Discovery
# ============================================================================

class BedrockDiscovery:
    """
    Discover available models from AWS Bedrock.

    Uses:
    - list_foundation_models: Get all available models
    - list_inference_profiles: Get cross-region inference profiles

    Example:
        discovery = BedrockDiscovery(region="us-east-1")
        result = discovery.discover()

        # Get latest Haiku model
        haiku_id = result.latest.get("haiku")

        # Get all fast models
        fast_models = result.by_agent_type.get(AgentType.FAST, [])
    """

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None

    @property
    def client(self):
        """Lazy-load Bedrock client."""
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock", region_name=self.region)
        return self._client

    def discover(
        self,
        include_inference_profiles: bool = True,
        providers: Optional[List[str]] = None
    ) -> DiscoveryResult:
        """
        Discover all available models and inference profiles.

        Args:
            include_inference_profiles: Whether to query inference profiles
            providers: Filter to specific providers (e.g., ["anthropic", "meta"])

        Returns:
            DiscoveryResult with models organized by type
        """
        models = self._discover_foundation_models(providers)
        profiles = []

        if include_inference_profiles:
            profiles = self._discover_inference_profiles()

        # Organize by agent type
        by_type: Dict[AgentType, List[DiscoveredModel]] = {}
        for model in models + profiles:
            if model.agent_type:
                if model.agent_type not in by_type:
                    by_type[model.agent_type] = []
                by_type[model.agent_type].append(model)

        # Find latest for each family
        latest = self._find_latest_models(models + profiles)

        return DiscoveryResult(
            models=models,
            inference_profiles=profiles,
            by_agent_type=by_type,
            latest=latest
        )

    def _discover_foundation_models(
        self,
        providers: Optional[List[str]] = None
    ) -> List[DiscoveredModel]:
        """Query Bedrock for available foundation models."""
        try:
            response = self.client.list_foundation_models()
            models = []

            for summary in response.get("modelSummaries", []):
                model_id = summary.get("modelId", "")
                provider_name = summary.get("providerName", "").lower()

                # Filter by provider if specified
                if providers and provider_name not in [p.lower() for p in providers]:
                    continue

                # Skip inactive models
                if summary.get("modelLifecycle", {}).get("status") != "ACTIVE":
                    continue

                # Determine provider enum
                provider = self._map_provider(provider_name)

                # Infer agent type from model ID
                agent_type = get_agent_type_for_model(model_id)

                models.append(DiscoveredModel(
                    model_id=model_id,
                    provider=provider,
                    model_name=summary.get("modelName", model_id),
                    agent_type=agent_type,
                    is_inference_profile=False,
                    capabilities=summary.get("outputModalities", []),
                    status="ACTIVE"
                ))

            logger.info(f"[Discovery] Found {len(models)} foundation models")
            return models

        except Exception as e:
            logger.warning(f"[Discovery] Failed to list foundation models: {e}")
            return []

    def _discover_inference_profiles(self) -> List[DiscoveredModel]:
        """Query Bedrock for cross-region inference profiles."""
        try:
            response = self.client.list_inference_profiles()
            profiles = []

            for profile in response.get("inferenceProfileSummaries", []):
                profile_id = profile.get("inferenceProfileId", "")
                profile_name = profile.get("inferenceProfileName", "")

                # Extract model references
                model_refs = profile.get("models", [])
                regions = [ref.get("region") for ref in model_refs if ref.get("region")]

                # Infer agent type from profile ID
                agent_type = get_agent_type_for_model(profile_id)

                profiles.append(DiscoveredModel(
                    model_id=profile_id,
                    provider=Provider.ANTHROPIC,  # Most profiles are Anthropic
                    model_name=profile_name,
                    agent_type=agent_type,
                    is_inference_profile=True,
                    regions=regions,
                    status=profile.get("status", "ACTIVE")
                ))

            logger.info(f"[Discovery] Found {len(profiles)} inference profiles")
            return profiles

        except Exception as e:
            logger.warning(f"[Discovery] Failed to list inference profiles: {e}")
            return []

    def _map_provider(self, provider_name: str) -> Provider:
        """Map provider name string to Provider enum."""
        mapping = {
            "anthropic": Provider.ANTHROPIC,
            "openai": Provider.OPENAI,
            "google": Provider.GOOGLE,
            "mistral": Provider.MISTRAL,
            "meta": Provider.META,
            "amazon": Provider.AMAZON,
            "cohere": Provider.COHERE,
        }
        return mapping.get(provider_name.lower(), Provider.AMAZON)

    def _find_latest_models(
        self,
        models: List[DiscoveredModel]
    ) -> Dict[str, str]:
        """
        Find the latest model ID for each family.

        Parses version dates from model IDs to determine recency.
        """
        latest: Dict[str, str] = {}
        family_dates: Dict[str, tuple[str, str]] = {}  # family → (date, model_id)

        for model in models:
            family = self._extract_family(model.model_id)
            if not family:
                continue

            date = self._extract_date(model.model_id)
            if not date:
                continue

            current = family_dates.get(family)
            if not current or date > current[0]:
                family_dates[family] = (date, model.model_id)

        for family, (_, model_id) in family_dates.items():
            latest[family] = model_id

        return latest

    def _extract_family(self, model_id: str) -> Optional[str]:
        """Extract family name from model ID."""
        model_lower = model_id.lower()

        if "haiku" in model_lower:
            return "haiku"
        if "3-7-sonnet" in model_lower or "3.7" in model_lower:
            return "sonnet-3.7"
        if "opus" in model_lower:
            return "opus"
        if "sonnet" in model_lower:
            return "sonnet"
        if "llama3" in model_lower:
            if "8b" in model_lower:
                return "llama-3-8b"
            if "70b" in model_lower:
                return "llama-3-70b"
            if "405b" in model_lower:
                return "llama-3.1-405b"
        if "mixtral" in model_lower:
            return "mixtral"
        if "mistral" in model_lower:
            return "mistral"

        return None

    def _extract_date(self, model_id: str) -> Optional[str]:
        """Extract date from model ID for version comparison."""
        import re

        # Match patterns like 20250410, 20241022, etc.
        match = re.search(r"(\d{8})", model_id)
        if match:
            return match.group(1)

        return None


# ============================================================================
# Cached Discovery
# ============================================================================

@lru_cache(maxsize=1)
def _cached_discovery(region: str) -> DiscoveryResult:
    """Cached discovery to avoid repeated API calls."""
    discovery = BedrockDiscovery(region=region)
    return discovery.discover()


def discover_models(
    region: str = "us-east-1",
    refresh: bool = False
) -> DiscoveryResult:
    """
    Discover available models with caching.

    Args:
        region: AWS region for Bedrock
        refresh: If True, bypass cache and re-query

    Returns:
        DiscoveryResult with discovered models

    Example:
        result = discover_models()
        latest_haiku = result.latest.get("haiku")
        fast_models = result.by_agent_type.get(AgentType.FAST, [])
    """
    if refresh:
        _cached_discovery.cache_clear()
    return _cached_discovery(region)


def get_latest_model(
    family: str,
    region: str = "us-east-1",
    fallback: Optional[str] = None
) -> str:
    """
    Get the latest model ID for a family.

    Args:
        family: Model family (e.g., "haiku", "sonnet", "opus")
        region: AWS region for Bedrock
        fallback: Fallback model ID if discovery fails

    Returns:
        Latest model ID for the family

    Example:
        >>> get_latest_model("haiku")
        "anthropic.claude-4-haiku-20260115-v1:0"

        >>> get_latest_model("opus", fallback="anthropic.claude-4-opus-20251105-v1:0")
        "anthropic.claude-4-opus-20251105-v1:0"
    """
    try:
        result = discover_models(region)
        model_id = result.latest.get(family.lower())
        if model_id:
            return model_id
    except Exception as e:
        logger.warning(f"[Discovery] Failed to get latest {family}: {e}")

    # Use fallback or hardcoded defaults
    if fallback:
        return fallback

    # Last resort: use hardcoded constants
    defaults = {
        "haiku": BEDROCK_MODELS.get("claude-4-haiku", "anthropic.claude-4-haiku-20260115-v1:0"),
        "sonnet": BEDROCK_MODELS.get("claude-4-sonnet", "anthropic.claude-4-sonnet-20250812-v1:0"),
        "sonnet-3.7": BEDROCK_MODELS.get("claude-3.7-sonnet", "anthropic.claude-3-7-sonnet-20250410-v1:0"),
        "opus": BEDROCK_MODELS.get("claude-4-opus", "anthropic.claude-4-opus-20251105-v1:0"),
    }
    return defaults.get(family.lower(), fallback or "")


def get_models_for_type(
    agent_type: AgentType,
    region: str = "us-east-1",
    include_profiles: bool = True
) -> List[str]:
    """
    Get all discovered models suitable for an agent type.

    Args:
        agent_type: The agent type to get models for
        region: AWS region for Bedrock
        include_profiles: Whether to include inference profiles

    Returns:
        List of model IDs suitable for this agent type

    Example:
        >>> get_models_for_type(AgentType.FAST)
        ["anthropic.claude-4-haiku-20260115-v1:0", ...]
    """
    try:
        result = discover_models(region)
        models = result.by_agent_type.get(agent_type, [])

        if include_profiles:
            return [m.model_id for m in models]
        else:
            return [m.model_id for m in models if not m.is_inference_profile]

    except Exception as e:
        logger.warning(f"[Discovery] Failed to get models for {agent_type}: {e}")
        return []


def get_cross_region_profile(
    family: str = "sonnet",
    region: str = "us-east-1"
) -> Optional[str]:
    """
    Get a cross-region inference profile for high-throughput scenarios.

    Cross-region profiles help bypass 429 throttling by distributing
    requests across multiple regions.

    Args:
        family: Model family (e.g., "sonnet")
        region: AWS region for Bedrock

    Returns:
        Inference profile ID, or None if not found

    Example:
        >>> get_cross_region_profile("sonnet")
        "us.anthropic.claude-4-sonnet-20250812-v1:0"
    """
    try:
        result = discover_models(region)
        for profile in result.inference_profiles:
            if family.lower() in profile.model_id.lower():
                return profile.model_id
    except Exception as e:
        logger.warning(f"[Discovery] Failed to get cross-region profile: {e}")

    # Fallback to hardcoded
    return BEDROCK_MODELS.get("us-claude-4-sonnet")
