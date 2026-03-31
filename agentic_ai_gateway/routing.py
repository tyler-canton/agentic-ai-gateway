"""
Intent-Based Model Router Module
=======================

Route prompts to optimal models based on content analysis.

Features:
- Keyword-based routing rules
- Intent classification
- Complexity estimation
- Custom routing functions

Author: Tyler Canton
License: MIT
"""

import logging
import re
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Intent Categories
# ============================================================================

class PromptIntent(Enum):
    """Categories of prompt intent."""
    CODE = "code"           # Programming, debugging, code generation
    ANALYSIS = "analysis"   # Data analysis, reasoning, complex tasks
    CREATIVE = "creative"   # Writing, brainstorming, creative tasks
    CHAT = "chat"           # Simple conversation, Q&A
    MATH = "math"           # Mathematical calculations, proofs
    TRANSLATION = "translation"  # Language translation
    SUMMARIZATION = "summarization"  # Summarizing content
    UNKNOWN = "unknown"


class PromptComplexity(Enum):
    """Estimated complexity of prompt."""
    SIMPLE = "simple"       # Quick, straightforward
    MODERATE = "moderate"   # Some thinking required
    COMPLEX = "complex"     # Deep reasoning needed


# ============================================================================
# Routing Rules
# ============================================================================

@dataclass
class RoutingRule:
    """A rule for routing prompts."""
    name: str
    condition: Callable[[str], bool]
    target_model: str
    priority: int = 0  # Higher = checked first
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing analysis."""
    model_id: str
    intent: PromptIntent
    complexity: PromptComplexity
    confidence: float  # 0.0 to 1.0
    rule_matched: Optional[str] = None
    reasoning: Optional[str] = None


# ============================================================================
# Default Keyword Patterns
# ============================================================================

CODE_KEYWORDS = [
    r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bmethod\b",
    r"\bpython\b", r"\bjavascript\b", r"\btypescript\b", r"\bjava\b",
    r"\brust\b", r"\bgo\b", r"\bc\+\+\b", r"\bc#\b",
    r"\bdebug\b", r"\bbug\b", r"\berror\b", r"\bfix\b",
    r"\bapi\b", r"\bsql\b", r"\bquery\b", r"\bdatabase\b",
    r"\bgit\b", r"\bcommit\b", r"\bmerge\b",
    r"\brefactor\b", r"\boptimize\b", r"\bimplement\b",
    r"```", r"\bdef\b", r"\breturn\b", r"\bimport\b",
]

ANALYSIS_KEYWORDS = [
    r"\banalyze\b", r"\banalysis\b", r"\bcompare\b",
    r"\bevaluate\b", r"\bassess\b", r"\breview\b",
    r"\bexplain\b", r"\bwhy\b", r"\bhow\b",
    r"\breason\b", r"\breasoning\b", r"\blogic\b",
    r"\bpros\b", r"\bcons\b", r"\btrade-?off\b",
    r"\bstrategy\b", r"\bplan\b", r"\barchitect\b",
]

CREATIVE_KEYWORDS = [
    r"\bwrite\b", r"\bstory\b", r"\bpoem\b", r"\bessay\b",
    r"\bcreative\b", r"\bimagine\b", r"\bcreate\b",
    r"\bbrainstorm\b", r"\bideas?\b", r"\binvent\b",
    r"\bdesign\b", r"\bcraft\b", r"\bcompose\b",
]

MATH_KEYWORDS = [
    r"\bcalculate\b", r"\bcompute\b", r"\bsolve\b",
    r"\bequation\b", r"\bformula\b", r"\bproof\b",
    r"\bmath\b", r"\balgebra\b", r"\bcalculus\b",
    r"\bderivative\b", r"\bintegral\b", r"\bmatrix\b",
    r"\d+\s*[\+\-\*\/\^]\s*\d+",  # Basic math expressions
]

TRANSLATION_KEYWORDS = [
    r"\btranslate\b", r"\btranslation\b",
    r"\bin\s+(spanish|french|german|chinese|japanese|korean)\b",
    r"\bto\s+(spanish|french|german|chinese|japanese|korean)\b",
]

SUMMARIZATION_KEYWORDS = [
    r"\bsummarize\b", r"\bsummary\b", r"\btldr\b",
    r"\bshorten\b", r"\bcondense\b", r"\bbrief\b",
    r"\bkey\s+points?\b", r"\bmain\s+points?\b",
]


# ============================================================================
# Semantic Router
# ============================================================================

class IntentRouter:
    """
    Route prompts to optimal models based on content.
    
    Example:
        router = IntentRouter(
            default_model="anthropic.claude-4-sonnet",
            model_mapping={
                PromptIntent.CODE: "anthropic.claude-4-sonnet",
                PromptIntent.CHAT: "anthropic.claude-4-haiku",
                PromptIntent.ANALYSIS: "anthropic.claude-4-opus",
            }
        )
        
        # Analyze prompt and get routing decision
        decision = router.route("Write a Python function to sort a list")
        print(f"Model: {decision.model_id}, Intent: {decision.intent}")
    """
    
    def __init__(
        self,
        default_model: str = "anthropic.claude-4-sonnet",
        model_mapping: Optional[Dict[PromptIntent, str]] = None,
        complexity_mapping: Optional[Dict[PromptComplexity, str]] = None,
        custom_rules: Optional[List[RoutingRule]] = None
    ):
        """
        Initialize semantic router.
        
        Args:
            default_model: Model to use when no rules match
            model_mapping: Map intents to models
            complexity_mapping: Map complexity levels to models
            custom_rules: Custom routing rules (checked first)
        """
        self.default_model = default_model
        self.model_mapping = model_mapping or {}
        self.complexity_mapping = complexity_mapping or {}
        self.custom_rules = sorted(
            custom_rules or [],
            key=lambda r: r.priority,
            reverse=True
        )
        
        # Compile regex patterns
        self._code_patterns = [re.compile(p, re.IGNORECASE) for p in CODE_KEYWORDS]
        self._analysis_patterns = [re.compile(p, re.IGNORECASE) for p in ANALYSIS_KEYWORDS]
        self._creative_patterns = [re.compile(p, re.IGNORECASE) for p in CREATIVE_KEYWORDS]
        self._math_patterns = [re.compile(p, re.IGNORECASE) for p in MATH_KEYWORDS]
        self._translation_patterns = [re.compile(p, re.IGNORECASE) for p in TRANSLATION_KEYWORDS]
        self._summarization_patterns = [re.compile(p, re.IGNORECASE) for p in SUMMARIZATION_KEYWORDS]
    
    def _count_matches(self, text: str, patterns: List[re.Pattern]) -> int:
        """Count pattern matches in text."""
        return sum(1 for p in patterns if p.search(text))
    
    def _classify_intent(self, prompt: str) -> Tuple[PromptIntent, float]:
        """Classify prompt intent with confidence score."""
        scores = {
            PromptIntent.CODE: self._count_matches(prompt, self._code_patterns),
            PromptIntent.ANALYSIS: self._count_matches(prompt, self._analysis_patterns),
            PromptIntent.CREATIVE: self._count_matches(prompt, self._creative_patterns),
            PromptIntent.MATH: self._count_matches(prompt, self._math_patterns),
            PromptIntent.TRANSLATION: self._count_matches(prompt, self._translation_patterns),
            PromptIntent.SUMMARIZATION: self._count_matches(prompt, self._summarization_patterns),
        }
        
        # Find highest scoring intent
        max_intent = max(scores.items(), key=lambda x: x[1])
        
        if max_intent[1] == 0:
            return PromptIntent.CHAT, 0.5
        
        # Calculate confidence based on match count and differentiation
        total_matches = sum(scores.values())
        confidence = min(1.0, max_intent[1] / 3)  # 3+ matches = high confidence
        
        # Boost confidence if one category clearly dominates
        if total_matches > 0:
            dominance = max_intent[1] / total_matches
            confidence = (confidence + dominance) / 2
        
        return max_intent[0], confidence
    
    def _estimate_complexity(self, prompt: str) -> PromptComplexity:
        """Estimate prompt complexity."""
        # Length-based heuristics
        word_count = len(prompt.split())
        
        # Check for complexity indicators
        has_multiple_questions = prompt.count("?") > 1
        has_code_blocks = "```" in prompt
        has_numbered_list = bool(re.search(r"\d+\.", prompt))
        has_constraints = any(
            word in prompt.lower() 
            for word in ["must", "should", "requirements", "constraints", "ensure"]
        )
        
        complexity_score = 0
        
        if word_count > 200:
            complexity_score += 2
        elif word_count > 50:
            complexity_score += 1
        
        if has_multiple_questions:
            complexity_score += 1
        if has_code_blocks:
            complexity_score += 1
        if has_numbered_list:
            complexity_score += 1
        if has_constraints:
            complexity_score += 1
        
        if complexity_score >= 3:
            return PromptComplexity.COMPLEX
        elif complexity_score >= 1:
            return PromptComplexity.MODERATE
        else:
            return PromptComplexity.SIMPLE
    
    def add_rule(self, rule: RoutingRule) -> None:
        """Add a custom routing rule."""
        self.custom_rules.append(rule)
        self.custom_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def route(self, prompt: str) -> RoutingDecision:
        """
        Route a prompt to the optimal model.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            RoutingDecision with model and analysis
        """
        # Check custom rules first
        for rule in self.custom_rules:
            try:
                if rule.condition(prompt):
                    logger.info(f"[Router] Matched rule: {rule.name} → {rule.target_model}")
                    return RoutingDecision(
                        model_id=rule.target_model,
                        intent=PromptIntent.UNKNOWN,
                        complexity=self._estimate_complexity(prompt),
                        confidence=1.0,
                        rule_matched=rule.name,
                        reasoning=f"Matched custom rule: {rule.name}"
                    )
            except Exception as e:
                logger.warning(f"[Router] Rule {rule.name} failed: {e}")
        
        # Classify intent
        intent, confidence = self._classify_intent(prompt)
        complexity = self._estimate_complexity(prompt)
        
        # Determine model
        model = self.default_model
        reasoning = f"Default model for {intent.value} intent"
        
        # Check intent mapping
        if intent in self.model_mapping:
            model = self.model_mapping[intent]
            reasoning = f"Intent-based routing: {intent.value}"
        
        # Override with complexity mapping if applicable
        if complexity in self.complexity_mapping and complexity == PromptComplexity.COMPLEX:
            model = self.complexity_mapping[complexity]
            reasoning = f"Complexity override: {complexity.value}"
        
        logger.info(
            f"[Router] {intent.value} ({confidence:.0%} conf), "
            f"{complexity.value} complexity → {model}"
        )
        
        return RoutingDecision(
            model_id=model,
            intent=intent,
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning
        )


# ============================================================================
# Routing Gateway Wrapper
# ============================================================================

class AdaptiveGateway:
    """
    Wrapper that adds semantic routing to any AIGateway.
    
    Example:
        from agentic_ai_gateway import create_bedrock_gateway
        from agentic_ai_gateway.routing import AdaptiveGateway, PromptIntent
        
        gateway = create_bedrock_gateway(...)
        
        routed = AdaptiveGateway(
            gateway,
            model_mapping={
                PromptIntent.CODE: "anthropic.claude-4-sonnet",
                PromptIntent.CHAT: "anthropic.claude-4-haiku",
                PromptIntent.ANALYSIS: "anthropic.claude-4-opus",
            }
        )
        
        # Automatically routes to appropriate model
        response = routed.invoke("Write a Python function to sort a list")
        print(f"Used model: {response.model_used}")  # claude-4-sonnet
    """
    
    def __init__(
        self,
        gateway,
        model_mapping: Optional[Dict[PromptIntent, str]] = None,
        complexity_mapping: Optional[Dict[PromptComplexity, str]] = None,
        custom_rules: Optional[List[RoutingRule]] = None
    ):
        self.gateway = gateway
        self.router = IntentRouter(
            default_model=gateway.config.primary_model,
            model_mapping=model_mapping,
            complexity_mapping=complexity_mapping,
            custom_rules=custom_rules
        )
    
    def invoke(self, prompt: str, **kwargs):
        """Invoke with semantic routing."""
        # Get routing decision
        decision = self.router.route(prompt)
        
        # Call gateway with routed model
        response = self.gateway.invoke(
            prompt,
            force_model=decision.model_id,
            **kwargs
        )
        
        # Add routing metadata
        response.metadata["routing"] = {
            "intent": decision.intent.value,
            "complexity": decision.complexity.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }
        
        return response
    
    def add_rule(self, rule: RoutingRule) -> None:
        """Add a custom routing rule."""
        self.router.add_rule(rule)
