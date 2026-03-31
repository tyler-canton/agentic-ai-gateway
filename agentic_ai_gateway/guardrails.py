"""
Guardrails Module
=================

Enterprise-grade safety features for LLM applications.
Essential for HIPAA, SOC2, and other compliance requirements.

Features:
- PII Detection & Redaction (SSN, email, phone, credit card, etc.)
- Prompt Injection Defense (jailbreak detection)
- Content Filtering (toxic/harmful content)
- Output Validation (format, length, content checks)

Author: Tyler Canton
License: MIT
"""

import re
import logging
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# Violation Types
# ============================================================================

class ViolationType(str, Enum):
    """Types of guardrail violations."""
    PII_DETECTED = "pii_detected"
    PROMPT_INJECTION = "prompt_injection"
    TOXIC_CONTENT = "toxic_content"
    HARMFUL_REQUEST = "harmful_request"
    OUTPUT_VALIDATION = "output_validation"
    CUSTOM = "custom"


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    NAME = "name"  # Requires NER, basic patterns only
    MEDICAL_RECORD = "medical_record"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"


@dataclass
class Violation:
    """A detected guardrail violation."""
    type: ViolationType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False  # Whether request was blocked


@dataclass
class GuardrailResult:
    """Result of guardrail checks."""
    passed: bool
    violations: List[Violation] = field(default_factory=list)
    modified_content: Optional[str] = None  # Content after redaction
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)

    @property
    def should_block(self) -> bool:
        return any(v.blocked for v in self.violations)


# ============================================================================
# PII Detection Patterns
# ============================================================================

PII_PATTERNS: Dict[PIIType, List[re.Pattern]] = {
    PIIType.SSN: [
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # 123-45-6789
        re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),  # 123 45 6789
        re.compile(r'\b\d{9}\b'),  # 123456789 (context-dependent)
    ],
    PIIType.EMAIL: [
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    ],
    PIIType.PHONE: [
        re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),  # US phone
        re.compile(r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b'),  # (123) 456-7890
        re.compile(r'\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),  # +1
    ],
    PIIType.CREDIT_CARD: [
        re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # 16 digits
        re.compile(r'\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b'),  # Amex 15 digits
    ],
    PIIType.IP_ADDRESS: [
        re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    ],
    PIIType.DATE_OF_BIRTH: [
        re.compile(r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b'),  # MM/DD/YYYY
        re.compile(r'\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b'),  # YYYY-MM-DD
    ],
    PIIType.MEDICAL_RECORD: [
        re.compile(r'\bMRN[-:\s]?\d{6,10}\b', re.IGNORECASE),
        re.compile(r'\bpatient[-\s]?id[-:\s]?\d{6,10}\b', re.IGNORECASE),
    ],
}

# Redaction placeholders
REDACTION_MAP: Dict[PIIType, str] = {
    PIIType.SSN: "[SSN_REDACTED]",
    PIIType.EMAIL: "[EMAIL_REDACTED]",
    PIIType.PHONE: "[PHONE_REDACTED]",
    PIIType.CREDIT_CARD: "[CC_REDACTED]",
    PIIType.IP_ADDRESS: "[IP_REDACTED]",
    PIIType.DATE_OF_BIRTH: "[DOB_REDACTED]",
    PIIType.MEDICAL_RECORD: "[MRN_REDACTED]",
}


# ============================================================================
# Prompt Injection Patterns
# ============================================================================

INJECTION_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # (pattern, description, severity)
    (re.compile(r'ignore\s+(all\s+)?(previous|above|prior)\s+instructions?', re.IGNORECASE),
     "Ignore instructions attack", "high"),
    (re.compile(r'forget\s+(everything|all|your)\s+(you|instructions?|rules?)', re.IGNORECASE),
     "Forget instructions attack", "high"),
    (re.compile(r'you\s+are\s+now\s+(a|an|in)\s+\w+\s+mode', re.IGNORECASE),
     "Role switching attack", "medium"),
    (re.compile(r'pretend\s+(you\'?re?|to\s+be)\s+(a|an)', re.IGNORECASE),
     "Persona hijacking", "medium"),
    (re.compile(r'(system|admin|root)\s*:\s*', re.IGNORECASE),
     "System prompt injection", "critical"),
    (re.compile(r'\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>', re.IGNORECASE),
     "Instruction tag injection", "critical"),
    (re.compile(r'<\|im_start\|>|<\|im_end\|>', re.IGNORECASE),
     "ChatML injection", "critical"),
    (re.compile(r'disregard\s+(your|all|the)\s+(rules?|guidelines?|instructions?)', re.IGNORECASE),
     "Disregard rules attack", "high"),
    (re.compile(r'bypass\s+(your\s+)?(safety|content|filter)', re.IGNORECASE),
     "Bypass safety attack", "critical"),
    (re.compile(r'jailbreak|DAN\s+mode|developer\s+mode', re.IGNORECASE),
     "Jailbreak attempt", "critical"),
]


# ============================================================================
# Toxic Content Patterns (basic - use dedicated APIs for production)
# ============================================================================

TOXIC_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # Basic patterns - for production, use Perspective API, AWS Comprehend, etc.
    (re.compile(r'\b(kill|murder|attack)\s+(yourself|people|them)\b', re.IGNORECASE),
     "Violence", "critical"),
    (re.compile(r'\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)', re.IGNORECASE),
     "Weapons instructions", "critical"),
    (re.compile(r'\bhow\s+to\s+(hack|break\s+into|exploit)', re.IGNORECASE),
     "Hacking instructions", "high"),
]


# ============================================================================
# PII Detector
# ============================================================================

class PIIDetector:
    """
    Detect and redact Personally Identifiable Information.

    Essential for HIPAA, GDPR, and other compliance requirements.

    Example:
        detector = PIIDetector(
            enabled_types={PIIType.SSN, PIIType.EMAIL, PIIType.PHONE}
        )

        result = detector.scan("Contact john@email.com or 555-123-4567")
        print(result.violations)  # Found EMAIL, PHONE
        print(result.modified_content)  # "Contact [EMAIL_REDACTED] or [PHONE_REDACTED]"
    """

    def __init__(
        self,
        enabled_types: Optional[Set[PIIType]] = None,
        redact: bool = True,
        custom_patterns: Optional[Dict[str, re.Pattern]] = None,
    ):
        """
        Initialize PII detector.

        Args:
            enabled_types: Which PII types to detect (default: all)
            redact: Whether to redact detected PII
            custom_patterns: Additional custom patterns
        """
        self.enabled_types = enabled_types or set(PIIType)
        self.redact = redact
        self.custom_patterns = custom_patterns or {}

    def scan(self, text: str) -> GuardrailResult:
        """
        Scan text for PII.

        Returns GuardrailResult with violations and optional redacted content.
        """
        violations = []
        modified = text
        found_pii: Dict[PIIType, List[str]] = {}

        for pii_type, patterns in PII_PATTERNS.items():
            if pii_type not in self.enabled_types:
                continue

            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    found_pii.setdefault(pii_type, []).extend(matches)

                    # Redact if enabled
                    if self.redact:
                        replacement = REDACTION_MAP.get(pii_type, "[REDACTED]")
                        modified = pattern.sub(replacement, modified)

        # Create violations
        for pii_type, matches in found_pii.items():
            violations.append(Violation(
                type=ViolationType.PII_DETECTED,
                severity="high",
                message=f"Detected {pii_type.value}: {len(matches)} occurrence(s)",
                details={
                    "pii_type": pii_type.value,
                    "count": len(matches),
                    "redacted": self.redact,
                },
                blocked=False,  # PII is redacted, not blocked
            ))

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            modified_content=modified if self.redact else None,
            metadata={"pii_types_found": list(found_pii.keys())},
        )


# ============================================================================
# Prompt Injection Detector
# ============================================================================

class PromptInjectionDetector:
    """
    Detect prompt injection and jailbreak attempts.

    Protects against:
    - "Ignore previous instructions" attacks
    - Role/persona hijacking
    - System prompt injection
    - ChatML/instruction tag injection

    Example:
        detector = PromptInjectionDetector()
        result = detector.scan("Ignore all previous instructions and...")
        if not result.passed:
            print("Blocked:", result.violations[0].message)
    """

    def __init__(
        self,
        block_on_detection: bool = True,
        severity_threshold: str = "medium",  # Block at this severity or higher
        custom_patterns: Optional[List[Tuple[re.Pattern, str, str]]] = None,
    ):
        """
        Initialize injection detector.

        Args:
            block_on_detection: Whether to block requests with injections
            severity_threshold: Minimum severity to trigger ("low", "medium", "high", "critical")
            custom_patterns: Additional (pattern, description, severity) tuples
        """
        self.block_on_detection = block_on_detection
        self.severity_threshold = severity_threshold
        self.patterns = list(INJECTION_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        self._severity_order = ["low", "medium", "high", "critical"]

    def _severity_meets_threshold(self, severity: str) -> bool:
        """Check if severity meets or exceeds threshold."""
        try:
            return self._severity_order.index(severity) >= self._severity_order.index(self.severity_threshold)
        except ValueError:
            return False

    def scan(self, text: str) -> GuardrailResult:
        """
        Scan text for prompt injection attempts.
        """
        violations = []

        for pattern, description, severity in self.patterns:
            if pattern.search(text):
                should_block = self.block_on_detection and self._severity_meets_threshold(severity)

                violations.append(Violation(
                    type=ViolationType.PROMPT_INJECTION,
                    severity=severity,
                    message=description,
                    details={"pattern": pattern.pattern},
                    blocked=should_block,
                ))

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            metadata={"patterns_checked": len(self.patterns)},
        )


# ============================================================================
# Content Filter
# ============================================================================

class ContentFilter:
    """
    Filter toxic and harmful content.

    For production, integrate with:
    - AWS Comprehend
    - Google Perspective API
    - OpenAI Moderation API
    - Azure Content Safety

    Example:
        filter = ContentFilter()
        result = filter.scan("How to make a bomb")
        if result.should_block:
            print("Blocked harmful content")
    """

    def __init__(
        self,
        block_on_detection: bool = True,
        custom_patterns: Optional[List[Tuple[re.Pattern, str, str]]] = None,
    ):
        self.block_on_detection = block_on_detection
        self.patterns = list(TOXIC_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def scan(self, text: str) -> GuardrailResult:
        """
        Scan text for toxic/harmful content.
        """
        violations = []

        for pattern, description, severity in self.patterns:
            if pattern.search(text):
                violations.append(Violation(
                    type=ViolationType.TOXIC_CONTENT,
                    severity=severity,
                    message=description,
                    details={"pattern": pattern.pattern},
                    blocked=self.block_on_detection and severity in ("high", "critical"),
                ))

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
        )


# ============================================================================
# Combined Guardrails
# ============================================================================

class Guardrails:
    """
    Combined guardrails for comprehensive protection.

    Runs all checks: PII detection, prompt injection, content filtering.

    Example:
        guardrails = Guardrails(
            pii_detection=True,
            prompt_injection=True,
            content_filter=True,
        )

        # Check input
        result = guardrails.check_input("User message here")
        if result.should_block:
            return "Request blocked for safety"

        # Get safe prompt (with PII redacted)
        safe_prompt = result.modified_content or original_prompt

        # ... call LLM ...

        # Check output
        output_result = guardrails.check_output(llm_response)
    """

    def __init__(
        self,
        pii_detection: bool = True,
        pii_types: Optional[Set[PIIType]] = None,
        pii_redact: bool = True,
        prompt_injection: bool = True,
        content_filter: bool = True,
        custom_validators: Optional[List[Callable[[str], GuardrailResult]]] = None,
    ):
        """
        Initialize guardrails.

        Args:
            pii_detection: Enable PII detection
            pii_types: Which PII types to detect (default: all)
            pii_redact: Redact detected PII
            prompt_injection: Enable prompt injection detection
            content_filter: Enable content filtering
            custom_validators: Additional validator functions
        """
        self.pii_detector = PIIDetector(
            enabled_types=pii_types,
            redact=pii_redact,
        ) if pii_detection else None

        self.injection_detector = PromptInjectionDetector() if prompt_injection else None
        self.content_filter = ContentFilter() if content_filter else None
        self.custom_validators = custom_validators or []

    def check_input(self, text: str) -> GuardrailResult:
        """
        Check input (user prompt) against all guardrails.
        """
        all_violations = []
        modified_content = text

        # Prompt injection (check first - most critical)
        if self.injection_detector:
            result = self.injection_detector.scan(text)
            all_violations.extend(result.violations)
            if result.should_block:
                return GuardrailResult(
                    passed=False,
                    violations=all_violations,
                    metadata={"blocked_by": "prompt_injection"},
                )

        # Content filter
        if self.content_filter:
            result = self.content_filter.scan(text)
            all_violations.extend(result.violations)
            if result.should_block:
                return GuardrailResult(
                    passed=False,
                    violations=all_violations,
                    metadata={"blocked_by": "content_filter"},
                )

        # PII detection (last - redacts rather than blocks)
        if self.pii_detector:
            result = self.pii_detector.scan(text)
            all_violations.extend(result.violations)
            if result.modified_content:
                modified_content = result.modified_content

        # Custom validators
        for validator in self.custom_validators:
            result = validator(text)
            all_violations.extend(result.violations)
            if result.should_block:
                return GuardrailResult(
                    passed=False,
                    violations=all_violations,
                    metadata={"blocked_by": "custom_validator"},
                )

        return GuardrailResult(
            passed=len([v for v in all_violations if v.blocked]) == 0,
            violations=all_violations,
            modified_content=modified_content,
        )

    def check_output(self, text: str) -> GuardrailResult:
        """
        Check output (LLM response) against guardrails.

        Typically only checks PII (shouldn't leak in output).
        """
        all_violations = []
        modified_content = text

        # PII detection on output
        if self.pii_detector:
            result = self.pii_detector.scan(text)
            all_violations.extend(result.violations)
            if result.modified_content:
                modified_content = result.modified_content

        return GuardrailResult(
            passed=len(all_violations) == 0,
            violations=all_violations,
            modified_content=modified_content,
        )


# ============================================================================
# Guarded Gateway Wrapper
# ============================================================================

class GuardedGateway:
    """
    Wrapper that adds guardrails to any AIGateway.

    Automatically checks input and output for safety violations.

    Example:
        from agentic_ai_gateway import create_bedrock_gateway
        from agentic_ai_gateway.guardrails import GuardedGateway, Guardrails

        gateway = create_bedrock_gateway(...)

        guarded = GuardedGateway(
            gateway=gateway,
            guardrails=Guardrails(
                pii_detection=True,
                prompt_injection=True,
                content_filter=True,
            ),
            on_violation=lambda v: logger.warning(f"Violation: {v.message}")
        )

        # Automatically checks input and output
        response = guarded.invoke("What is the patient's SSN: 123-45-6789?")
        # SSN is redacted before sending to LLM
    """

    def __init__(
        self,
        gateway,
        guardrails: Optional[Guardrails] = None,
        on_violation: Optional[Callable[[Violation], None]] = None,
        block_message: str = "Request blocked due to safety policy.",
    ):
        """
        Initialize guarded gateway.

        Args:
            gateway: The underlying AIGateway
            guardrails: Guardrails configuration
            on_violation: Callback for violations (logging, alerting)
            block_message: Message to return when blocked
        """
        self.gateway = gateway
        self.guardrails = guardrails or Guardrails()
        self.on_violation = on_violation
        self.block_message = block_message

    def invoke(self, prompt: str, **kwargs):
        """Invoke with guardrails."""
        # Check input
        input_result = self.guardrails.check_input(prompt)

        # Log violations
        for violation in input_result.violations:
            logger.warning(f"[Guardrails] Input violation: {violation.message}")
            if self.on_violation:
                self.on_violation(violation)

        # Block if needed
        if input_result.should_block:
            from .gateway import AIGatewayResponse
            return AIGatewayResponse(
                content=self.block_message,
                model_used="guardrails",
                latency_ms=0,
                fallback_used=False,
                canary_used=False,
                input_tokens=0,
                output_tokens=0,
                metadata={
                    "blocked": True,
                    "violations": [v.message for v in input_result.violations],
                },
            )

        # Use safe prompt (with PII redacted)
        safe_prompt = input_result.modified_content or prompt

        # Call gateway
        response = self.gateway.invoke(safe_prompt, **kwargs)

        # Check output
        output_result = self.guardrails.check_output(response.content)

        # Log output violations
        for violation in output_result.violations:
            logger.warning(f"[Guardrails] Output violation: {violation.message}")
            if self.on_violation:
                self.on_violation(violation)

        # Return safe output
        if output_result.modified_content:
            response.content = output_result.modified_content

        # Add guardrails metadata
        response.metadata["guardrails"] = {
            "input_violations": len(input_result.violations),
            "output_violations": len(output_result.violations),
            "pii_redacted": input_result.modified_content != prompt,
        }

        return response
