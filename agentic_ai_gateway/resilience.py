"""
Resilience Module
=================

Retry with exponential backoff and circuit breaker patterns for robust LLM calls.

Features:
- Exponential backoff with jitter
- Configurable retry policies
- Circuit breaker for failing endpoints
- Retry budget tracking

Author: Tyler Canton
License: MIT
"""

import logging
import random
import time
from typing import Optional, Callable, TypeVar, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Retry Configuration
# ============================================================================

class RetryStrategy(Enum):
    """Retry backoff strategies."""
    CONSTANT = "constant"       # Same delay each retry
    LINEAR = "linear"           # Delay increases linearly
    EXPONENTIAL = "exponential" # Delay doubles each retry


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True  # Add randomness to prevent thundering herd
    jitter_factor: float = 0.25  # ±25% jitter
    
    # Retryable exceptions (empty = retry all)
    retryable_exceptions: List[type] = field(default_factory=list)
    
    # Non-retryable exceptions (always skip)
    non_retryable_exceptions: List[type] = field(default_factory=lambda: [
        ValueError,
        TypeError,
    ])


# ============================================================================
# Retry with Backoff
# ============================================================================

class RetryWithBackoff:
    """
    Retry operations with exponential backoff.
    
    Example:
        retry = RetryWithBackoff(
            max_retries=3,
            base_delay_seconds=1.0,
            strategy=RetryStrategy.EXPONENTIAL
        )
        
        # Use as decorator
        @retry
        def call_api():
            return requests.get("https://api.example.com")
        
        # Or use directly
        result = retry.execute(call_api)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        jitter: bool = True,
        config: Optional[RetryConfig] = None
    ):
        if config:
            self.config = config
        else:
            self.config = RetryConfig(
                max_retries=max_retries,
                base_delay_seconds=base_delay_seconds,
                max_delay_seconds=max_delay_seconds,
                strategy=strategy,
                jitter=jitter
            )
        
        self._total_retries = 0
        self._successful_retries = 0
        self._lock = Lock()
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        if self.config.strategy == RetryStrategy.CONSTANT:
            delay = self.config.base_delay_seconds
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay_seconds * attempt
        else:  # EXPONENTIAL
            delay = self.config.base_delay_seconds * (2 ** (attempt - 1))
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay_seconds)
        
        # Add jitter
        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Minimum 100ms
        
        return delay
    
    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        # Check non-retryable first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # If retryable list is empty, retry all
        if not self.config.retryable_exceptions:
            return True
        
        # Check retryable list
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False
    
    def execute(
        self,
        func: Callable[[], T],
        on_retry: Optional[Callable[[Exception, int, float], None]] = None
    ) -> T:
        """
        Execute function with retry.
        
        Args:
            func: Function to execute
            on_retry: Callback on each retry (exception, attempt, delay)
            
        Returns:
            Result of successful function call
            
        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 2):  # +1 for initial try, +1 for range
            try:
                result = func()
                
                if attempt > 1:
                    with self._lock:
                        self._successful_retries += 1
                    logger.info(f"[Retry] Succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if attempt > self.config.max_retries or not self._should_retry(e):
                    logger.warning(
                        f"[Retry] Not retrying: {type(e).__name__}: {e}"
                    )
                    raise
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                with self._lock:
                    self._total_retries += 1
                
                logger.warning(
                    f"[Retry] Attempt {attempt} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                # Call retry callback
                if on_retry:
                    try:
                        on_retry(e, attempt, delay)
                    except Exception:
                        pass
                
                # Wait before retry
                time.sleep(delay)
        
        raise last_exception
    
    def __call__(self, func: Callable[[], T]) -> Callable[[], T]:
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            return self.execute(lambda: func(*args, **kwargs))
        return wrapper
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        with self._lock:
            return {
                "total_retries": self._total_retries,
                "successful_retries": self._successful_retries,
            }


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout_seconds: float = 30.0  # Time before trying again
    success_threshold: int = 2  # Successes to close circuit


class CircuitBreaker:
    """
    Circuit breaker pattern for failing endpoints.
    
    Prevents overwhelming failing services by "opening" the circuit
    after too many failures. After a timeout, allows test requests
    through to check if service has recovered.
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5)
        
        def call_model():
            if breaker.is_open:
                raise CircuitOpenError("Circuit is open")
            
            try:
                result = api.call()
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        success_threshold: int = 2,
        config: Optional[CircuitBreakerConfig] = None
    ):
        if config:
            self.config = config
        else:
            self.config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
                success_threshold=success_threshold
            )
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_recovery()
            return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    def _check_recovery(self) -> None:
        """Check if circuit should transition to half-open."""
        if self._state != CircuitState.OPEN:
            return
        
        if self._last_failure_time is None:
            return
        
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        if elapsed >= self.config.recovery_timeout_seconds:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            logger.info("[CircuitBreaker] Transitioning to HALF_OPEN")
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("[CircuitBreaker] Circuit CLOSED (recovered)")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                # Immediately open on failure during recovery
                self._state = CircuitState.OPEN
                logger.warning("[CircuitBreaker] Circuit OPEN (recovery failed)")
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"[CircuitBreaker] Circuit OPEN "
                        f"(threshold {self.config.failure_threshold} reached)"
                    )
    
    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info("[CircuitBreaker] Circuit manually reset")
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
            }


class CircuitOpenError(Exception):
    """Raised when circuit is open and rejecting requests."""
    pass


# ============================================================================
# Convenience Functions
# ============================================================================

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    strategy: str = "exponential"
) -> RetryWithBackoff:
    """
    Create a retry decorator with common settings.
    
    Example:
        @with_retry(max_retries=3)
        def call_api():
            return api.call()
    """
    strategy_map = {
        "constant": RetryStrategy.CONSTANT,
        "linear": RetryStrategy.LINEAR,
        "exponential": RetryStrategy.EXPONENTIAL,
    }
    return RetryWithBackoff(
        max_retries=max_retries,
        base_delay_seconds=base_delay,
        strategy=strategy_map.get(strategy, RetryStrategy.EXPONENTIAL)
    )
