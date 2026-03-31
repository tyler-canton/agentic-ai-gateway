"""
Conversation Memory Module
==========================

Stateful multi-turn conversation management with persistent storage.
Works across load-balanced servers and survives restarts.

Features:
- Redis-backed conversation history
- Automatic context window management
- Token-aware truncation
- Session TTL (auto-expire old conversations)
- In-memory option for testing

Author: Tyler Canton
License: MIT
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Message Types
# ============================================================================

class MessageRole(str, Enum):
    """Role of a message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0  # Estimated tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LLM API."""
        return {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
        }

    def to_json(self) -> str:
        """Serialize for storage."""
        data = asdict(self)
        data["role"] = self.role.value if isinstance(self.role, MessageRole) else self.role
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "Message":
        """Deserialize from storage."""
        parsed = json.loads(data)
        parsed["role"] = MessageRole(parsed["role"])
        return cls(**parsed)


@dataclass
class Conversation:
    """A conversation session with message history."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0

    def add_message(self, role: MessageRole, content: str, token_count: int = 0) -> Message:
        """Add a message to the conversation."""
        msg = Message(role=role, content=content, token_count=token_count)
        self.messages.append(msg)
        self.total_tokens += token_count
        self.updated_at = datetime.now().isoformat()
        return msg

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM API."""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend([m.to_dict() for m in self.messages])
        return result

    def to_json(self) -> str:
        """Serialize conversation."""
        data = {
            "session_id": self.session_id,
            "messages": [m.to_json() for m in self.messages],
            "system_prompt": self.system_prompt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "total_tokens": self.total_tokens,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "Conversation":
        """Deserialize conversation."""
        parsed = json.loads(data)
        messages = [Message.from_json(m) for m in parsed.get("messages", [])]
        return cls(
            session_id=parsed["session_id"],
            messages=messages,
            system_prompt=parsed.get("system_prompt"),
            created_at=parsed.get("created_at", datetime.now().isoformat()),
            updated_at=parsed.get("updated_at", datetime.now().isoformat()),
            metadata=parsed.get("metadata", {}),
            total_tokens=parsed.get("total_tokens", 0),
        )


# ============================================================================
# Memory Backend Interface
# ============================================================================

class ConversationMemory(ABC):
    """Abstract base for conversation storage backends."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[Conversation]:
        """Get conversation by session ID."""
        pass

    @abstractmethod
    def save(self, conversation: Conversation) -> None:
        """Save conversation."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete conversation."""
        pass

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if conversation exists."""
        pass


# ============================================================================
# In-Memory Backend (for testing)
# ============================================================================

class InMemoryConversationMemory(ConversationMemory):
    """
    In-memory conversation storage.

    Use for testing or single-instance deployments.
    Data is lost on restart.

    Example:
        memory = InMemoryConversationMemory()
        conv = Conversation(session_id="test123")
        conv.add_message(MessageRole.USER, "Hello!")
        memory.save(conv)
    """

    def __init__(self, max_conversations: int = 1000):
        self._store: Dict[str, Conversation] = {}
        self.max_conversations = max_conversations

    def get(self, session_id: str) -> Optional[Conversation]:
        return self._store.get(session_id)

    def save(self, conversation: Conversation) -> None:
        # Evict oldest if at capacity
        if len(self._store) >= self.max_conversations and conversation.session_id not in self._store:
            oldest = min(self._store.values(), key=lambda c: c.updated_at)
            del self._store[oldest.session_id]

        self._store[conversation.session_id] = conversation

    def delete(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    def exists(self, session_id: str) -> bool:
        return session_id in self._store

    def clear(self) -> int:
        count = len(self._store)
        self._store.clear()
        return count


# ============================================================================
# Redis Backend (for production)
# ============================================================================

class RedisConversationMemory(ConversationMemory):
    """
    Redis-backed conversation storage.

    Production-ready: works across load-balanced servers,
    survives restarts, with automatic session expiry.

    Example:
        import redis
        redis_client = redis.Redis.from_url("redis://localhost:6379")

        memory = RedisConversationMemory(
            redis_client,
            ttl_seconds=86400,  # 24 hours
            key_prefix="conv:"
        )

    Requirements:
        pip install redis
    """

    def __init__(
        self,
        redis_client,
        ttl_seconds: int = 86400,  # 24 hours default
        key_prefix: str = "conversation:"
    ):
        """
        Initialize Redis conversation memory.

        Args:
            redis_client: Redis client instance
            ttl_seconds: Session expiry time (default: 24 hours)
            key_prefix: Redis key namespace
        """
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

    def _key(self, session_id: str) -> str:
        """Generate Redis key."""
        return f"{self.key_prefix}{session_id}"

    def get(self, session_id: str) -> Optional[Conversation]:
        """Get conversation from Redis."""
        try:
            data = self.redis.get(self._key(session_id))
            if data is None:
                return None
            return Conversation.from_json(data.decode("utf-8"))
        except Exception as e:
            logger.error(f"[ConversationMemory] GET error: {e}")
            return None

    def save(self, conversation: Conversation) -> None:
        """Save conversation to Redis with TTL."""
        try:
            key = self._key(conversation.session_id)
            self.redis.set(key, conversation.to_json(), ex=self.ttl_seconds)
            logger.debug(f"[ConversationMemory] Saved: {conversation.session_id}")
        except Exception as e:
            logger.error(f"[ConversationMemory] SAVE error: {e}")

    def delete(self, session_id: str) -> bool:
        """Delete conversation from Redis."""
        try:
            result = self.redis.delete(self._key(session_id))
            return result > 0
        except Exception as e:
            logger.error(f"[ConversationMemory] DELETE error: {e}")
            return False

    def exists(self, session_id: str) -> bool:
        """Check if conversation exists in Redis."""
        try:
            return self.redis.exists(self._key(session_id)) > 0
        except Exception as e:
            logger.error(f"[ConversationMemory] EXISTS error: {e}")
            return False

    def refresh_ttl(self, session_id: str) -> bool:
        """Refresh TTL for active conversation."""
        try:
            return self.redis.expire(self._key(session_id), self.ttl_seconds)
        except Exception as e:
            logger.error(f"[ConversationMemory] REFRESH error: {e}")
            return False

    def get_ttl(self, session_id: str) -> int:
        """Get remaining TTL in seconds."""
        try:
            return self.redis.ttl(self._key(session_id))
        except Exception as e:
            logger.error(f"[ConversationMemory] TTL error: {e}")
            return -1


# ============================================================================
# Token Estimation
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses simple heuristic: ~4 characters per token.
    For production, use tiktoken or model-specific tokenizer.
    """
    return len(text) // 4


# ============================================================================
# Conversation Gateway
# ============================================================================

class ConversationGateway:
    """
    Wrapper that adds conversation memory to any AIGateway.

    Automatically manages:
    - Message history per session
    - Context window limits (auto-truncation)
    - System prompts
    - Token tracking

    Example:
        import redis
        from agentic_ai_gateway import create_bedrock_gateway
        from agentic_ai_gateway.conversation import (
            ConversationGateway,
            RedisConversationMemory
        )

        gateway = create_bedrock_gateway(...)

        redis_client = redis.Redis.from_url("redis://localhost:6379")
        memory = RedisConversationMemory(redis_client)

        conv_gateway = ConversationGateway(
            gateway=gateway,
            memory=memory,
            max_history=20,           # Keep last 20 messages
            max_tokens=8000,          # Context window limit
            system_prompt="You are a helpful assistant."
        )

        # Stateful conversation
        session_id = "user_123"

        r1 = conv_gateway.invoke("My name is Tyler", session_id=session_id)
        r2 = conv_gateway.invoke("What's my name?", session_id=session_id)
        # → "Your name is Tyler."

        # Get conversation history
        history = conv_gateway.get_history(session_id)

        # Clear conversation
        conv_gateway.clear_session(session_id)
    """

    def __init__(
        self,
        gateway,
        memory: Optional[ConversationMemory] = None,
        max_history: int = 50,
        max_tokens: int = 8000,
        system_prompt: Optional[str] = None,
        truncation_strategy: str = "oldest_first"
    ):
        """
        Initialize conversation gateway.

        Args:
            gateway: The underlying AIGateway
            memory: Conversation storage backend (default: in-memory)
            max_history: Maximum messages to keep
            max_tokens: Maximum total tokens before truncation
            system_prompt: Default system prompt for new conversations
            truncation_strategy: How to truncate ("oldest_first" or "summarize")
        """
        self.gateway = gateway
        self.memory = memory or InMemoryConversationMemory()
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.truncation_strategy = truncation_strategy

    def _get_or_create_conversation(self, session_id: str) -> Conversation:
        """Get existing conversation or create new one."""
        conv = self.memory.get(session_id)
        if conv is None:
            conv = Conversation(
                session_id=session_id,
                system_prompt=self.system_prompt
            )
        return conv

    def _truncate_if_needed(self, conversation: Conversation) -> None:
        """Truncate conversation if it exceeds limits."""
        # Truncate by message count
        while len(conversation.messages) > self.max_history:
            removed = conversation.messages.pop(0)
            conversation.total_tokens -= removed.token_count

        # Truncate by token count
        while conversation.total_tokens > self.max_tokens and len(conversation.messages) > 2:
            removed = conversation.messages.pop(0)
            conversation.total_tokens -= removed.token_count

    def invoke(
        self,
        prompt: str,
        session_id: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Invoke with conversation context.

        Args:
            prompt: User message
            session_id: Conversation session ID
            system_prompt: Override system prompt for this call
            **kwargs: Additional parameters for gateway

        Returns:
            Gateway response with conversation metadata
        """
        # Get or create conversation
        conversation = self._get_or_create_conversation(session_id)

        # Override system prompt if provided
        if system_prompt:
            conversation.system_prompt = system_prompt

        # Estimate tokens for user message
        user_tokens = estimate_tokens(prompt)

        # Add user message
        conversation.add_message(
            role=MessageRole.USER,
            content=prompt,
            token_count=user_tokens
        )

        # Truncate if needed
        self._truncate_if_needed(conversation)

        # Build messages for API
        messages = conversation.get_messages_for_api()

        # Call gateway
        response = self.gateway.invoke(
            prompt,
            messages=messages,
            **kwargs
        )

        # Estimate tokens for response
        response_tokens = estimate_tokens(response.content)

        # Add assistant message
        conversation.add_message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            token_count=response_tokens
        )

        # Save conversation
        self.memory.save(conversation)

        # Add conversation metadata to response
        response.metadata["conversation"] = {
            "session_id": session_id,
            "message_count": len(conversation.messages),
            "total_tokens": conversation.total_tokens,
        }

        logger.info(
            f"[Conversation] {session_id}: "
            f"{len(conversation.messages)} messages, "
            f"~{conversation.total_tokens} tokens"
        )

        return response

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history as list of messages."""
        conv = self.memory.get(session_id)
        if conv is None:
            return []
        return [m.to_dict() for m in conv.messages]

    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get full conversation object."""
        return self.memory.get(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history."""
        return self.memory.delete(session_id)

    def set_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Update system prompt for a conversation."""
        conv = self._get_or_create_conversation(session_id)
        conv.system_prompt = system_prompt
        self.memory.save(conv)

    def add_context(self, session_id: str, context: str, role: str = "system") -> None:
        """
        Inject context into conversation.

        Useful for RAG - add retrieved documents as context.
        """
        conv = self._get_or_create_conversation(session_id)
        conv.add_message(
            role=MessageRole(role),
            content=context,
            token_count=estimate_tokens(context)
        )
        self.memory.save(conv)
