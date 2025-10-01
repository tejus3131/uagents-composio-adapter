"""
uAgents Composio Adapter

A production-ready Composio integration module for LangChain-based AI agents built on uAgents framework.

This module provides a comprehensive, async-first integration layer for building
AI agents that can authenticate users, manage tool access, and execute actions
through the Composio platform. It includes robust error handling, structured
logging, connection management, and memory persistence capabilities.

Key Features:
    - Async/await support throughout
    - Type-safe configuration management
    - Comprehensive error handling with custom exceptions
    - Structured logging with contextual information
    - Connection status management and authentication flows
    - Tool retrieval with filtering and modification capabilities
    - PostgreSQL-based memory persistence
    - Thread-safe operations with proper resource management

Author: Tejus Gupta <tejus3131@gmail.com>
Version: 1.0.0
License: MIT
"""

from .core import (  # Exceptions; Main classes; Response models; Modifier function types; Type aliases; Version and metadata
    AfterExecuteModifierFunc,
    AuthConfigId,
    AuthenticationError,
    AuthResponse,
    BeforeExecuteModifierFunc,
    ComposioConfig,
    ComposioError,
    ComposioService,
    ConfigurationError,
    ConnectionError,
    ConnectionStatus,
    Modifiers,
    PostgresMemoryConfig,
    SchemaModifierFunc,
    SessionId,
    ToolConfig,
    ToolRetrievalError,
    ToolSlug,
    UserId,
    __author__,
    __license__,
    __version__,
)

# Re-export modifier decorators from composio
try:
    from composio import after_execute, before_execute, schema_modifier  # type: ignore

    __all__ = [
        # Main classes
        "ComposioService",
        "ComposioConfig",
        "ToolConfig",
        "Modifiers",
        "PostgresMemoryConfig",
        # Response models
        "ConnectionStatus",
        "AuthResponse",
        # Exceptions
        "ComposioError",
        "AuthenticationError",
        "ConnectionError",
        "ConfigurationError",
        "ToolRetrievalError",
        # Type aliases
        "UserId",
        "AuthConfigId",
        "ToolSlug",
        "SessionId",
        # Modifier function types
        "SchemaModifierFunc",
        "BeforeExecuteModifierFunc",
        "AfterExecuteModifierFunc",
        # Modifier decorators
        "schema_modifier",
        "before_execute",
        "after_execute",
        # Version and metadata
        "__version__",
        "__author__",
        "__license__",
    ]
except ImportError:
    # Composio not available - exclude modifier decorators
    __all__ = [
        # Main classes
        "ComposioService",
        "ComposioConfig",
        "ToolConfig",
        "Modifiers",
        "PostgresMemoryConfig",
        # Response models
        "ConnectionStatus",
        "AuthResponse",
        # Exceptions
        "ComposioError",
        "AuthenticationError",
        "ConnectionError",
        "ConfigurationError",
        "ToolRetrievalError",
        # Type aliases
        "UserId",
        "AuthConfigId",
        "ToolSlug",
        "SessionId",
        # Modifier function types
        "SchemaModifierFunc",
        "BeforeExecuteModifierFunc",
        "AfterExecuteModifierFunc",
        # Version and metadata
        "__version__",
        "__author__",
        "__license__",
    ]
