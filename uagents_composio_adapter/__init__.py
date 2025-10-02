"""
uAgents Composio Adapter

A production-ready Composio integration module for LangChain-based AI agents with multi-agent orchestrator built on uAgents framework.

This module provides a comprehensive, async-first integration layer for building
intelligent multi-agent systems that can authenticate users, manage tool access, and
execute actions through the Composio platform. It features an advanced orchestrator
architecture that automatically routes requests to specialized tool agents based on
their capabilities and includes robust error handling, structured logging, connection
management, and memory persistence capabilities.

Key Features:
    - Multi-agent orchestrator system with intelligent request routing
    - Specialized agents for different tool groups and capabilities
    - Persona customization for orchestrator behavior guidance
    - Async/await support throughout the entire system
    - Type-safe configuration management with comprehensive validation
    - Robust error handling with custom exceptions and detailed context
    - Structured logging with contextual information and performance metrics
    - Complete OAuth authentication flows with automatic retry logic
    - Advanced tool retrieval with filtering, modification, and optimization
    - PostgreSQL-based conversation memory with automatic schema management
    - Thread-safe operations with proper resource management and connection pooling

Author: Tejus Gupta <tejus3131@gmail.com>
Version: 1.0.2
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
