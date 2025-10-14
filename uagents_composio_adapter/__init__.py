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
Version: 1.0.8
License: MIT
"""

from .core import (  # Exceptions; Main classes; Response models; Modifier function types; Type aliases; Version and metadata
    AfterExecute,
    AuthConfigId,
    AuthenticationError,
    AuthResponse,
    BeforeExecute,
    ComposioConfig,
    ComposioError,
    ComposioService,
    ConfigurationError,
    ConnectionError,
    ConnectionStatus,
    Modifiers,
    PostgresMemoryConfig,
    SchemaModifier,
    SessionId,
    Tool,
    ToolConfig,
    ToolExecuteParams,
    ToolExecutionResponse,
    ToolRetrievalError,
    ToolSlug,
    UserId,
    __author__,
    __license__,
    __version__,
)

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
    # Tool types
    "Tool",
    "ToolExecuteParams",
    "ToolExecutionResponse",
    # Modifier function types
    "SchemaModifier",
    "BeforeExecute",
    "AfterExecute",
    # Version and metadata
    "__version__",
    "__author__",
    "__license__",
]
