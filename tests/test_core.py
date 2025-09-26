"""
Tests for uagents-composio-adapter package.

This module contains test cases for the uAgents Composio Adapter package.
"""

from typing import Any

import pytest

from uagents_composio_adapter import (
    ComposioError,
    Modifiers,
    PostgresMemoryConfig,
    ToolConfig,
    __version__,
)


def test_version() -> None:
    """Test that version is properly defined."""
    assert __version__ == "1.0.0"


def test_tool_config_from_tools() -> None:
    """Test ToolConfig creation with specific tools."""
    config = ToolConfig.from_tools(
        tool_group_name="Test Tools",
        auth_config_id="test_auth_123",
        tools=["GITHUB_CREATE_AN_ISSUE", "GITHUB_LIST_ISSUES"]
    )

    assert config.tool_group_name == "Test Tools"
    assert config.auth_config_id == "test_auth_123"
    assert config.tools == ["GITHUB_CREATE_AN_ISSUE", "GITHUB_LIST_ISSUES"]
    assert config.toolkit is None
    assert config.search is None


def test_tool_config_from_toolkits() -> None:
    """Test ToolConfig creation with toolkit."""
    config = ToolConfig.from_toolkits(
        tool_group_name="GitHub Tools",
        auth_config_id="test_auth_456",
        toolkit="GITHUB",
        limit=5
    )

    assert config.tool_group_name == "GitHub Tools"
    assert config.auth_config_id == "test_auth_456"
    assert config.toolkit == "GITHUB"
    assert config.limit == 5
    assert config.tools is None
    assert config.search is None


def test_tool_config_from_search() -> None:
    """Test ToolConfig creation with search."""
    config = ToolConfig.from_search(
        tool_group_name="Search Tools",
        auth_config_id="test_auth_789",
        search="email management",
        limit=3
    )

    assert config.tool_group_name == "Search Tools"
    assert config.auth_config_id == "test_auth_789"
    assert config.search == "email management"
    assert config.limit == 3
    assert config.tools is None
    assert config.toolkit is None


def test_tool_config_validation_errors() -> None:
    """Test ToolConfig validation errors."""

    # Empty tool group name
    with pytest.raises(ValueError, match="tool_group_name cannot be empty"):
        ToolConfig.from_tools("", "auth_123", ["TOOL_1"])

    # Empty auth config ID
    with pytest.raises(ValueError, match="auth_config_id cannot be empty"):
        ToolConfig.from_tools("Test", "", ["TOOL_1"])

    # Empty tools list
    with pytest.raises(ValueError, match="tools list cannot be empty"):
        ToolConfig.from_tools("Test", "auth_123", [])

    # Invalid limit
    with pytest.raises(ValueError, match="limit must be an integer between 1 and 100"):
        ToolConfig.from_toolkits("Test", "auth_123", "GITHUB", limit=0)


def test_postgres_memory_config() -> None:
    """Test PostgresMemoryConfig creation."""
    config = PostgresMemoryConfig(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_password"
    )

    assert config.host == "localhost"
    assert config.port == 5432
    assert config.database == "test_db"
    assert config.user == "test_user"
    assert config.password == "test_password"


def test_modifiers_creation() -> None:
    """Test Modifiers creation and validation."""

    # Test empty modifiers (should raise error)
    with pytest.raises(ValueError, match="At least one modifier function list must be provided"):
        Modifiers()

    # Test with schema functions
    def dummy_schema_modifier(tool: str, toolkit: str, schema: Any) -> Any:
        return schema

    modifiers = Modifiers.with_schema(dummy_schema_modifier)
    assert modifiers.schema_functions == [dummy_schema_modifier]
    assert modifiers.before_execute_functions is None
    assert modifiers.after_execute_functions is None


def test_composio_error() -> None:
    """Test ComposioError exception."""
    error = ComposioError(
        "Test error message",
        details={"key": "value"},
        operation="test_operation"
    )

    assert error.message == "Test error message"
    assert error.details == {"key": "value"}
    assert error.operation == "test_operation"
    assert "test_operation: Test error message" in str(error)
