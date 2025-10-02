# uAgents Composio Adapter

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Typed with mypy](https://img.shields.io/badge/typed-mypy-blue.svg)](https://github.com/python/mypy)

A production-ready **Composio integration module** for **uAgents framework** with **LangChain-based AI agents**. This module provides a comprehensive, async-first integration layer for building intelligent multi-agent systems that can authenticate users, manage tool access, and execute actions through the Composio platform with an advanced orchestrator architecture.

## 🚀 Features

- **🔧 Async/Await Support**: Built from the ground up with async/await for optimal performance
- **🛡️ Type-Safe Configuration**: Comprehensive type annotations and Pydantic models for configuration management
- **🚨 Robust Error Handling**: Custom exceptions with detailed error context and structured logging
- **🔐 Authentication Management**: Complete OAuth flow handling with automatic retry logic
- **🛠️ Flexible Tool Configuration**: Support for specific tools, toolkit-based, scoped, and search-based tool retrieval
- **💾 Memory Persistence**: PostgreSQL-based conversation memory with automatic schema management
- **🧵 Thread-Safe Operations**: Proper resource management and connection pooling
- **📊 Health Monitoring**: Built-in health checks and service monitoring capabilities
- **🎛️ Tool Modifiers**: Comprehensive support for schema, before-execute, and after-execute modifiers
- **🎭 Multi-Agent Orchestrator**: Intelligent orchestrator agent that routes requests to specialized tool agents
- **👤 Persona Customization**: Configurable persona prompts to guide orchestrator agent behavior and decision-making
- **🤖 Specialized Agents**: Automatic creation of specialized agents for different tool groups with optimized prompts

## 📦 Installation

### Using pip (recommended)

```bash
pip install uagents-composio-adapter
```

### Using uv (faster)

```bash
uv add uagents-composio-adapter
```

### Development Installation

```bash
git clone https://github.com/tejus3131/uagents-composio-adapter.git
cd uagents-composio-adapter
pip install -e ".[dev,examples]"
```

## 🏁 Quick Start

### 1. Basic Setup

```python
import asyncio
from uagents import Agent
from uagents_composio_adapter import ComposioConfig, ToolConfig, ComposioService

async def main():
    # Configure tools
    tool_config = ToolConfig.from_toolkits(
        tool_group_name="GitHub Tools",
        auth_config_id="your_github_auth_config_id",
        toolkit="GITHUB",
        limit=5
    )

    # Create Composio configuration
    composio_config = ComposioConfig.from_env(tool_configs=[tool_config])

    # Initialize agent
    agent = Agent(name="My Composio Agent", seed="my_seed", port=8001, mailbox=True)

    # Create and configure Composio service
    async with ComposioService(composio_config=composio_config) as service:
        agent.include(service.protocol, publish_manifest=True)
        await agent.run_async()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Environment Variables

Create a `.env` file with the following variables:

```bash
# Composio Configuration
COMPOSIO_API_KEY=your_composio_api_key_here
COMPOSIO_DEFAULT_TIMEOUT=300

# Authentication Configuration IDs (from Composio dashboard)
GITHUB_AUTH_CONFIG_ID=your_github_auth_config_id
LINKEDIN_AUTH_CONFIG_ID=your_linkedin_auth_config_id

# LLM Configuration (ASI1 or OpenAI compatible)
ASI_API_KEY=your_llm_api_key_here
ASI_BASE_URL=https://api.asi1.ai/v1
ASI_MODEL_NAME=asi1-mini

# PostgreSQL Configuration (optional - for memory persistence)
PSQL_HOST=localhost
PSQL_PORT=5432
PSQL_DATABASE=composio_memory
PSQL_USERNAME=your_db_username
PSQL_PASSWORD=your_db_password
```

## 📚 Comprehensive Examples

### Tool Configuration Patterns

```python
from uagents_composio_adapter import ToolConfig

# 1. Specific tools by name
github_tools = ToolConfig.from_tools(
    tool_group_name="GitHub Issue Management",
    auth_config_id="auth_123",
    tools=["GITHUB_CREATE_AN_ISSUE", "GITHUB_LIST_ISSUES"]
)

# 2. All tools from a toolkit with limit
slack_tools = ToolConfig.from_toolkits(
    tool_group_name="Slack Communication",
    auth_config_id="auth_456",
    toolkit="SLACK",
    limit=10
)

# 3. Tools filtered by OAuth scopes
gmail_tools = ToolConfig.from_toolkit_with_scopes(
    tool_group_name="Email Management",
    auth_config_id="auth_789",
    toolkit="GMAIL",
    scopes=["gmail.send", "gmail.compose"],
    limit=5
)

# 4. Semantic search for relevant tools
crm_tools = ToolConfig.from_search(
    tool_group_name="CRM Tools",
    auth_config_id="auth_101",
    search="customer relationship management contacts",
    limit=8
)
```

### Tool Modifiers for Customization

```python
from uagents_composio_adapter import Modifiers, schema_modifier, before_execute, after_execute

# Schema modifier to enhance tool descriptions
@schema_modifier(tools=["GITHUB_CREATE_AN_ISSUE"])
def enhance_github_schema(tool_name: str, toolkit: str, schema):
    schema.description += " [Enhanced with context awareness]"
    return schema

# Before-execute modifier to inject parameters
@before_execute(tools=["SLACK_SEND_MESSAGE"])
def add_slack_context(tool_name: str, toolkit: str, params):
    if 'channel' not in params.arguments:
        params.arguments['channel'] = '#general'
    return params

# After-execute modifier to process results
@after_execute(tools=["GMAIL_SEND_EMAIL"])
def log_email_sent(tool_name: str, toolkit: str, response):
    print(f"Email sent successfully via {tool_name}")
    return response

# Combine modifiers
modifiers = Modifiers.combine(
    schema_functions=[enhance_github_schema],
    before_execute_functions=[add_slack_context],
    after_execute_functions=[log_email_sent]
)

# Apply to tool configuration
config = ToolConfig.from_toolkits(
    tool_group_name="Enhanced Tools",
    auth_config_id="auth_123",
    toolkit="GITHUB",
    modifiers=modifiers
)
```

### Memory Persistence with PostgreSQL

```python
from uagents_composio_adapter import PostgresMemoryConfig, ComposioService

# Configure PostgreSQL memory
memory_config = PostgresMemoryConfig.from_env()

# Or configure manually
memory_config = PostgresMemoryConfig(
    host="localhost",
    port=5432,
    database="agent_memory",
    user="postgres",
    password="your_password"
)

# Use with ComposioService
async with ComposioService(
    composio_config=composio_config,
    memory_config=memory_config
) as service:
    # Service now has persistent conversation memory
    agent.include(service.protocol, publish_manifest=True)
```

### Multi-Agent Orchestrator System

```python
import asyncio
from uagents import Agent
from uagents_composio_adapter import ComposioConfig, ToolConfig, ComposioService

async def main():
    # Multiple tool configurations for different specialized agents
    tool_configs = [
        ToolConfig.from_toolkits(
            tool_group_name="GitHub Management",
            auth_config_id=os.getenv("GITHUB_AUTH_ID"),
            toolkit="GITHUB",
            limit=5
        ),
        ToolConfig.from_toolkits(
            tool_group_name="Email Operations",
            auth_config_id=os.getenv("GMAIL_AUTH_ID"),
            toolkit="GMAIL",
            limit=3
        ),
        ToolConfig.from_toolkits(
            tool_group_name="Calendar Management",
            auth_config_id=os.getenv("CALENDAR_AUTH_ID"),
            toolkit="GOOGLECALENDAR",
            limit=4
        )
    ]

    # Create configuration with persona prompt
    composio_config = ComposioConfig.from_env(
        tool_configs=tool_configs,
        persona_prompt="You are a productivity-focused AI assistant. Prioritize efficiency and provide clear, actionable guidance for task automation."
    )

    # Initialize agent
    agent = Agent(
        name="Multi-Agent Orchestrator",
        seed="orchestrator_seed",
        port=8001,
        mailbox=True
    )

    # Create orchestrator service with specialized agents
    async with ComposioService(composio_config=composio_config) as service:
        # The service automatically creates:
        # 1. GitHub Management Agent - handles repository operations
        # 2. Email Operations Agent - manages email tasks
        # 3. Calendar Management Agent - handles scheduling
        # 4. Orchestrator Agent - routes requests to appropriate specialists

        agent.include(service.protocol, publish_manifest=True)
        logger.info("🚀 Multi-agent orchestrator system started!")
        await agent.run_async()

if __name__ == "__main__":
    asyncio.run(main())
```

### Complete Production Example

```python
import os
import asyncio
import logging
from uagents import Agent
from uagents_composio_adapter import (
    ComposioConfig, ToolConfig, ComposioService,
    PostgresMemoryConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Multiple tool configurations
    tool_configs = [
        ToolConfig.from_tools(
            tool_group_name="GitHub Management",
            auth_config_id=os.getenv("GITHUB_AUTH_ID"),
            tools=["GITHUB_CREATE_AN_ISSUE", "GITHUB_LIST_ISSUES"]
        ),
        ToolConfig.from_toolkits(
            tool_group_name="Slack Communication",
            auth_config_id=os.getenv("SLACK_AUTH_ID"),
            toolkit="SLACK",
            limit=5
        ),
        ToolConfig.from_search(
            tool_group_name="Email Tools",
            auth_config_id=os.getenv("GMAIL_AUTH_ID"),
            search="send email compose",
            limit=3
        )
    ]

    # Create configurations
    composio_config = ComposioConfig.from_env(tool_configs=tool_configs)
    memory_config = PostgresMemoryConfig.from_env()

    # Initialize agent
    agent = Agent(
        name="Production Composio Agent",
        seed="production_agent_seed",
        port=8001,
        mailbox=True,
        publish_agent_details=True
    )

    # Run with proper resource management
    async with ComposioService(
        composio_config=composio_config,
        memory_config=memory_config
    ) as service:
        # Health check
        health = await service.health_check()
        if health['status'] != 'healthy':
            logger.error(f"Service unhealthy: {health}")
            return

        # Integrate with agent
        agent.include(service.protocol, publish_manifest=True)

        logger.info("🚀 Production agent started successfully!")
        agent.run_async()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    uAgents Agent                            │
├─────────────────────────────────────────────────────────────┤
│                 ComposioService                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               Protocol Handler                      │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │           Authentication Flow               │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │          Orchestrator Agent                 │    │    │
│  │  │  ┌─────────────┬─────────────┬───────────┐   │    │    │
│  │  │  │ GitHub      │ Email       │ Calendar  │   │    │    │
│  │  │  │ Agent       │ Agent       │ Agent     │   │    │    │
│  │  │  └─────────────┴─────────────┴───────────┘   │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │           Tool Management                   │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                  ComposioClient                             │
│              (Composio API Interface)                       │
├─────────────────────────────────────────────────────────────┤
│               PostgreSQL Memory                             │
│            (Optional Persistence)                           │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=uagents_composio_adapter --cov-report=html
```

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/tejus3131/uagents-composio-adapter.git
cd uagents-composio-adapter

# Install in development mode
pip install -e ".[dev,examples]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality Tools

```bash
# Format code
black uagents_composio_adapter/
isort uagents_composio_adapter/

# Type checking
mypy uagents_composio_adapter/

# Linting
ruff check uagents_composio_adapter/
```

## 📖 API Reference

### Core Classes

#### `ComposioService`
Main service class for Composio integration with multi-agent orchestrator support.

```python
class ComposioService:
    def __init__(
        self,
        composio_config: ComposioConfig,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        memory_config: PostgresMemoryConfig | None = None,
    ) -> None: ...

    @property
    def protocol(self) -> Protocol: ...

    async def health_check(self) -> dict[str, Any]: ...
    async def cleanup(self) -> None: ...
```

#### `ComposioConfig`
Configuration class with persona prompt support.

```python
class ComposioConfig:
    api_key: str
    persona_prompt: str | None = None
    tool_configs: list[ToolConfig]
    timeout: int = 300

    @classmethod
    def from_env(
        cls,
        tool_configs: list[ToolConfig] | None = None,
        persona_prompt: str | None = None,
    ) -> "ComposioConfig": ...
```

#### `ToolConfig`
Configuration for tool retrieval with multiple patterns.

```python
class ToolConfig:
    @classmethod
    def from_tools(cls, tool_group_name: str, auth_config_id: str, tools: list[str]) -> "ToolConfig": ...

    @classmethod
    def from_toolkits(cls, tool_group_name: str, auth_config_id: str, toolkit: str, limit: int | None = None) -> "ToolConfig": ...

    @classmethod
    def from_search(cls, tool_group_name: str, auth_config_id: str, search: str, toolkit: str | None = None, limit: int | None = None) -> "ToolConfig": ...
```

#### `Modifiers`
Tool modifier management for customization.

```python
class Modifiers:
    @classmethod
    def with_schema(cls, *modifiers: SchemaModifierFunc) -> "Modifiers": ...

    @classmethod
    def with_before_execute(cls, *modifiers: BeforeExecuteModifierFunc) -> "Modifiers": ...

    @classmethod
    def combine(cls, schema_functions: list | None = None, before_execute_functions: list | None = None, after_execute_functions: list | None = None) -> "Modifiers": ...
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Composio](https://composio.dev/) for the excellent tool integration platform
- [uAgents](https://github.com/fetchai/uAgents) for the robust agent framework
- [LangChain](https://langchain.com/) for the powerful LLM integration tools
- The open-source community for continuous inspiration and support

## 📞 Support

- **Documentation**: [GitHub README](https://github.com/tejus3131/uagents-composio-adapter#readme)
- **Issues**: [GitHub Issues](https://github.com/tejus3131/uagents-composio-adapter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tejus3131/uagents-composio-adapter/discussions)

---

**Made with ❤️ by [Tejus Gupta](https://github.com/tejus3131)**
