# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.7] - 2025-10-14

### Fixed
- **🔧 Modifier System**: Fixed modifier setup methods and improved type safety
  - Updated modifier type aliases for better clarity:
    - `SchemaModifierFunc` → `SchemaModifier`
    - `BeforeExecuteModifierFunc` → `BeforeExecute`
    - `AfterExecuteModifierFunc` → `AfterExecute`
  - Simplified module imports by removing conditional try-catch blocks for composio decorator imports
  - Enhanced type hints and added new tool-related types: `Tool`, `ToolExecuteParams`, `ToolExecutionResponse`

### Changed
- **📊 Enhanced Logging**: Upgraded logging levels from debug to info for better visibility of important operations:
  - Connection status checks
  - Tool retrieval operations
  - Memory configuration setup
  - Query processing stages
  - Chat acknowledgements and message processing
- **📦 Project Metadata**: Added comprehensive project classifiers to `pyproject.toml`:
  - Development status, intended audience, license
  - Operating system and Python version compatibility
  - Topic classification for better package discovery
- **🧹 Module Organization**: Streamlined `__init__.py` exports with consistent naming and better categorization

### Updated
- **📦 Dependency Updates**: Updated project dependencies via lock file refresh
  - Various dependency version bumps for security and performance improvements

### Technical Details
- Improved protocol type hints with `AgentChatProtocol` for better type safety
- Enhanced `__all__` exports with proper categorization of classes, exceptions, and types
- Better separation of tool-related types from modifier types in public API

## [1.0.6] - 2025-10-09

### Fixed
- **🔒 Dependency Stability**: Pinned `langchain` to exact version `1.0.0a10` for improved stability and reproducible builds
- **🏗️ Import Organization**: Refactored import structure in `core.py` by moving all imports to the top of the file and removing redundant try-catch blocks around imports

### Changed
- **⚙️ CI Improvements**: Enhanced CI pipeline with better dependency management:
  - Added `--pre` flag for installing pre-release dependencies
  - Added `continue-on-error: true` for mypy checks to prevent pipeline failures
- **📦 Dependency Management**: Changed `langchain>=1.0.0a10` to `langchain==1.0.0a10` in `pyproject.toml`

### Technical Details
- `pyproject.toml`: Fixed langchain dependency version constraint
- `uagents_composio_adapter/core.py`: Reorganized imports for better maintainability
- `.github/workflows/ci.yml`: Enhanced CI configuration for better reliability

## [1.0.5] - 2025-10-09

### Added
- Architecture Overview documentation (`ARCHITECTURE.md`) with components, flows, diagrams, limitations, and future improvements
- README updated with a direct link to the new Architecture document

### Changed
- Dependency: bump `uagents` to `>=0.22.10` (updated `pyproject.toml` and lockfile); indirect bump of `uagents-core` to `0.3.11`
- Context propagation: pass `ctx` into specialized agent creation/invocation to align with context-aware logging and tracing

### Technical Details
- `pyproject.toml`: `uagents>=0.22.10`
- `uagents_composio_adapter/core.py`: ensure `ctx=ctx` is forwarded where specialized agents are created/invoked
- `README.md`: add pointer to `ARCHITECTURE.md`

## [1.0.4] - 2025-10-06

### Changed
- **📝 Improved Logging Architecture**: Enhanced logging consistency by requiring `Context` parameter in core methods for better operation tracking and debugging
- **🔧 Context-Aware Operations**: Updated all client methods to accept and use `Context` objects for centralized logging and session management:
  - `create_auth_request()` now requires `ctx: Context` parameter
  - `verify_auth_request()` now requires `ctx: Context` parameter
  - `get_tools()` now requires `ctx: Context` parameter
  - `connection_exists()` now requires `ctx: Context` parameter
  - `initialize_db_pool()` now requires `ctx: Context` parameter
- **🧠 Enhanced Memory Management**: Improved memory trimming algorithm with better token counting and agent prompt consideration:
  - Added `agent_prompt` parameter to memory trimming factory
  - Implemented iterative token trimming with safety margins
  - Added fallback token management for edge cases
- **🎯 Better Error Handling**: All logging operations now use context-specific loggers for improved traceability
- **⚡ Performance Optimizations**: Enhanced memory trimming with multiple iteration approach to ensure token limits are respected

### Updated
- **📦 Dependency Updates**: Updated development dependencies:
  - `ruff` from 0.13.2 to 0.13.3
  - `xxhash` from 3.5.0 to 3.6.1
  - `yarl` from 1.20.1 to 1.21.0

### Technical Details
- All async methods in `ComposioClient` and `ComposioService` now properly integrate with uAgents context system
- Enhanced memory trimming logic with safety buffers and iterative optimization
- Improved logging granularity with context-aware message tracking
- Better separation of concerns between logging and business logic

## [1.0.3] - 2025-10-03

### Changed
- **🔧 Method Renaming**: Renamed `ToolConfig.from_toolkits()` to `ToolConfig.from_toolkit()` for better naming consistency and clarity
- **📝 Documentation Updates**: Updated all examples and documentation to use the new `from_toolkit()` method name
- **🌐 LLM Configuration Flexibility**: Changed environment variable names from ASI-specific to generic LLM configuration:
  - `ASI_API_KEY` → `LLM_API_KEY`
  - `ASI_BASE_URL` → `LLM_BASE_URL`
  - `ASI_MODEL_NAME` → `LLM_MODEL_NAME`
- **🤖 Enhanced Multi-Agent Coordination**: Improved orchestrator and specialized agent prompts for better multi-step task handling:
  - Added critical rules for memory isolation between agents
  - Enhanced data passing between agents via `relevant_data` parameter
  - Improved guidelines for sequential task processing
  - Better error handling instructions
- **📋 Installation Instructions**: Updated README with `--pre` and `--prerelease=allow` flags for pip and uv installations
- **🔄 Optional Parameters**: Made `relevant_data` parameter optional in specialized agent tools with proper type hint (`dict[str, Any] | None = None`)
- **📊 Improved Logging**: Added detailed logging for agent invocations with relevant_data tracking

### Fixed
- **📝 Documentation Clarity**: Removed redundant docstring sections in prompt generation methods
- **🎯 Prompt Formatting**: Improved specialized agent prompt structure with better visibility for relevant_data
- **📦 Build Configuration**: Reorganized `pyproject.toml` dependency groups for better consistency

### Removed
- **🧹 Coverage Files**: Removed `coverage.xml` from repository and added to `.gitignore`
- **📝 Installation Docs**: Removed "Development Installation" section from README (redundant with existing instructions)
- **💬 Verbose Comments**: Cleaned up unnecessary inline comments in core.py

## [1.0.2]- 2025-10-02

### Added
- **🚀 Multi-Agent Orchestrator System**: Intelligent orchestrator agent that automatically routes user requests to specialized tool agents based on their capabilities
- **🎭 Persona Customization**: Configurable persona prompts via `ComposioConfig.persona_prompt` to guide orchestrator agent behavior and decision-making
- **🤖 Specialized Agents**: Automatic creation of specialized agents for different tool groups with optimized prompts and capabilities
- **🧠 Intelligent Request Routing**: Smart analysis of user requests to determine the most appropriate specialized agent for task execution
- **⚡ Enhanced LangChain Integration**: Updated to use latest LangChain architecture with `create_agent` and improved tool handling
- **📝 Advanced Prompt Engineering**: Sophisticated prompt generation for both orchestrator and specialized agents with context-aware descriptions
- **🔄 Tool-to-Agent Conversion**: Seamless conversion of Composio tools into callable specialized agents within the orchestrator framework
- **📊 Enhanced Logging**: Detailed logging for orchestrator decision-making and agent interactions
- **🛠️ Improved Error Handling**: Better error management across the multi-agent system with graceful fallbacks

### Changed
- **🏗️ Architecture Overhaul**: Refactored from single-agent to multi-agent orchestrator architecture for better scalability and specialization
- **📦 Updated Dependencies**:
  - Updated `langchain` to `>=1.0.0a10` for latest agent creation features
  - Updated `langgraph` to `>=0.6.8` for enhanced graph capabilities
  - Updated `langgraph-checkpoint-postgres` to `>=2.0.24` for improved memory management
- **🔧 Enhanced ComposioConfig**: Added `persona_prompt` parameter to `from_env()` method for persona customization
- **📋 Improved Tool Management**: Enhanced tool grouping and organization for better agent specialization
- **💬 Better Chat Protocol**: Optimized message handling and response generation in the chat protocol
- **🎯 Refined Agent Creation**: Streamlined agent creation process with better tool assignment and prompt optimization

### Fixed
- **🐛 Import Error Handling**: Improved error messages for missing dependencies with clear installation instructions
- **🔄 Memory Management**: Fixed potential memory leaks in agent lifecycle management
- **⚡ Performance Optimizations**: Enhanced async operations and reduced latency in agent interactions
- **🔧 Type Safety**: Improved type annotations and validation across the codebase

### Technical Details
- Replaced single `_create_agent` method with `_create_orchestrator_agent` and `_create_specialized_agent_tool` methods
- Introduced `_create_orchestrator_prompt` and `_create_agent_tool_prompt` for sophisticated prompt generation
- Added `_structured_tool_to_dict` utility for tool metadata extraction
- Enhanced error handling with detailed logging and graceful degradation
- Improved code formatting and style consistency across the module

## [1.0.1] - 2025-10-01

### Added
- Added typing support for better type safety and development experience
- Improved token limit configuration for agent calls

### Changed
- Enhanced prompt handling for LangChain agent integration
- Updated agent call token limit configuration

## [1.0.0] - 2025-09-26

### Added
- Initial release of uAgents Composio Adapter
- Async integration between uAgents and Composio platform
- Tool configuration and authentication management
- PostgreSQL memory persistence support
- Comprehensive documentation and examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
