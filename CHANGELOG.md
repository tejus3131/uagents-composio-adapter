# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2025-10-02

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
