# Contributing to uAgents Composio Adapter

Thank you for your interest in contributing to the uAgents Composio Adapter! This guide will help you get started with contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- PostgreSQL (for testing memory features)
- uAgents framework
- Composio API access

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/tejus3131/uagents-composio-adapter.git
   cd uagents-composio-adapter
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,examples]"
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Set up environment variables for testing**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## 🧪 Running Tests

### Basic Test Run
```bash
pytest
```

### With Coverage
```bash
pytest --cov=uagents_composio_adapter --cov-report=html
```

### Specific Test File
```bash
pytest tests/test_core.py
```

### Integration Tests (requires PostgreSQL)
```bash
# Start PostgreSQL locally
docker run --name postgres-test -e POSTGRES_PASSWORD=postgres -d -p 5432:5432 postgres:13

# Run tests with database
PSQL_HOST=localhost PSQL_DATABASE=postgres PSQL_USERNAME=postgres PSQL_PASSWORD=postgres pytest
```

## 🔧 Code Quality

We use several tools to maintain code quality:

### Formatting
```bash
black uagents_composio_adapter/
isort uagents_composio_adapter/
```

### Linting
```bash
ruff check uagents_composio_adapter/
```

### Type Checking
```bash
mypy uagents_composio_adapter/
```

### Run All Quality Checks
```bash
# Format code
black uagents_composio_adapter/ tests/
isort uagents_composio_adapter/ tests/

# Check linting
ruff check uagents_composio_adapter/ tests/

# Type checking
mypy uagents_composio_adapter/

# Run tests
pytest
```

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Example of Good Code Style

```python
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

async def create_tool_config(
    tool_group_name: str,
    auth_config_id: str,
    tools: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> ToolConfig:
    """
    Create a tool configuration with validation.

    Args:
        tool_group_name: Human-readable name for the tool group
        auth_config_id: Authentication configuration ID from Composio
        tools: Optional list of specific tools to include
        limit: Optional limit on number of tools to retrieve

    Returns:
        ToolConfig: Validated tool configuration instance

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    if not tool_group_name.strip():
        raise ValueError("tool_group_name cannot be empty")

    logger.info(f"Creating tool config for group: {tool_group_name}")

    # Implementation here...
```

### Commit Messages

Use conventional commit format:

```
type(scope): description

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect meaning (white-space, formatting)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to build process or auxiliary tools

Examples:
```
feat(core): add support for tool modifiers
fix(auth): handle timeout in authentication flow
docs(readme): add installation instructions
test(core): add tests for ToolConfig validation
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   - Python version
   - Package version
   - Operating system
   - Relevant dependencies versions

2. **Clear description**:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce

3. **Code example**:
   - Minimal reproducible example
   - Error messages and stack traces
   - Configuration details (without sensitive data)

4. **Additional context**:
   - Logs (with sensitive information removed)
   - Screenshots if applicable

### Bug Report Template

```markdown
## Bug Description
A clear and concise description of what the bug is.

## Environment
- Python version:
- Package version:
- OS and version:
- uAgents version:
- Composio version:

## Steps to Reproduce
1.
2.
3.

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Code Example
```python
# Minimal code example that reproduces the issue
```

## Error Messages
```
Paste any error messages or stack traces here
```

## Additional Context
Any other context about the problem.
```

## 💡 Feature Requests

For feature requests, please provide:

1. **Clear description** of the feature
2. **Use case** - why is this needed?
3. **Proposed implementation** (if you have ideas)
4. **Alternatives considered**
5. **Additional context**

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature you'd like to see.

## Use Case
Describe your use case and why this feature would be helpful.

## Proposed Solution
Describe how you think this should work.

## Alternatives Considered
Describe alternative solutions you've considered.

## Additional Context
Any other context, mockups, or examples.
```

## 🔄 Pull Request Process

### Before Submitting

1. **Check existing issues** - Make sure your change isn't already being worked on
2. **Create an issue** - For non-trivial changes, create an issue first to discuss
3. **Branch naming** - Use descriptive branch names: `feature/add-tool-modifiers`, `fix/auth-timeout`

### PR Requirements

1. **Tests**: Include tests for new functionality
2. **Documentation**: Update documentation if needed
3. **Code quality**: Pass all linting and formatting checks
4. **Type hints**: All new code should have proper type annotations
5. **Changelog**: Add entry to changelog for significant changes

### PR Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran and how to run them.

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)
Add screenshots to help explain your changes.
```

## 🧩 Adding New Features

### Architecture Guidelines

1. **Async First**: All new features should use async/await
2. **Type Safety**: Use Pydantic models for configuration and data validation
3. **Error Handling**: Use custom exceptions with proper error context
4. **Logging**: Add structured logging for debugging and monitoring
5. **Thread Safety**: Ensure new features are thread-safe
6. **Resource Management**: Proper cleanup of resources
7. **Multi-Agent Compatibility**: Consider how new features integrate with the orchestrator system

### Multi-Agent Architecture Guidelines

When contributing to the multi-agent orchestrator system:

#### Orchestrator Agent Development
- The orchestrator agent should remain tool-agnostic and focus on routing decisions
- Use clear, descriptive prompts that help the orchestrator understand agent capabilities
- Implement robust error handling for agent routing failures
- Consider performance implications of agent selection algorithms

#### Specialized Agent Development
- Keep specialized agents focused on their specific domain (e.g., GitHub, Email, Calendar)
- Ensure specialized agents have optimized prompts for their tool sets
- Implement proper tool result validation and error handling
- Consider memory and context management for specialized agents

#### Tool Integration
- Group tools logically by functionality and domain
- Provide clear tool descriptions and usage examples
- Implement proper tool modifiers for enhanced functionality
- Consider tool dependencies and authentication requirements

#### Persona and Prompt Engineering
- Design prompts that work well with persona customization
- Ensure prompts are clear, concise, and actionable
- Consider different user personas and use cases
- Test prompts with various LLM models for compatibility

### Example Feature Implementation

```python
from typing import Optional, Dict, Any
import logging
from pydantic import BaseModel
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class NewFeatureConfig(BaseModel):
    """Configuration for new feature."""

    feature_name: str
    enabled: bool = True
    options: Optional[Dict[str, Any]] = None

    class Config:
        frozen = True  # Make immutable

async def implement_new_feature(
    config: NewFeatureConfig
) -> Dict[str, Any]:
    """
    Implement new feature with proper error handling.

    Args:
        config: Feature configuration

    Returns:
        Dict containing feature results

    Raises:
        ConfigurationError: If configuration is invalid
    """
    logger.info(f"Implementing feature: {config.feature_name}")

    try:
        # Implementation here
        result = {"status": "success", "feature": config.feature_name}

        logger.info(f"Feature {config.feature_name} implemented successfully")
        return result

    except Exception as e:
        logger.error(
            f"Failed to implement feature {config.feature_name}",
            extra={"error": str(e)},
            exc_info=True
        )
        raise ConfigurationError(
            f"Feature implementation failed: {e}",
            details={"feature_name": config.feature_name},
            operation="feature_implementation"
        ) from e
```

## 📚 Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings, type hints, comments
2. **API Documentation**: Function and class documentation
3. **User Documentation**: README, examples, tutorials
4. **Developer Documentation**: Contributing guide, architecture docs

### Documentation Standards

- Use Google-style docstrings
- Include examples in docstrings
- Keep documentation up to date with code changes
- Write clear, concise explanations
- Include type information in docstrings

## 🎉 Recognition

Contributors will be recognized in:

- Project README
- Release notes
- Contributor page (if we create one)

## 📞 Getting Help

- **Discord/Slack**: [Link if available]
- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: [maintainer email if applicable]

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to uAgents Composio Adapter! 🚀
