# Multi-Agent Channel - Enhanced Agentic Workflow Architecture

🚀 A powerful multi-agent system built with Pydantic AI that enables seamless orchestration of specialized AI agents with support for 100+ models through OpenRouter and direct provider integrations.

## 🌟 Key Features

- **🤖 Multi-Model Support**: Use any AI model from OpenRouter, OpenAI, Anthropic, Google, and more
- **🔀 Specialized Agents**: Code generation, research, analysis, refinement, and custom agents
- **🧠 Knowledge Management**: Vector storage, semantic search, and validation systems
- **⚡ Real-time Progress**: Live streaming updates and progress visualization
- **🛠️ Extensive Tool Library**: 25+ built-in tools with template system
- **🎮 Multiple Interfaces**: CLI, Web UI (Streamlit), and API
- **🐳 Production Ready**: Docker support with monitoring and scaling

## 📚 Architecture Overview

```
Multi-Agent Channel/
├── src/
│   ├── agents/          # Specialized Pydantic AI agents
│   ├── core/            # Core infrastructure (models, orchestration)
│   ├── tools/           # Tool library with 25+ tools
│   ├── knowledge/       # Vector store and knowledge management
│   ├── progress/        # Progress tracking and visualization
│   ├── ui/              # Streamlit UI components
│   ├── api/             # FastAPI backend
│   ├── cli/             # Command-line interface
│   └── monitoring/      # Metrics and observability
├── docker/              # Docker configuration
├── tests/               # Comprehensive test suite
└── examples/            # Usage examples and demos
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vedantparmar12/Multi-agent-channel.git
cd Multi-agent-channel

# Install dependencies
pip install -r requirements.txt

# Create your configuration
cp config.yaml.example config.yaml
# Add your OpenRouter API key to config.yaml
```

### Basic Usage

```python
from src.agents.base_agent import BaseAgent
from src.core.model_provider import ModelConfig

# Configure with your API keys
config = ModelConfig(
    openrouter_api_key="your-openrouter-key",
    openai_api_key="your-openai-key",
    anthropic_api_key="your-anthropic-key"
)

# Create an agent with any model
agent = BaseAgent(
    model="anthropic/claude-3.5-sonnet",  # or any model from 100+ options
    model_config=config
)

# Run the agent
result = await agent.run("Your prompt here", deps)
```

## 🤖 Available Agents

### 1. Code Generator Agent
Generates high-quality code with documentation, tests, and security checks.

```python
from src.agents.code_generator import CodeGeneratorAgent, CodeGeneratorDeps

agent = CodeGeneratorAgent(model="openai/gpt-4-turbo")
deps = CodeGeneratorDeps(
    language="python",
    include_tests=True,
    include_docs=True,
    security_check=True
)
result = await agent.run("Create a REST API", deps)
```

### 2. Research Agent
Conducts comprehensive research with source tracking and analysis.

```python
from src.agents.research_agent import ResearchAgent, ResearchDeps

agent = ResearchAgent(model="anthropic/claude-3-opus")
deps = ResearchDeps(
    topic="quantum computing",
    depth="comprehensive",
    include_sources=True
)
result = await agent.run("Research latest developments", deps)
```

### 3. Analysis Agent
Performs deep analysis on code, data, or systems.

```python
from src.agents.analysis_agent import AnalysisAgent, AnalysisDeps

agent = AnalysisAgent(model="anthropic/claude-3.5-sonnet")
deps = AnalysisDeps(
    analysis_type="code_quality",
    include_recommendations=True
)
result = await agent.run("Analyze this codebase", deps)
```

### 4. Tools Refiner Agent
Enhances code with additional tools and capabilities.

```python
from src.agents.tools_refiner import ToolsRefinerAgent, ToolsRefinerDeps

agent = ToolsRefinerAgent(model="groq/mixtral-8x7b-32768")
deps = ToolsRefinerDeps(
    code="your code here",
    requested_tools=["logging", "error_handling", "caching"]
)
result = await agent.run("Add production features", deps)
```

## 🎯 Multi-Model Support

Access 100+ AI models through a unified interface:

### Supported Providers
- **OpenRouter**: Access to all OpenRouter models
- **OpenAI**: GPT-4, GPT-3.5, and other OpenAI models
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini Pro models
- **Meta**: Llama models
- **Mistral**: Mixtral and other models
- **Groq**: Ultra-fast inference

### Model Selection

```python
# Automatic model recommendations
from src.core.model_provider import ModelProvider

provider = ModelProvider(config=model_config)

# Get best model for coding
coding_model = provider.recommend_model(
    task_type="coding",
    budget_priority=False
)

# Get fastest model
fast_model = provider.recommend_model(
    task_type="general",
    speed_priority=True
)

# Get most affordable model
budget_model = provider.recommend_model(
    task_type="general",
    budget_priority=True
)
```

## 🛠️ Tool System

25+ built-in tools organized by category:

### Available Tools

**Development Tools**
- `code_analyzer`: Analyze code structure and quality
- `code_formatter`: Format code according to standards
- `dependency_manager`: Manage project dependencies
- `test_generator`: Generate unit tests
- `documentation_generator`: Create documentation

**Research Tools**
- `web_searcher`: Search the web for information
- `arxiv_searcher`: Search academic papers
- `news_aggregator`: Aggregate news from multiple sources
- `trend_analyzer`: Analyze trends and patterns

**Data Tools**
- `data_cleaner`: Clean and preprocess data
- `data_transformer`: Transform data formats
- `schema_validator`: Validate data schemas
- `data_visualizer`: Create data visualizations

**System Tools**
- `file_manager`: Manage files and directories
- `process_monitor`: Monitor system processes
- `log_analyzer`: Analyze log files
- `performance_profiler`: Profile performance

**Security Tools**
- `vulnerability_scanner`: Scan for vulnerabilities
- `encryption_tool`: Encrypt/decrypt data
- `auth_manager`: Manage authentication
- `security_auditor`: Audit security

## 🎮 Interfaces

### 1. Command Line Interface (CLI)

```bash
# Run a single agent
agent-cli run "Your prompt" --model "anthropic/claude-3.5-sonnet"

# Use a specific tool
agent-cli tools search-web --query "latest AI news"

# Generate from template
agent-cli generate rest-api --name "UserAPI"

# Index codebase for search
agent-cli index ./src --output index.json
```

### 2. Web Interface (Streamlit)

```bash
# Start the Streamlit UI
streamlit run src/ui/app.py

# Features:
# - Visual model selection
# - Real-time progress tracking
# - Multi-agent orchestration
# - Results visualization
```

### 3. API Interface

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload

# Endpoints:
# POST /agents/run
# GET /models/list
# POST /tools/execute
# GET /progress/{task_id}
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Services included:
# - API server
# - Streamlit UI
# - Redis for caching
# - PostgreSQL for persistence
# - Prometheus for monitoring
# - Grafana for visualization
```

## 📊 Knowledge Management

### Vector Store Integration

```python
from src.knowledge.vector_store import VectorStore

# Initialize vector store
store = VectorStore(
    supabase_url="your-url",
    supabase_key="your-key",
    openai_api_key="your-key"
)

# Add documents
await store.add_document(
    content="Your content",
    metadata={"category": "docs", "tags": ["api", "guide"]}
)

# Semantic search
results = await store.search(
    query="How to implement authentication",
    limit=5
)
```

### Validation System

```python
from src.knowledge.validation_gate import ValidationGate

# Create validation gate
gate = ValidationGate()

# Validate code
result = await gate.validate(
    data={"code": "your code"},
    rules=["syntax", "security", "performance"]
)
```

## 🚀 Advanced Features

### Grok Heavy Mode

Deep analysis mode for complex tasks:

```python
from src.progress.grok_mode import GrokHeavyMode, GrokContext

grok = GrokHeavyMode()
context = GrokContext(
    file_path="complex_system.py",
    analysis_depth="DEEP",
    include_patterns=True
)

result = await grok.analyze(context)
```

### Progress Tracking

Real-time progress updates:

```python
from src.progress.progress_tracker import ProgressTracker

tracker = ProgressTracker()

# Create hierarchical tasks
main_task = await tracker.create_task("main", "Main Task")
sub_task = await tracker.create_task("sub", "Sub Task", parent_id="main")

# Update progress
await tracker.update_progress("sub", 50, "Processing...")
```

## 📈 Monitoring

Built-in monitoring with Prometheus and Grafana:

- Request metrics
- Model usage statistics
- Cost tracking
- Performance metrics
- Error rates

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_agents.py
pytest tests/test_tools.py
pytest tests/test_knowledge.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Examples

Check the `examples/` directory for:
- Multi-model usage examples
- Agent orchestration patterns
- Tool integration examples
- Knowledge management demos
- Production deployment guides

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Powered by [OpenRouter](https://openrouter.ai/) for multi-model access
- Vector storage by [Supabase](https://supabase.com/)
- Monitoring with [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/)

## 📞 Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/vedantparmar12/Multi-agent-channel/issues)
- Discussions: [GitHub Discussions](https://github.com/vedantparmar12/Multi-agent-channel/discussions)

---

**Ready to build with multiple AI models?** 🚀

```bash
# Get started now
git clone https://github.com/vedantparmar12/Multi-agent-channel.git
cd Multi-agent-channel
pip install -r requirements.txt
cp config.yaml.example config.yaml
# Edit config.yaml and add your OpenRouter API key
```