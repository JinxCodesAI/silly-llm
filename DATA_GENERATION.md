# Synthetic Data Generation Pipeline

This directory contains a comprehensive pipeline for generating synthetic bedtime stories using large language models with efficient batched generation and k-shot prompting.

## Features

- **Template-based generation** with customizable prompts
- **K-shot prompting** using conversation examples
- **Efficient batched generation** using the proven `method_batched_generation` approach
- **Word diversity enforcement** to ensure varied vocabulary usage
- **Automatic validation** of generated stories
- **Configurable story features** for different narrative elements
- **Mock provider for testing** without requiring torch/transformers
- **Modular architecture** with proper separation of concerns
- **JSON configuration** for all settings
- **Comprehensive logging and statistics**

## Project Structure

```
training/synthetic_data_generation/
├── main.py                    # Main entry point
├── src/                       # Core implementation
│   ├── config.py             # Configuration management
│   ├── template_manager.py   # Template handling
│   ├── prompt_generator.py   # Prompt generation with k-shot
│   ├── batch_processor.py    # Efficient batch processing
│   └── story_generator.py    # Main orchestrator
├── config/                    # Configuration files and data
│   ├── vocabulary.json       # Word vocabulary
│   ├── example_conversation.txt  # K-shot examples
│   ├── default_config.json   # Default configuration
│   └── example_config.json   # Example configuration
├── examples/                  # Example scripts and demos
│   ├── demo.py              # Basic demo
│   ├── k_shot_demo.py       # K-shot prompting demo
│   └── test_mock_provider.py # Mock provider test
└── tests/                     # Test files
    └── test_pipeline.py      # Pipeline tests
```

## Quick Start

### Option 1: With Mock Provider (No Dependencies)
```bash
# Install minimal dependencies
pip install -r requirements-minimal.txt

# Test without installing torch/transformers
python -m training.synthetic_data_generation.main --mock-provider --num-stories 5
```

### Option 2: With OpenAI-Compatible API
1. **Install dependencies**:
   ```bash
   pip install -r requirements-api.txt
   ```

2. **Set your API key**:
   ```bash
   export AI_API_KEY=your_api_key_here
   ```

3. **Generate stories**:
   ```bash
   # Using OpenAI
   python -m training.synthetic_data_generation.main --openai-provider --model-name gpt-4.1-nano --num-stories 10

   # Using custom API (e.g., local LLM server)
   python -m training.synthetic_data_generation.main --openai-provider --api-base-url http://localhost:8000/v1 --model-name your-model --num-stories 10
   ```

### Option 3: With Local Transformers Models
1. **Install dependencies**:
   ```bash
   pip install -r requirements-transformers.txt
   ```

2. **Create a configuration file**:
   ```bash
   python -m training.synthetic_data_generation.main --create-config my_config.json
   ```

3. **Edit the configuration** to match your setup (model, paths, etc.)

4. **Generate stories**:
   ```bash
   python -m training.synthetic_data_generation.main --config my_config.json
   ```

### Option 4: Install All Dependencies
```bash
# Install all dependencies (supports all providers)
pip install -r requirements.txt
```

## Configuration

The system uses JSON configuration files that work with all providers. Multiple example configs are provided:

- **`config/example_config.json`** - General configuration for all providers
- **`config/openai_config.json`** - Optimized for OpenAI-compatible APIs
- **`config/mock_config.json`** - Configuration for testing with MockProvider

### Using Configuration Files

```bash
# Use with any provider
python -m training.synthetic_data_generation.main --config config/example_config.json

# Use with OpenAI provider (requires --openai-provider flag)
python -m training.synthetic_data_generation.main --config config/openai_config.json --openai-provider

# Use with mock provider (requires --mock-provider flag)
python -m training.synthetic_data_generation.main --config config/mock_config.json --mock-provider

# Override specific files from config
python -m training.synthetic_data_generation.main \
    --config config/example_config.json \
    --conversation-examples-path "path/to/your/examples.txt" \
    --story-features-path "path/to/your/features.json"
```

### Key Configuration Sections

#### Model and Provider Settings
- **model_name**: Model to use (HuggingFace model or API model name)
- **device**: Device for local models ("auto", "cuda", "cpu") or "api"/"mock"

#### Generation Parameters
- **batch_size**: Batch size for efficient generation
- **max_new_tokens**: Maximum tokens to generate per story
- **temperature**: Sampling temperature (0.0-2.0)
- **top_p**: Top-p sampling parameter

#### Data Paths (can be overridden via command line)
- **vocabulary_path**: Path to vocabulary JSON file
- **story_features_path**: Path to story features JSON file
- **conversation_examples_path**: Path to k-shot examples file

#### Generation Settings
- **num_stories**: Number of stories to generate
- **k_shot_count**: Number of example conversations to include
- **use_k_shot**: Whether to use k-shot prompting
- **ensure_diversity**: Ensure word combinations are diverse across prompts

#### Validation Settings (works with all providers)
- **validate_stories**: Whether to validate generated stories
- **min_words**: Minimum words per story
- **max_words**: Maximum words per story

### Provider Options

The pipeline supports three different providers:

1. **TransformersProvider** (default): Uses local HuggingFace transformers models
2. **OpenAICompatibleProvider**: Uses OpenAI-compatible APIs (OpenAI, local servers, etc.)
3. **MockProvider**: For testing without dependencies

## Template System

The base template is:
```
Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old
containing words {word1} {word2} {word3}

{additional_condition}

keep story coherent and gramatically correct
```

Where:
- `{word1}`, `{word2}`, `{word3}` are randomly selected from vocabulary
- `{additional_condition}` is randomly selected from `docs/story_features.json`

## K-Shot Prompting

The system supports k-shot prompting using examples from `example_conversation.txt`. The format is:

```
messages_batch = [
    [
        {"role": "user", "content": first_prompt}, 
        {"role": "assistant", "content": first_response}, 
        {"role": "user", "content": second_prompt}
    ]
]
```

## Command Line Usage

### Basic Usage
```bash
# Generate with default settings
python -m training.synthetic_data_generation.main

# Use custom configuration
python -m training.synthetic_data_generation.main --config my_config.json

# Quick overrides
python -m training.synthetic_data_generation.main \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --num-stories 500 \
    --batch-size 16 \
    --output-path "my_stories.jsonl"
```

### Provider-Specific Usage

#### OpenAI-Compatible API
```bash
# Set API key
export AI_API_KEY=your_api_key

# Use OpenAI
python -m training.synthetic_data_generation.main \
    --openai-provider \
    --model-name "gpt-4.1-nano" \
    --num-stories 100

# Use custom API endpoint (e.g., local LLM server)
python -m training.synthetic_data_generation.main \
    --openai-provider \
    --api-base-url "http://localhost:8000/v1" \
    --model-name "your-local-model" \
    --num-stories 100

# Use other OpenAI-compatible services
python -m training.synthetic_data_generation.main \
    --openai-provider \
    --api-base-url "https://api.together.xyz/v1" \
    --model-name "meta-llama/Llama-2-7b-chat-hf" \
    --num-stories 100
```

#### Local Transformers
```bash
# Use specific device
python -m training.synthetic_data_generation.main --device cuda

# Use different model
python -m training.synthetic_data_generation.main \
    --model-name "microsoft/DialoGPT-medium"
```

#### Testing
```bash
# Use mock provider for testing
python -m training.synthetic_data_generation.main --mock-provider

# Test OpenAI provider
python training/synthetic_data_generation/examples/test_openai_provider.py
```

### Command Line Overrides

You can override any configuration setting via command line:

```bash
# Override data files
python -m training.synthetic_data_generation.main \
    --config config/example_config.json \
    --vocabulary-path "path/to/custom/vocabulary.json" \
    --story-features-path "path/to/custom/features.json" \
    --conversation-examples-path "path/to/custom/examples.txt"

# Override generation settings
python -m training.synthetic_data_generation.main \
    --config config/example_config.json \
    --num-stories 500 \
    --batch-size 16 \
    --model-name "different-model"

# Mix config with provider flags
python -m training.synthetic_data_generation.main \
    --config config/openai_config.json \
    --openai-provider \
    --api-base-url "http://localhost:8000/v1" \
    --model-name "local-llama-model"
```

### Advanced Options
```bash
# Disable k-shot examples
python -m training.synthetic_data_generation.main --no-k-shot

# Disable word diversity
python -m training.synthetic_data_generation.main --no-diversity

# Increase logging verbosity
python -m training.synthetic_data_generation.main --log-level DEBUG
```

## Output Format

Generated stories are saved in JSONL format with the following structure:

```json
{
  "story_id": "story_abc12345",
  "prompt_id": "prompt_000001",
  "content": "Once upon a time...",
  "word_count": 127,
  "generation_time": 0.85,
  "tokens_generated": 156,
  "tokens_per_second": 183.5,
  "memory_used_gb": 0.12,
  "created_at": "2024-01-01T12:00:00",
  "metadata": {
    "selected_words": {"word1": "moon", "word2": "dance", "word3": "happy"},
    "additional_condition": "make sure story has sad elements but ends well",
    "k_shot_count": 2,
    "template_version": "1.0"
  }
}
```

## Architecture

The pipeline consists of several modular components:

### Core Components (`training/common/`)
- **LLMProvider**: Abstract interface with TransformersProvider implementation
- **DataModels**: Pydantic models for type safety
- **Utils**: Common utilities for validation and file handling

### Generation Components (`training/synthetic_data_generation/`)
- **TemplateManager**: Manages prompt templates and formatting
- **PromptGenerator**: Generates prompts with k-shot examples and word selection
- **BatchProcessor**: Efficient batched generation using transformers
- **StoryGenerator**: Main orchestrator coordinating all components

## Performance

The system is optimized for efficient generation:

- **Batched processing** for parallel generation
- **Memory management** with automatic cleanup between batches
- **Progress tracking** with intermediate saves
- **Validation filtering** to ensure quality

Typical performance on modern GPUs:
- **3B models**: ~50-100 stories/minute
- **7B models**: ~20-50 stories/minute
- **Memory usage**: 2-8GB depending on model size and batch size

## Customization

### Adding New Story Features
Edit `docs/story_features.json` to add new story conditions:
```json
[
  "make sure the story contains a dialogue",
  "make sure story has magical elements",
  "your custom condition here"
]
```

### Custom Templates
Modify `TemplateManager` to support additional template formats or create template variants.

### Custom Validation
Extend the validation system in `utils.py` to add custom story quality checks.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in configuration
2. **Model not found**: Ensure model name is correct and accessible
3. **Vocabulary file missing**: Check path to `vocabulary.json`
4. **Generation too slow**: Try smaller model or increase batch size

### Debug Mode
Run with `--log-level DEBUG` for detailed logging:
```bash
python -m training.synthetic_data_generation.main --log-level DEBUG --config my_config.yaml
```

## Complete Examples

### 1. Using Configuration Files

#### With TransformersProvider (Local Models)
```bash
# Create and edit config
python -m training.synthetic_data_generation.main --create-config my_config.json
# Edit my_config.json as needed

# Run with config
python -m training.synthetic_data_generation.main --config my_config.json
```

#### With OpenAI-Compatible API
```bash
# Set API key
export AI_API_KEY=your_api_key

# Use provided OpenAI config
python -m training.synthetic_data_generation.main \
    --config config/openai_config.json \
    --openai-provider

# Or override settings
python -m training.synthetic_data_generation.main \
    --config config/openai_config.json \
    --openai-provider \
    --model-name "gpt-4" \
    --num-stories 200
```

#### With MockProvider (Testing)
```bash
# Use provided mock config
python -m training.synthetic_data_generation.main \
    --config config/mock_config.json \
    --mock-provider

# Quick test with custom files
python -m training.synthetic_data_generation.main \
    --config config/mock_config.json \
    --mock-provider \
    --conversation-examples-path "path/to/your/examples.txt"
```

### 2. Using Custom Data Files

#### Custom Conversation Examples
```bash
# Use your own k-shot examples
python -m training.synthetic_data_generation.main \
    --conversation-examples-path "path/to/your/examples.txt" \
    --mock-provider \
    --num-stories 10

# With any provider
python -m training.synthetic_data_generation.main \
    --config config/example_config.json \
    --conversation-examples-path "path/to/your/examples.txt"
```

#### Custom Story Features
```bash
# Use custom story features
python -m training.synthetic_data_generation.main \
    --story-features-path "path/to/your/features.json" \
    --mock-provider \
    --num-stories 10
```

#### Custom Vocabulary
```bash
# Use custom vocabulary
python -m training.synthetic_data_generation.main \
    --vocabulary-path "path/to/your/vocabulary.json" \
    --mock-provider \
    --num-stories 10
```

### 3. Production Examples

#### Large Scale Generation
```bash
python -m training.synthetic_data_generation.main \
    --config config/example_config.json \
    --num-stories 10000 \
    --batch-size 16 \
    --output-path "large_dataset.jsonl"
```

#### With Monitoring
```bash
python -m training.synthetic_data_generation.main \
    --config config/openai_config.json \
    --openai-provider \
    --log-level INFO > generation.log 2>&1
```

#### Custom API Endpoint
```bash
export AI_API_KEY=your_local_api_key
python -m training.synthetic_data_generation.main \
    --config config/openai_config.json \
    --openai-provider \
    --api-base-url "http://localhost:8000/v1" \
    --model-name "local-llama-7b"
```

## Dependencies

The project provides multiple requirements files for different use cases:

### Requirements Files

- **`requirements.txt`** - All dependencies (supports all providers)
- **`requirements-minimal.txt`** - Minimal dependencies (MockProvider only)
- **`requirements-api.txt`** - API provider dependencies (OpenAI-compatible)
- **`requirements-transformers.txt`** - Local model dependencies (TransformersProvider)
- **`requirements-dev.txt`** - Development dependencies (includes testing tools)

### Installation Options

```bash
# For testing only (MockProvider)
pip install -r requirements-minimal.txt

# For API-based generation (OpenAI, Together AI, etc.)
pip install -r requirements-api.txt

# For local model generation (HuggingFace transformers)
pip install -r requirements-transformers.txt

# For all features
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Core Dependencies by Provider

#### MockProvider (testing)
- `pydantic>=2.0.0`

#### OpenAICompatibleProvider (API-based)
- `pydantic>=2.0.0`
- `httpx>=0.24.0`

#### TransformersProvider (local models)
- `pydantic>=2.0.0`
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `accelerate>=0.20.0`
- `safetensors>=0.3.0`

## Environment Variables

### For OpenAI-Compatible Provider
- `AI_API_KEY` - Your API key for the service (required)

Example:
```bash
export AI_API_KEY=sk-your-openai-api-key-here
# or for other services:
export AI_API_KEY=your-api-key-here
```

## Supported API Providers

The OpenAI-compatible provider works with any service that implements the OpenAI chat completions API:

- **OpenAI** - `https://api.openai.com/v1`
- **Together AI** - `https://api.together.xyz/v1`
- **Anyscale** - `https://api.endpoints.anyscale.com/v1`
- **Local servers** (e.g., vLLM, text-generation-webui) - `http://localhost:8000/v1`
- **Other OpenAI-compatible services**

Simply set the appropriate `--api-base-url` and `--model-name` for your service.

## Command Line Parameters Reference

### Configuration Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--config`, `-c` | string | Path to JSON configuration file. Works with all providers. | `--config config/openai_config.json` |
| `--create-config` | string | Create a default configuration template at specified path | `--create-config my_config.json` |

### Provider Selection (mutually exclusive)

| Parameter | Description | Requirements |
|-----------|-------------|--------------|
| *(default)* | TransformersProvider - Local HuggingFace models | `torch`, `transformers` |
| `--mock-provider` | MockProvider - Testing without dependencies | None (built-in) |
| `--openai-provider` | OpenAI-compatible API provider | `httpx`, `AI_API_KEY` env var |

### Model and Generation Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-name` | string | `Qwen/Qwen2.5-3B-Instruct` | Model identifier. HuggingFace model name for local, API model name for remote |
| `--num-stories` | integer | `100` | Total number of stories to generate |
| `--batch-size` | integer | `8` | Stories processed simultaneously. Higher = faster but more memory |
| `--output-path` | string | `generated_stories.jsonl` | Output file path. Metadata saved to `{path}.metadata.json` |
| `--device` | choice | `auto` | Device for local models: `auto`, `cuda`, `cpu` |

### Data File Overrides

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--vocabulary-path` | string | `config/vocabulary.json` | JSON file with word lists (nouns, verbs, adjectives) |
| `--story-features-path` | string | `docs/story_features.json` | JSON file with story conditions (randomly selected) |
| `--conversation-examples-path` | string | `config/example_conversation.txt` | Text file with k-shot examples |

### API Provider Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--api-base-url` | string | `https://api.openai.com/v1` | Base URL for OpenAI-compatible APIs |

### Generation Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--no-k-shot` | flag | `False` | Disable k-shot examples (faster but lower quality) |
| `--no-diversity` | flag | `False` | Allow repeated word combinations across stories |
| `--log-level` | choice | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Environment Variables

| Variable | Required For | Description |
|----------|--------------|-------------|
| `AI_API_KEY` | OpenAI provider | API key for authentication |

### Parameter Priority

Settings are applied in this order (later overrides earlier):
1. Default values
2. Configuration file settings
3. Command line arguments

### Examples by Use Case

#### Quick Testing
```bash
# Minimal test
python -m training.synthetic_data_generation.main --mock-provider --num-stories 5

# Test with custom data
python -m training.synthetic_data_generation.main --mock-provider \
    --conversation-examples-path "my_examples.txt" --num-stories 10
```

#### API Usage
```bash
# OpenAI
export AI_API_KEY=sk-your-key
python -m training.synthetic_data_generation.main --openai-provider \
    --model-name "gpt-3.5-turbo" --num-stories 50

# Local API server
export AI_API_KEY=local-key
python -m training.synthetic_data_generation.main --openai-provider \
    --api-base-url "http://localhost:8000/v1" --model-name "llama-7b"
```

#### Local Models
```bash
# Default model
python -m training.synthetic_data_generation.main --num-stories 100

# Custom model and settings
python -m training.synthetic_data_generation.main \
    --model-name "microsoft/DialoGPT-medium" \
    --device cuda --batch-size 16 --num-stories 1000
```

#### Configuration Files
```bash
# Use config file
python -m training.synthetic_data_generation.main --config config/openai_config.json --openai-provider

# Override config settings
python -m training.synthetic_data_generation.main --config config/example_config.json \
    --num-stories 500 --batch-size 12
```
