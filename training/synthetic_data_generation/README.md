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
# Test without installing torch/transformers
python -m training.synthetic_data_generation.main --mock-provider --num-stories 5
```

### Option 2: With OpenAI-Compatible API
1. **Install dependencies**:
   ```bash
   pip install httpx pydantic
   ```

2. **Set your API key**:
   ```bash
   export AI_API_KEY=your_api_key_here
   ```

3. **Generate stories**:
   ```bash
   # Using OpenAI
   python -m training.synthetic_data_generation.main --openai-provider --model-name gpt-3.5-turbo --num-stories 10

   # Using custom API (e.g., local LLM server)
   python -m training.synthetic_data_generation.main --openai-provider --api-base-url http://localhost:8000/v1 --model-name your-model --num-stories 10
   ```

### Option 3: With Local Transformers Models
1. **Install dependencies**:
   ```bash
   pip install torch transformers pydantic
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

## Configuration

The system uses JSON configuration files. See `config/example_config.json` for a complete example.

### Key Configuration Options

- **model_name**: Model to use (HuggingFace model or API model name)
- **num_stories**: Number of stories to generate
- **batch_size**: Batch size for efficient generation
- **k_shot_count**: Number of example conversations to include
- **use_k_shot**: Whether to use k-shot prompting
- **ensure_diversity**: Ensure word combinations are diverse across prompts

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
    --model-name "gpt-3.5-turbo" \
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

## Examples

### Generate 1000 stories with custom model
```bash
python -m training.synthetic_data_generation.main \
    --model-name "microsoft/DialoGPT-medium" \
    --num-stories 1000 \
    --batch-size 12 \
    --output-path "bedtime_stories_1k.jsonl"
```

### Quick test run
```bash
python -m training.synthetic_data_generation.main \
    --num-stories 10 \
    --batch-size 2 \
    --output-path "test_stories.jsonl" \
    --log-level DEBUG
```

### Production run with monitoring
```bash
python -m training.synthetic_data_generation.main \
    --config production_config.json \
    --log-level INFO > generation.log 2>&1
```

## Dependencies

### Core Dependencies (always required)
- `pydantic` - Data validation and settings management

### Provider-Specific Dependencies

#### For TransformersProvider (local models)
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace transformers library

#### For OpenAICompatibleProvider (API-based)
- `httpx` - HTTP client for API requests

#### For MockProvider (testing)
- No additional dependencies

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
