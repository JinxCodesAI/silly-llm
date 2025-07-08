# Synthetic Data Generation Pipeline

This directory contains a comprehensive pipeline for generating synthetic bedtime stories using large language models with efficient batched generation and k-shot prompting.

## Features

- **Template-based generation** with customizable prompts
- **K-shot prompting** using conversation examples
- **Efficient batched generation** using the proven `method_batched_generation` approach
- **Word diversity enforcement** to ensure varied vocabulary usage
- **Automatic validation** of generated stories
- **Configurable story features** for different narrative elements
- **Comprehensive logging and statistics**

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install torch transformers pydantic pyyaml
   ```

2. **Create a configuration file**:
   ```bash
   python -m training.synthetic_data_generation.main --create-config my_config.yaml
   ```

3. **Edit the configuration** to match your setup (model, paths, etc.)

4. **Generate stories**:
   ```bash
   python -m training.synthetic_data_generation.main --config my_config.yaml
   ```

## Configuration

The system uses YAML configuration files. See `example_config.yaml` for a complete example.

### Key Configuration Options

- **model_name**: HuggingFace model to use (default: "Qwen/Qwen2.5-3B-Instruct")
- **num_stories**: Number of stories to generate
- **batch_size**: Batch size for efficient generation
- **k_shot_count**: Number of example conversations to include
- **use_k_shot**: Whether to use k-shot prompting
- **ensure_diversity**: Ensure word combinations are diverse across prompts

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
python -m training.synthetic_data_generation.main --config my_config.yaml

# Quick overrides
python -m training.synthetic_data_generation.main \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --num-stories 500 \
    --batch-size 16 \
    --output-path "my_stories.jsonl"
```

### Advanced Options
```bash
# Disable k-shot examples
python -m training.synthetic_data_generation.main --no-k-shot

# Disable word diversity
python -m training.synthetic_data_generation.main --no-diversity

# Use specific device
python -m training.synthetic_data_generation.main --device cuda

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
    --config production_config.yaml \
    --log-level INFO > generation.log 2>&1
```
