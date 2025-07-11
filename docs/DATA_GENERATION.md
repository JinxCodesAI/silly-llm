# Synthetic Data Generation Pipeline

This directory contains a comprehensive pipeline for generating synthetic bedtime stories using large language models with efficient batched generation and k-shot prompting.

## Features

- **Template-based generation** with customizable prompts
- **Advanced K-shot prompting** with JSON configuration system and custom sample selectors
- **Efficient batched generation** using the proven `method_batched_generation` approach
- **Word diversity enforcement** to ensure varied vocabulary usage
- **Automatic validation** of generated stories
- **Configurable story features** for different narrative elements
- **Mock provider for testing** without requiring torch/transformers
- **Modular architecture** with proper separation of concerns
- **JSON configuration** for all settings
- **Comprehensive logging and statistics**
- **Multiple LLM provider support** (Transformers, OpenAI-compatible APIs, Mock)

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
│   ├── example_conversation.txt  # Legacy k-shot examples (text format)
│   ├── k_shot_prompting_samples.json  # Default JSON k-shot configurations
│   ├── custom_kshot_samples.json      # Example custom k-shot configurations
│   ├── custom_selectors.py            # Example custom selector functions
│   ├── advanced_kshot_example.json    # Example keyword-based selector config
│   ├── custom_selector_example.json   # Example custom selector config
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

### Detailed Configuration Reference

The configuration system uses JSON files with the following structure. All settings can be overridden via command-line arguments.

#### Model and Provider Settings
- **model_name**: Model identifier to use
  - For TransformersProvider: HuggingFace model name (e.g., `"Qwen/Qwen2.5-3B-Instruct"`)
  - For OpenAI provider: API model name (e.g., `"gpt-3.5-turbo"`, `"gpt-4"`)
  - For MockProvider: Any string (used for logging only)
- **device**: Device selection for local models
  - `"auto"`: Automatically detect best available device (recommended)
  - `"cuda"`: Force GPU usage (requires CUDA-compatible GPU)
  - `"cpu"`: Force CPU usage (slower but works everywhere)
  - `"api"`: Used for API providers (OpenAI-compatible)
  - `"mock"`: Used for MockProvider testing

#### Generation Parameters (`generation` section)
Controls how the language model generates text. These parameters directly affect story quality and generation speed.

- **batch_size**: Number of stories processed simultaneously
  - **Range**: 1-64 (depends on available memory)
  - **Impact**: Higher values = faster generation but more memory usage
  - **Recommendations**:
    - Local models: 8-16 for 3B models, 4-8 for 7B+ models
    - API providers: 3-5 to avoid rate limits
    - Large models: 1-2 to prevent out-of-memory errors
  - **Memory usage**: Roughly linear scaling with batch size

- **max_new_tokens**: Maximum number of tokens to generate per story
  - **Range**: 50-4096 tokens
  - **Impact**: Controls maximum story length (1 token ≈ 0.75 words)
  - **Recommendations**:
    - Short stories: 200-512 tokens (150-400 words)
    - Medium stories: 512-1024 tokens (400-800 words)
    - Long stories: 1024-2048 tokens (800-1500 words)
  - **Note**: Stories may be shorter if model generates end-of-text token

- **temperature**: Controls randomness in text generation
  - **Range**: 0.0-2.0
  - **Impact**:
    - 0.0: Deterministic, always picks most likely token (repetitive)
    - 0.1-0.3: Very focused, consistent but may be repetitive
    - 0.6-0.8: Balanced creativity and coherence (recommended)
    - 1.0-1.2: More creative but less coherent
    - 1.5+: Very random, often incoherent
  - **Recommendations**: 0.6-0.8 for bedtime stories

- **top_p**: Nucleus sampling parameter (alternative to temperature)
  - **Range**: 0.0-1.0
  - **Impact**: Only considers tokens with cumulative probability up to top_p
  - **Recommendations**:
    - 0.9-0.95: Good balance of quality and diversity
    - 0.8-0.9: More focused generation
    - 0.95-1.0: More diverse but potentially less coherent
  - **Note**: Works together with temperature for fine-tuned control

- **do_sample**: Whether to use sampling or greedy decoding
  - **Values**: `true` (recommended) or `false`
  - **Impact**:
    - `true`: Uses temperature/top_p for varied outputs
    - `false`: Always picks most likely token (deterministic but repetitive)
  - **Recommendation**: Always use `true` for creative text generation

- **repetition_penalty**: Penalty for repeating tokens/phrases
  - **Range**: 1.0-1.5
  - **Impact**:
    - 1.0: No penalty (may repeat phrases)
    - 1.1: Light penalty (recommended for most cases)
    - 1.2-1.3: Moderate penalty (reduces repetition)
    - 1.4+: Strong penalty (may hurt coherence)
  - **Recommendation**: 1.1 for balanced repetition control

- **use_cache**: Whether to use key-value cache for faster generation
  - **Values**: `true` (recommended) or `false`
  - **Impact**:
    - `true`: Faster generation, uses more memory
    - `false`: Slower generation, uses less memory
  - **Recommendation**: `true` unless memory is extremely limited

#### Data Paths (`data_paths` section)
File paths for input data. All paths can be overridden via command line.

- **vocabulary_path**: Path to vocabulary JSON file
  - **Format**: JSON with `{"nouns": [...], "verbs": [...], "adjectives": [...]}`
  - **Purpose**: Words randomly selected for each story (word1, word2, word3)
  - **Default**: `"training/synthetic_data_generation/config/vocabulary.json"`

- **story_features_path**: Path to story features JSON file
  - **Format**: JSON array of strings `["condition1", "condition2", ...]`
  - **Purpose**: Additional story conditions randomly selected for variety
  - **Example**: `"make sure story has dialogue"`, `"include magical elements"`
  - **Default**: `"docs/story_features.json"`
  - **Optional**: Can be `null` to disable additional conditions

- **conversation_examples_path**: Path to legacy text k-shot examples file
  - **Format**: Text file with conversation examples for k-shot prompting
  - **Purpose**: Provides context examples to improve story quality (legacy format)
  - **Default**: `"training/synthetic_data_generation/config/example_conversation.txt"`
  - **Optional**: Can be `null` to disable legacy k-shot prompting

- **k_shot_config_file**: Path to JSON k-shot configuration file (recommended)
  - **Format**: JSON file with structured k-shot configurations and metadata
  - **Purpose**: Modern k-shot system with custom sample selection
  - **Default**: `"training/synthetic_data_generation/config/k_shot_prompting_samples.json"`
  - **Optional**: Can be `null` to use legacy format or disable k-shot prompting

- **k_shot_config_name**: Name of specific k-shot configuration to use
  - **Format**: String matching a configuration name in the JSON file
  - **Purpose**: Select specific k-shot configuration from multiple options
  - **Default**: `null` (uses first configuration in file)
  - **Optional**: Can be `null` to use default configuration

#### Generation Settings (`generation_settings` section)
Controls the overall generation process and story variety.

- **num_stories**: Total number of stories to generate
  - **Range**: 1-unlimited
  - **Impact**: Total dataset size
  - **Processing**: Stories are generated in batches for efficiency
  - **Recommendation**: Start with 100-1000 for testing, scale up for production

- **k_shot_count**: Number of example conversations to include in prompts
  - **Range**: 0-10 (practical limit)
  - **Impact**:
    - 0: No examples (faster but lower quality)
    - 1-2: Light context (good balance)
    - 3-5: Rich context (better quality, slower)
    - 5+: Diminishing returns, much slower
  - **Recommendation**: 2-3 for most use cases
  - **Memory impact**: Each example adds ~100-500 tokens to input

- **use_k_shot**: Whether to use k-shot examples at all
  - **Values**: `true` (recommended) or `false`
  - **Impact**:
    - `true`: Uses conversation examples for better story quality
    - `false`: Faster generation but potentially lower quality
  - **Recommendation**: `true` unless speed is critical

- **ensure_diversity**: Prevent word combination repetition across stories
  - **Values**: `true` (recommended) or `false`
  - **Impact**:
    - `true`: Each story uses unique word combinations (more diverse dataset)
    - `false`: Word combinations may repeat (faster but less diverse)
  - **Implementation**: Tracks used combinations, tries up to 100 attempts for uniqueness
  - **Recommendation**: `true` for dataset creation, `false` for quick testing

#### Output Settings (`output_settings` section)
Controls how and where generated stories are saved.

- **output_path**: Path where generated stories will be saved
  - **Format**: JSONL file (one JSON object per line)
  - **Automatic features**:
    - Timestamp added to filename (e.g., `stories_20240101_120000.jsonl`)
    - Metadata saved to `{output_path}.metadata.json`
  - **Example**: `"generated_stories.jsonl"`

- **save_intermediate**: Whether to save progress during generation
  - **Values**: `true` (recommended) or `false`
  - **Impact**:
    - `true`: Saves intermediate files every N stories (recovery from failures)
    - `false`: Only saves final result (risk of losing progress)
  - **Files created**: `{output_path}.intermediate_{count}.jsonl`
  - **Recommendation**: `true` for long-running generations

- **intermediate_save_interval**: How often to save intermediate results
  - **Range**: 10-1000 stories
  - **Impact**: Balance between safety and disk I/O
  - **Recommendations**:
    - Fast generation: 100-500 stories
    - Slow generation: 25-100 stories
    - Very slow/unstable: 10-25 stories

#### Validation Settings (`validation_settings` section)
Quality control for generated stories.

- **validate_stories**: Whether to validate generated stories
  - **Values**: `true` (recommended) or `false`
  - **Impact**:
    - `true`: Filters out invalid stories (better quality, fewer stories)
    - `false`: Keeps all generated text (faster, may include poor quality)
  - **Validation checks**: Word count, required word presence
  - **Recommendation**: `true` for production datasets

- **min_words**: Minimum word count for valid stories
  - **Range**: 10-500 words
  - **Impact**: Filters out very short stories
  - **Recommendations**:
    - Bedtime stories: 50-100 words minimum
    - Short stories: 100-200 words minimum
    - Longer stories: 200+ words minimum

- **max_words**: Maximum word count for valid stories
  - **Range**: 50-2000 words
  - **Impact**: Filters out very long stories
  - **Recommendations**:
    - Bedtime stories: 200-300 words maximum
    - Short stories: 500-800 words maximum
    - Longer stories: 1000+ words maximum
  - **Note**: Should be consistent with `max_new_tokens` setting

#### Logging Settings (`logging` section)
Controls logging verbosity and debugging information.

- **log_level**: Logging verbosity level
  - **Values**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
  - **Impact**:
    - `"DEBUG"`: Detailed execution info, validation details, timing
    - `"INFO"`: Progress updates, batch processing, statistics (recommended)
    - `"WARNING"`: Only warnings and errors
    - `"ERROR"`: Only errors
  - **Recommendation**: `"INFO"` for normal use, `"DEBUG"` for troubleshooting

### Configuration Examples by Use Case

Here are complete configuration examples for different scenarios:

#### Example 1: Development/Testing Configuration
```json
{
  "model_name": "mock-test-model",
  "device": "mock",
  "generation": {
    "batch_size": 10,
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_p": 0.9,
    "do_sample": true,
    "repetition_penalty": 1.0,
    "use_cache": false
  },
  "generation_settings": {
    "num_stories": 20,
    "k_shot_count": 1,
    "use_k_shot": true,
    "ensure_diversity": true
  },
  "output_settings": {
    "output_path": "test_stories.jsonl",
    "save_intermediate": false,
    "intermediate_save_interval": 10
  },
  "validation_settings": {
    "validate_stories": true,
    "min_words": 30,
    "max_words": 200
  },
  "logging": {
    "log_level": "DEBUG"
  }
}
```

#### Example 2: Production Local Model Configuration
```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "device": "auto",
  "generation": {
    "batch_size": 8,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true,
    "repetition_penalty": 1.1,
    "use_cache": true
  },
  "generation_settings": {
    "num_stories": 5000,
    "k_shot_count": 3,
    "use_k_shot": true,
    "ensure_diversity": true
  },
  "output_settings": {
    "output_path": "production_stories.jsonl",
    "save_intermediate": true,
    "intermediate_save_interval": 250
  },
  "validation_settings": {
    "validate_stories": true,
    "min_words": 80,
    "max_words": 400
  },
  "logging": {
    "log_level": "INFO"
  }
}
```

#### Example 3: API Provider Configuration
```json
{
  "model_name": "gpt-3.5-turbo",
  "device": "api",
  "generation": {
    "batch_size": 3,
    "max_new_tokens": 300,
    "temperature": 0.8,
    "top_p": 0.9,
    "do_sample": true,
    "repetition_penalty": 1.0,
    "use_cache": false
  },
  "generation_settings": {
    "num_stories": 1000,
    "k_shot_count": 2,
    "use_k_shot": true,
    "ensure_diversity": true
  },
  "output_settings": {
    "output_path": "api_stories.jsonl",
    "save_intermediate": true,
    "intermediate_save_interval": 100
  },
  "validation_settings": {
    "validate_stories": true,
    "min_words": 50,
    "max_words": 250
  },
  "logging": {
    "log_level": "INFO"
  }
}
```

#### Example 4: High-Performance Large Model Configuration
```json
{
  "model_name": "Qwen/Qwen2.5-32B-Instruct",
  "device": "cuda",
  "generation": {
    "batch_size": 2,
    "max_new_tokens": 1024,
    "temperature": 0.6,
    "top_p": 0.95,
    "do_sample": true,
    "repetition_penalty": 1.1,
    "use_cache": true
  },
  "generation_settings": {
    "num_stories": 10000,
    "k_shot_count": 5,
    "use_k_shot": true,
    "ensure_diversity": true
  },
  "output_settings": {
    "output_path": "large_dataset.jsonl",
    "save_intermediate": true,
    "intermediate_save_interval": 500
  },
  "validation_settings": {
    "validate_stories": true,
    "min_words": 100,
    "max_words": 800
  },
  "logging": {
    "log_level": "INFO"
  }
}
```

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

The system supports advanced k-shot prompting through **configuration files**. All k-shot settings are configured in your JSON config file - no command-line parameters needed for normal usage.

### Step-by-Step Setup Guide

#### Step 1: Create Your K-Shot Sample File

Create a JSON file with your k-shot examples (e.g., `config/my_kshot_samples.json`):

```json
{
  "description": "Custom k-shot prompts for bedtime story generation",
  "samples": [
    {
      "name": "simple-stories",
      "k_shot_count": 1,
      "messages": [
        {
          "role": "user",
          "content": "Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old\ncontaining 3 English words rabbit garden happy\n\nkeep story coherent and gramatically correct"
        },
        {
          "role": "assistant",
          "content": "Rosie the rabbit loved her garden. Every morning, she would hop between the carrot rows, feeling so happy. The garden was full of colorful flowers and tasty vegetables..."
        }
      ],
      "metadata": {
        "theme": "animals",
        "difficulty": "easy"
      }
    }
  ]
}
```

#### Step 2: Configure Your Main Config File

Add k-shot configuration to your main config file (e.g., `config/my_config.json`):

```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "device": "auto",
  "data_paths": {
    "vocabulary_path": "training/synthetic_data_generation/config/vocabulary.json",
    "story_features_path": "docs/story_features.json",
    "k_shot_config_file": "config/my_kshot_samples.json",
    "k_shot_config_name": "simple-stories"
  },
  "k_shot_settings": {
    "selector_type": "default"
  },
  "generation_settings": {
    "num_stories": 100,
    "k_shot_count": 2,
    "use_k_shot": true
  },
  "output_settings": {
    "output_path": "my_stories.jsonl"
  }
}
```

#### Step 3: Run Story Generation

```bash
# Use your configuration file
python -m training.synthetic_data_generation.main --config config/my_config.json
```

That's it! The system will automatically load your k-shot examples and use them during generation.

### Built-in Sample Selection Strategies

Configure different selection strategies in your config file's `k_shot_settings` section:

#### Default Selection (Always First Sample)
```json
{
  "k_shot_settings": {
    "selector_type": "default"
  }
}
```

#### Random Selection
```json
{
  "k_shot_settings": {
    "selector_type": "random"
  }
}
```

#### Keyword-Based Selection
```json
{
  "k_shot_settings": {
    "selector_type": "keyword",
    "keyword_mappings": {
      "animals": "animal-stories",
      "adventure": "adventure-stories",
      "family": "family-stories"
    }
  }
}
```

### Advanced Configuration Examples

#### Multiple K-Shot Samples

Create multiple samples in your k-shot file for variety:

```json
{
  "description": "Multiple themed k-shot examples",
  "samples": [
    {
      "name": "animal-stories",
      "k_shot_count": 1,
      "messages": [
        {
          "role": "user",
          "content": "Generate story with words rabbit garden happy..."
        },
        {
          "role": "assistant",
          "content": "Rosie the rabbit loved her garden..."
        }
      ],
      "metadata": {
        "theme": "animals",
        "difficulty": "easy"
      }
    },
    {
      "name": "adventure-stories",
      "k_shot_count": 2,
      "messages": [
        {
          "role": "user",
          "content": "Generate story with words treasure map brave..."
        },
        {
          "role": "assistant",
          "content": "Captain Sam found an old treasure map..."
        },
        {
          "role": "user",
          "content": "Generate story with words mountain climb courage..."
        },
        {
          "role": "assistant",
          "content": "Little Maya wanted to climb the tall mountain..."
        }
      ],
      "metadata": {
        "theme": "adventure",
        "difficulty": "medium"
      }
    }
  ]
}
```

Then configure keyword-based selection:

```json
{
  "data_paths": {
    "k_shot_config_file": "config/themed_kshot_samples.json",
    "k_shot_config_name": null
  },
  "k_shot_settings": {
    "selector_type": "keyword",
    "keyword_mappings": {
      "animals": "animal-stories",
      "adventure": "adventure-stories"
    },
    "fallback_config": "animal-stories"
  }
}
```

#### Custom Sample Selector Functions

For advanced selection logic, create a custom selector function:

**Step 1**: Create `config/my_selectors.py`:

```python
from typing import List
from training.common.data_models import KShotConfiguration

def theme_based_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Select based on detected theme in prompt."""
    prompt_lower = prompt.lower()

    # Define theme keywords
    themes = {
        "animals": ["rabbit", "cat", "dog", "bird", "zoo"],
        "adventure": ["treasure", "map", "brave", "journey"],
        "magic": ["wizard", "fairy", "spell", "magical"]
    }

    # Find matching theme
    for theme, keywords in themes.items():
        if any(keyword in prompt_lower for keyword in keywords):
            for config in configurations:
                if config.metadata.get("theme") == theme:
                    return config

    # Fallback to first configuration
    return configurations[0]
```

**Step 2**: Configure custom selector in your config file:

```json
{
  "k_shot_settings": {
    "selector_type": "custom",
    "selector_function": "theme_based_selector",
    "selector_module": "config.my_selectors",
    "fallback_config": "animal-stories"
  }
}
```

### Legacy Text Format Support

For backward compatibility, you can still use text-based examples:

```json
{
  "data_paths": {
    "conversation_examples_path": "config/example_conversation.txt"
  },
  "generation_settings": {
    "use_k_shot": true
  }
}
```

### Command-Line Overrides (Optional)

While configuration files are the recommended approach, you can override settings via command line:

```bash
# Override k-shot configuration
python -m training.synthetic_data_generation.main \
    --config config/my_config.json \
    --k-shot-config-file "config/different_samples.json" \
    --k-shot-config-name "specific-sample"

# List available configurations
python -m training.synthetic_data_generation.main \
    --list-k-shot-configs \
    --k-shot-config-file "config/my_kshot_samples.json"
```

### Architecture Improvements (Phase 3)

The k-shot system has been completely redesigned with a clean provider interface:

**New LLMRequest Interface:**
- All providers now use `List[LLMRequest]` instead of `List[str]`
- Proper conversation structure preservation across all providers
- No more provider-specific hacks in BatchProcessor

**Fixed Provider Issues:**
- **OpenAI Provider**: Now properly sends conversation messages to API (was losing k-shot context)
- **Transformers Provider**: Uses `apply_chat_template()` correctly with message structure
- **Mock Provider**: Analyzes k-shot context for better mock responses

**Enhanced Data Models:**
- `KShotConfiguration`: Structured k-shot configurations with metadata
- `KShotLoader`: Unified loader supporting both JSON and text formats
- Custom sample selector functions for intelligent k-shot selection

## Advanced K-Shot Customization

### Complete JSON Sample File Structure

For advanced use cases, here's the complete structure for k-shot sample files:

```json
{
  "description": "Complete k-shot configuration with all features",
  "format": "Structured k-shot examples with comprehensive metadata",
  "samples": [
    {
      "name": "advanced-example",
      "k_shot_count": 2,
      "messages": [
        {
          "role": "user",
          "content": "Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old\ncontaining 3 English words rabbit garden happy\n\nkeep story coherent and gramatically correct"
        },
        {
          "role": "assistant",
          "content": "Rosie the rabbit loved her garden. Every morning, she would hop between the carrot rows, feeling so happy..."
        },
        {
          "role": "user",
          "content": "Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old\ncontaining 3 English words cat sleep dream\n\nkeep story coherent and gramatically correct"
        },
        {
          "role": "assistant",
          "content": "Whiskers the cat loved to sleep in sunny spots. One afternoon, she had the most wonderful dream..."
        }
      ],
      "metadata": {
        "theme": "animals",
        "difficulty": "easy",
        "target_age": "3-5",
        "story_length": "short",
        "keywords": ["animals", "nature", "peaceful"]
      }
    }
  ],
  "usage_notes": [
    "Use for animal-themed stories for young children",
    "Emphasizes simple vocabulary and peaceful themes"
  ],
  "story_features_used": [
    "Character development",
    "Moral lessons",
    "Age-appropriate vocabulary"
  ]
}
```

### Advanced Selector Function Examples

For complex selection logic, you can create custom selector functions. Here are some examples:

#### Theme-Based Selector

Create `config/advanced_selectors.py`:

```python
from typing import List
from training.common.data_models import KShotConfiguration

def theme_based_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Select based on detected theme in prompt."""
    prompt_lower = prompt.lower()

    # Define theme keywords
    themes = {
        "animals": ["rabbit", "cat", "dog", "bird", "zoo", "farm"],
        "adventure": ["treasure", "map", "brave", "journey", "explore"],
        "magic": ["wizard", "fairy", "spell", "magical", "enchanted"],
        "family": ["mom", "dad", "sister", "brother", "family"]
    }

    # Find matching theme
    for theme, keywords in themes.items():
        if any(keyword in prompt_lower for keyword in keywords):
            for config in configurations:
                if config.metadata.get("theme") == theme:
                    return config

    return configurations[0]  # Fallback

def difficulty_based_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Select based on story complexity."""
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ["simple", "easy", "young", "3 years"]):
        target_difficulty = "easy"
    elif any(word in prompt_lower for word in ["complex", "detailed", "older", "8 years"]):
        target_difficulty = "hard"
    else:
        target_difficulty = "medium"

    for config in configurations:
        if config.metadata.get("difficulty") == target_difficulty:
            return config

    return configurations[0]  # Fallback
```

Then configure it in your config file:

```json
{
  "k_shot_settings": {
    "selector_type": "custom",
    "selector_function": "theme_based_selector",
    "selector_module": "config.advanced_selectors"
  }
}
```

### Testing Your K-Shot Configuration

#### Validate Configuration Loading

```bash
# Test your configuration file
python -c "
from training.synthetic_data_generation.src.config import load_config
config = load_config('config/my_config.json')
print(f'✓ Config loaded: {config.data_paths.k_shot_config_file}')
print(f'✓ Selector type: {config.k_shot_settings.selector_type}')
"

# List available k-shot configurations
python -m training.synthetic_data_generation.main \
    --list-k-shot-configs \
    --k-shot-config-file "config/my_kshot_samples.json"
```

#### Test with Mock Provider

```bash
# Generate test stories to verify k-shot examples are working
python -m training.synthetic_data_generation.main \
    --config config/my_config.json \
    --mock-provider \
    --num-stories 3 \
    --output-path test_kshot.jsonl
```

### Best Practices

#### K-Shot Sample Quality
- **Consistency**: All examples should follow the same format and style
- **Relevance**: Examples should match your target story domain
- **Diversity**: Include varied examples covering different scenarios
- **Quality**: Use well-written, grammatically correct examples

#### Configuration Management
- **Version Control**: Keep your k-shot files in version control
- **Documentation**: Add clear descriptions and usage notes
- **Testing**: Test configurations before production use
- **Backup**: Maintain backups of working configurations

#### Selector Function Guidelines
- **Robustness**: Handle edge cases gracefully
- **Performance**: Keep selector functions fast and efficient
- **Fallback**: Always provide a fallback configuration
- **Documentation**: Document your selection logic clearly

### Available Example Configurations

The system includes several example configurations you can use as starting points:

#### Basic Examples
- `config/k_shot_prompting_samples.json` - Default k-shot examples
- `config/advanced_kshot_example.json` - Keyword-based selector configuration
- `config/custom_selector_example.json` - Custom function selector configuration

#### Advanced Examples
- `config/custom_kshot_samples.json` - Themed samples with comprehensive metadata
- `config/custom_selectors.py` - Example custom selector function implementations

You can copy and modify these examples for your specific use case.



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

### Common Configuration Issues

#### Memory and Performance Problems

1. **CUDA out of memory**
   - **Symptoms**: `RuntimeError: CUDA out of memory`
   - **Solutions**:
     - Reduce `batch_size` (try 4, 2, or 1)
     - Reduce `max_new_tokens` (try 256 or 512)
     - Use smaller model (e.g., 3B instead of 7B)
     - Set `use_cache: false` to save memory
   - **Example fix**:
     ```json
     "generation": {
       "batch_size": 2,
       "max_new_tokens": 256,
       "use_cache": false
     }
     ```

2. **Generation too slow**
   - **Symptoms**: Very low stories/minute rate
   - **Solutions**:
     - Increase `batch_size` (if memory allows)
     - Reduce `k_shot_count` or set `use_k_shot: false`
     - Use smaller model or switch to API provider
     - Reduce `max_new_tokens`
   - **Example optimization**:
     ```json
     "generation": {
       "batch_size": 16,
       "max_new_tokens": 300
     },
     "generation_settings": {
       "k_shot_count": 1,
       "use_k_shot": false
     }
     ```

3. **High memory usage**
   - **Symptoms**: System running out of RAM
   - **Solutions**:
     - Reduce `batch_size`
     - Set `save_intermediate: true` with lower `intermediate_save_interval`
     - Use `device: "cpu"` if GPU memory is limited
     - Enable `use_cache: false`

#### Model and Provider Issues

4. **Model not found**
   - **Symptoms**: `Model not found` or `Repository not found`
   - **Solutions**:
     - Verify model name spelling and availability
     - Check HuggingFace Hub access for private models
     - Ensure model supports the required architecture
   - **Common model names**:
     - `"Qwen/Qwen2.5-3B-Instruct"`
     - `"microsoft/DialoGPT-medium"`
     - `"gpt-3.5-turbo"` (for API)

5. **API authentication failed**
   - **Symptoms**: `401 Unauthorized` or `Invalid API key`
   - **Solutions**:
     - Verify `AI_API_KEY` environment variable is set
     - Check API key validity and permissions
     - Verify `api_base_url` is correct for your provider
   - **Example setup**:
     ```bash
     export AI_API_KEY=sk-your-key-here
     python -m training.synthetic_data_generation.main --openai-provider
     ```

#### Data and File Issues

6. **Vocabulary file missing**
   - **Symptoms**: `FileNotFoundError: vocabulary.json`
   - **Solutions**:
     - Check path in `vocabulary_path` setting
     - Verify file exists and is readable
     - Use absolute paths if relative paths fail
   - **Example fix**:
     ```json
     "data_paths": {
       "vocabulary_path": "/absolute/path/to/vocabulary.json"
     }
     ```

7. **Invalid JSON configuration**
   - **Symptoms**: `JSONDecodeError` when loading config
   - **Solutions**:
     - Validate JSON syntax (use online JSON validator)
     - Check for trailing commas, missing quotes
     - Use `--create-config` to generate valid template
   - **Common mistakes**:
     ```json
     // ❌ Wrong: trailing comma
     "temperature": 0.8,

     // ✅ Correct: no trailing comma
     "temperature": 0.8
     ```

#### Generation Quality Issues

8. **Stories too short/long**
   - **Symptoms**: Stories consistently fail validation
   - **Solutions**:
     - Adjust `min_words`/`max_words` in validation settings
     - Modify `max_new_tokens` to control length
     - Check if model is generating end-of-text tokens early
   - **Example adjustment**:
     ```json
     "generation": {
       "max_new_tokens": 800
     },
     "validation_settings": {
       "min_words": 100,
       "max_words": 600
     }
     ```

9. **Poor story quality**
   - **Symptoms**: Incoherent or repetitive stories
   - **Solutions**:
     - Enable k-shot prompting: `use_k_shot: true`
     - Adjust temperature (try 0.7-0.8)
     - Increase `repetition_penalty` (try 1.2)
     - Use better conversation examples
   - **Example quality settings**:
     ```json
     "generation": {
       "temperature": 0.7,
       "repetition_penalty": 1.2
     },
     "generation_settings": {
       "use_k_shot": true,
       "k_shot_count": 3
     }
     ```

10. **Missing required words in stories**
    - **Symptoms**: Stories fail validation for missing words
    - **Solutions**:
      - Check vocabulary file format
      - Verify words are appropriate for the model
      - Adjust validation settings if too strict
      - Review story features for conflicts

### Debug Mode and Logging

Use detailed logging to diagnose issues:

```bash
# Enable debug logging
python -m training.synthetic_data_generation.main --log-level DEBUG --config my_config.json

# Save logs to file
python -m training.synthetic_data_generation.main --log-level DEBUG --config my_config.json > debug.log 2>&1

# Monitor progress with INFO level
python -m training.synthetic_data_generation.main --log-level INFO --config my_config.json
```

### Performance Optimization Guidelines

#### For Local Models
- **Small models (3B)**: `batch_size: 8-16`, `max_new_tokens: 512`
- **Medium models (7B)**: `batch_size: 4-8`, `max_new_tokens: 512`
- **Large models (13B+)**: `batch_size: 1-2`, `max_new_tokens: 256-512`

#### For API Providers
- **Batch size**: 3-5 (avoid rate limits)
- **Save intermediate**: Every 25-100 stories
- **Retry logic**: Built-in for failed requests

#### Memory Usage Estimates
- **3B model**: ~6-8GB GPU memory (batch_size=8)
- **7B model**: ~14-16GB GPU memory (batch_size=4)
- **13B model**: ~26-30GB GPU memory (batch_size=1)

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

- **OpenRouter** - `https://openrouter.ai/api/v1`
- **Together AI** - `https://api.together.xyz/v1`
- **Anyscale** - `https://api.endpoints.anyscale.com/v1`
- **Local servers** (e.g., vLLM, text-generation-webui) - `http://localhost:8000/v1`
- **Other OpenAI-compatible services**

Simply set the appropriate `--api-base-url` and `--model-name` for your service.

## Command Line Parameters Reference

All configuration file settings can be overridden via command-line arguments. Command-line arguments take precedence over configuration file values.

### Configuration Management

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--config`, `-c` | string | Path to JSON configuration file. Works with all providers. Loads all settings from file, then applies command-line overrides. | `--config config/openai_config.json` |
| `--create-config` | string | Create a default configuration template at specified path. Creates a complete JSON template that you can customize for your needs. | `--create-config my_config.json` |

### Provider Selection (mutually exclusive)

Choose exactly one provider. If none specified, defaults to TransformersProvider.

| Parameter | Description | Requirements | Use Case |
|-----------|-------------|--------------|----------|
| *(default)* | **TransformersProvider** - Local HuggingFace models | `torch`, `transformers`, CUDA (optional) | Local generation, full control, no API costs |
| `--mock-provider` | **MockProvider** - Testing without dependencies | None (built-in) | Testing pipeline, development, CI/CD |
| `--openai-provider` | **OpenAI-compatible API provider** | `httpx`, `AI_API_KEY` env var | Cloud generation, no local GPU needed |

### Model and Generation Settings

Core settings that control what model to use and how many stories to generate.

| Parameter | Type | Default | Description | Examples |
|-----------|------|---------|-------------|----------|
| `--model-name` | string | `Qwen/Qwen2.5-3B-Instruct` | Model identifier. Format depends on provider:<br/>• TransformersProvider: HuggingFace model name<br/>• OpenAI provider: API model name<br/>• MockProvider: any string (logging only) | `--model-name "Qwen/Qwen2.5-7B-Instruct"`<br/>`--model-name "gpt-3.5-turbo"`<br/>`--model-name "test-model"` |
| `--num-stories` | integer | `100` | Total number of stories to generate in this run. Stories are processed in batches for efficiency. | `--num-stories 1000` |
| `--batch-size` | integer | `8` | Number of stories processed simultaneously. **Performance impact:**<br/>• Higher = faster generation but more memory<br/>• Lower = slower but uses less memory<br/>**Recommendations:**<br/>• Local 3B models: 8-16<br/>• Local 7B+ models: 4-8<br/>• API providers: 3-5<br/>• Large models: 1-2 | `--batch-size 16` |
| `--output-path` | string | `generated_stories.jsonl` | Output file path for generated stories. **Automatic features:**<br/>• Timestamp added to filename<br/>• Metadata saved to `{path}.metadata.json`<br/>• JSONL format (one story per line) | `--output-path "my_stories.jsonl"` |
| `--device` | choice | `auto` | Device selection for TransformersProvider only:<br/>• `auto`: Automatically detect best device<br/>• `cuda`: Force GPU (requires CUDA)<br/>• `cpu`: Force CPU (slower but universal)<br/>**Ignored for API and Mock providers** | `--device cuda` |

### Data File Overrides

Override input data files from configuration. Useful for testing different vocabularies or examples.

| Parameter | Type | Default | Description | Format |
|-----------|------|---------|-------------|--------|
| `--vocabulary-path` | string | `config/vocabulary.json` | JSON file containing word lists used for story generation. Words are randomly selected as word1, word2, word3 for each story. | JSON: `{"nouns": [...], "verbs": [...], "adjectives": [...]}` |
| `--story-features-path` | string | `docs/story_features.json` | JSON file with additional story conditions. One condition is randomly selected per story to add variety. **Optional:** Can be omitted to disable additional conditions. | JSON: `["make sure story has dialogue", "include magical elements", ...]` |
| `--conversation-examples-path` | string | `config/example_conversation.txt` | Text file containing example conversations for k-shot prompting (legacy format). Improves story quality by providing context examples. **Optional:** Can be omitted to disable legacy k-shot prompting. | Text file with conversation examples |
| `--k-shot-config-file` | string | `config/k_shot_prompting_samples.json` | JSON file containing structured k-shot configurations with metadata. **Recommended** over legacy text format. Provides multiple configurations and custom sample selection. **Optional:** Can be omitted to use legacy format. | JSON: `{"samples": [{"name": "...", "messages": [...]}]}` |
| `--k-shot-config-name` | string | `null` | Name of specific k-shot configuration to use from JSON file. If not specified, uses first configuration. **Requires:** `--k-shot-config-file` to be specified. | String matching configuration name |
| `--list-k-shot-configs` | flag | `false` | List available k-shot configurations from JSON file and exit. **Requires:** `--k-shot-config-file` to be specified. | N/A (flag) |
| `--require-k-shot` | flag | `false` | Fail if k-shot data is missing instead of continuing without examples. Use for strict validation when k-shot prompting is essential. | N/A (flag) |

### API Provider Settings

Settings specific to OpenAI-compatible API providers.

| Parameter | Type | Default | Description | Examples |
|-----------|------|---------|-------------|----------|
| `--api-base-url` | string | `https://openrouter.ai/api/v1` | Base URL for OpenAI-compatible APIs. **Supported services:**<br/>• OpenAI: `https://api.openai.com/v1`<br/>• OpenRouter: `https://openrouter.ai/api/v1`<br/>• Together AI: `https://api.together.xyz/v1`<br/>• Local servers: `http://localhost:8000/v1` | `--api-base-url "https://api.openai.com/v1"` |

### Generation Control Flags

Boolean flags that modify generation behavior.

| Parameter | Default | Description | Impact | Use Cases |
|-----------|---------|-------------|--------|-----------|
| `--no-k-shot` | `False` | Disable k-shot examples completely. **Trade-offs:**<br/>• Faster generation (less input tokens)<br/>• Lower quality stories (no context examples)<br/>• Reduced API costs | **Speed:** ~20-30% faster<br/>**Quality:** May reduce coherence and story structure | Quick testing, minimal quality requirements |
| `--no-diversity` | `False` | Allow repeated word combinations across stories. **Trade-offs:**<br/>• Faster prompt generation (no uniqueness checking)<br/>• Less diverse dataset (repeated patterns)<br/>• Simpler processing | **Speed:** Minimal impact<br/>**Quality:** Reduces dataset diversity | Testing, when word variety isn't important |
| `--log-level` | `INFO` | Set logging verbosity level:<br/>• `DEBUG`: Detailed execution info, validation details<br/>• `INFO`: Progress updates, statistics (recommended)<br/>• `WARNING`: Only warnings and errors<br/>• `ERROR`: Only critical errors | **Performance:** DEBUG mode slightly slower due to extra logging | `DEBUG` for troubleshooting, `INFO` for normal use |

### Environment Variables

Required environment variables for certain providers.

| Variable | Required For | Description | Example |
|----------|--------------|-------------|---------|
| `AI_API_KEY` | `--openai-provider` | API key for authentication with OpenAI-compatible services. **Security:** Never commit API keys to version control. | `export AI_API_KEY=sk-your-key-here` |

### Configuration Best Practices

#### Choosing the Right Settings

**For Development and Testing:**
```json
{
  "generation": {
    "batch_size": 4,
    "max_new_tokens": 200,
    "temperature": 0.8
  },
  "generation_settings": {
    "num_stories": 50,
    "k_shot_count": 1,
    "ensure_diversity": false
  },
  "output_settings": {
    "save_intermediate": false
  },
  "logging": {
    "log_level": "DEBUG"
  }
}
```

**For Production Datasets:**
```json
{
  "generation": {
    "batch_size": 8,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "repetition_penalty": 1.1
  },
  "generation_settings": {
    "num_stories": 10000,
    "k_shot_count": 3,
    "ensure_diversity": true
  },
  "output_settings": {
    "save_intermediate": true,
    "intermediate_save_interval": 500
  },
  "validation_settings": {
    "validate_stories": true,
    "min_words": 80,
    "max_words": 400
  }
}
```

**For API Cost Optimization:**
```json
{
  "generation": {
    "batch_size": 3,
    "max_new_tokens": 250,
    "temperature": 0.8
  },
  "generation_settings": {
    "k_shot_count": 1,
    "use_k_shot": true
  },
  "output_settings": {
    "save_intermediate": true,
    "intermediate_save_interval": 25
  }
}
```

#### Configuration Validation Checklist

Before running large-scale generation, verify:

- [ ] **Model compatibility**: Model name is correct and accessible
- [ ] **Memory requirements**: `batch_size` × `max_new_tokens` fits in available memory
- [ ] **File paths**: All data files exist and are readable
- [ ] **Output settings**: Output directory is writable
- [ ] **Validation ranges**: `min_words`/`max_words` are reasonable for `max_new_tokens`
- [ ] **API settings**: API key is set and base URL is correct (for API providers)
- [ ] **Backup strategy**: `save_intermediate` is enabled for long runs

#### Common Configuration Patterns

**Pattern 1: Quality-focused (slower, better stories)**
```json
{
  "generation": {
    "temperature": 0.6,
    "top_p": 0.9,
    "repetition_penalty": 1.2
  },
  "generation_settings": {
    "k_shot_count": 5,
    "use_k_shot": true,
    "ensure_diversity": true
  },
  "validation_settings": {
    "validate_stories": true,
    "min_words": 100
  }
}
```

**Pattern 2: Speed-focused (faster, acceptable quality)**
```json
{
  "generation": {
    "batch_size": 16,
    "max_new_tokens": 300,
    "temperature": 0.8
  },
  "generation_settings": {
    "k_shot_count": 1,
    "use_k_shot": false,
    "ensure_diversity": false
  },
  "validation_settings": {
    "validate_stories": false
  }
}
```

**Pattern 3: Balanced (good speed and quality)**
```json
{
  "generation": {
    "batch_size": 8,
    "max_new_tokens": 400,
    "temperature": 0.7,
    "repetition_penalty": 1.1
  },
  "generation_settings": {
    "k_shot_count": 2,
    "use_k_shot": true,
    "ensure_diversity": true
  },
  "validation_settings": {
    "validate_stories": true,
    "min_words": 60,
    "max_words": 300
  }
}
```

### Parameter Priority

Settings are applied in this order (later overrides earlier):
1. Default values (from code)
2. Configuration file settings
3. Command line arguments

This allows you to have base configurations and override specific settings as needed.

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
