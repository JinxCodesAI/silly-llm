"""Configuration management for story generation."""

from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import json
from pydantic import BaseModel, Field

from ..common.data_models import GenerationConfig


class StoryGenerationConfig(BaseModel):
    """Complete configuration for story generation."""
    
    # Model configuration
    model_name: str = Field(default="Qwen/Qwen2.5-3B-Instruct", description="Model name to use")
    device: str = Field(default="auto", description="Device to use (auto, cuda, cpu)")
    
    # Generation parameters
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    
    # Data paths
    vocabulary_path: str = Field(description="Path to vocabulary JSON file")
    story_features_path: Optional[str] = Field(default=None, description="Path to story features JSON")
    conversation_examples_path: Optional[str] = Field(default=None, description="Path to conversation examples")
    
    # Generation settings
    num_stories: int = Field(default=1000, description="Number of stories to generate")
    k_shot_count: int = Field(default=2, description="Number of k-shot examples")
    use_k_shot: bool = Field(default=True, description="Whether to use k-shot examples")
    ensure_diversity: bool = Field(default=True, description="Ensure word diversity across prompts")
    
    # Output settings
    output_path: str = Field(description="Path to save generated stories")
    save_intermediate: bool = Field(default=True, description="Save intermediate results")
    intermediate_save_interval: int = Field(default=100, description="Save every N stories")
    
    # Validation settings
    validate_stories: bool = Field(default=True, description="Validate generated stories")
    min_words: int = Field(default=50, description="Minimum words per story")
    max_words: int = Field(default=300, description="Maximum words per story")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")


def load_config(config_path: str) -> StoryGenerationConfig:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        StoryGenerationConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load based on file extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return StoryGenerationConfig(**data)


def save_config(config: StoryGenerationConfig, output_path: str):
    """Save configuration to file.
    
    Args:
        config: Configuration object
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    
    # Save as YAML by default
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2)


def create_default_config(
    vocabulary_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_stories: int = 1000
) -> StoryGenerationConfig:
    """Create a default configuration.
    
    Args:
        vocabulary_path: Path to vocabulary file
        output_path: Path for output
        model_name: Model to use
        num_stories: Number of stories to generate
        
    Returns:
        StoryGenerationConfig object
    """
    return StoryGenerationConfig(
        model_name=model_name,
        vocabulary_path=vocabulary_path,
        output_path=output_path,
        num_stories=num_stories,
        story_features_path="docs/story_features.json",
        conversation_examples_path="training/synthetic_data_generation/example_conversation.txt"
    )


# Default configuration template
DEFAULT_CONFIG_TEMPLATE = """
# Story Generation Configuration

# Model settings
model_name: "Qwen/Qwen2.5-3B-Instruct"
device: "auto"  # auto, cuda, cpu

# Generation parameters
generation:
  batch_size: 8
  max_new_tokens: 512
  temperature: 0.8
  top_p: 0.9
  do_sample: true
  repetition_penalty: 1.1
  use_cache: true

# Data paths
vocabulary_path: "training/synthetic_data_generation/vocabulary.json"
story_features_path: "docs/story_features.json"
conversation_examples_path: "training/synthetic_data_generation/example_conversation.txt"

# Generation settings
num_stories: 1000
k_shot_count: 2
use_k_shot: true
ensure_diversity: true

# Output settings
output_path: "generated_stories.jsonl"
save_intermediate: true
intermediate_save_interval: 100

# Validation settings
validate_stories: true
min_words: 50
max_words: 300

# Logging
log_level: "INFO"
"""


def create_config_file(output_path: str):
    """Create a default configuration file.
    
    Args:
        output_path: Path to save the configuration file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(DEFAULT_CONFIG_TEMPLATE.strip())
    
    print(f"Created default configuration file: {output_path}")
    print("Please edit the configuration file to match your setup before running generation.")


if __name__ == "__main__":
    # Create a default config file when run as script
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "story_generation_config.yaml"
    
    create_config_file(output_path)
