"""Main CLI entry point for story generation."""

import asyncio
import argparse
import sys
import os
from pathlib import Path
import logging

# Add parent directories to path for imports
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from training.common.utils import setup_logging
from training.synthetic_data_generation.src.config import load_config, create_config_file, StoryGenerationConfig
from training.synthetic_data_generation.src.story_generator import StoryGenerator


async def main():
    """Main entry point for story generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic bedtime stories")
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    
    # Quick setup options
    parser.add_argument(
        "--create-config",
        type=str,
        help="Create a default configuration file at the specified path"
    )
    
    # Override options
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name from config"
    )
    
    parser.add_argument(
        "--num-stories",
        type=int,
        help="Override number of stories to generate"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        help="Override output path from config"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        help="Override device from config"
    )
    
    parser.add_argument(
        "--no-k-shot",
        action="store_true",
        help="Disable k-shot examples"
    )
    
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Disable word diversity enforcement"
    )

    parser.add_argument(
        "--mock-provider",
        action="store_true",
        help="Use mock LLM provider for testing (no torch/transformers required)"
    )

    parser.add_argument(
        "--openai-provider",
        action="store_true",
        help="Use OpenAI-compatible API provider (requires AI_API_KEY env var)"
    )

    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible API (default: OpenAI)"
    )

    # Data file overrides
    parser.add_argument(
        "--vocabulary-path",
        type=str,
        help="Override vocabulary file path from config"
    )

    parser.add_argument(
        "--story-features-path",
        type=str,
        help="Override story features file path from config"
    )

    parser.add_argument(
        "--conversation-examples-path",
        type=str,
        help="Override conversation examples file path from config"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Handle config file creation
    if args.create_config:
        create_config_file(args.create_config)
        return
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Use default configuration
        logger.info("No configuration file specified, using defaults")
        from training.synthetic_data_generation.src.config import DataPaths, OutputSettings
        config = StoryGenerationConfig(
            data_paths=DataPaths(
                vocabulary_path="training/synthetic_data_generation/config/vocabulary.json",
                story_features_path="docs/story_features.json",
                conversation_examples_path="training/synthetic_data_generation/config/example_conversation.txt"
            ),
            output_settings=OutputSettings(
                output_path="generated_stories.jsonl"
            )
        )
    
    # Apply command line overrides
    if args.model_name:
        config.model_name = args.model_name
    if args.num_stories:
        config.generation_settings.num_stories = args.num_stories
    if args.output_path:
        config.output_settings.output_path = args.output_path
    if args.batch_size:
        config.generation.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    if args.no_k_shot:
        config.generation_settings.use_k_shot = False
    if args.no_diversity:
        config.generation_settings.ensure_diversity = False

    # Apply data file overrides
    if args.vocabulary_path:
        config.data_paths.vocabulary_path = args.vocabulary_path
    if args.story_features_path:
        config.data_paths.story_features_path = args.story_features_path
    if args.conversation_examples_path:
        config.data_paths.conversation_examples_path = args.conversation_examples_path

    # Validate provider options
    provider_count = sum([args.mock_provider, args.openai_provider])
    if provider_count > 1:
        logger.error("Cannot use multiple providers simultaneously. Choose one: --mock-provider, --openai-provider, or default transformers.")
        sys.exit(1)
    
    # Validate required paths
    if not Path(config.data_paths.vocabulary_path).exists():
        logger.error(f"Vocabulary file not found: {config.data_paths.vocabulary_path}")
        sys.exit(1)

    if config.data_paths.story_features_path and not Path(config.data_paths.story_features_path).exists():
        logger.warning(f"Story features file not found: {config.data_paths.story_features_path}")
        config.data_paths.story_features_path = None

    if config.data_paths.conversation_examples_path and not Path(config.data_paths.conversation_examples_path).exists():
        logger.warning(f"Conversation examples file not found: {config.data_paths.conversation_examples_path}")
        config.data_paths.conversation_examples_path = None
    
    # Log configuration
    provider_type = "Mock" if args.mock_provider else "OpenAI-compatible API" if args.openai_provider else "Transformers"
    logger.info("Generation Configuration:")
    logger.info(f"  Provider: {provider_type}")
    logger.info(f"  Model: {config.model_name}")
    if args.openai_provider:
        logger.info(f"  API Base URL: {args.api_base_url}")
        logger.info(f"  API Key: {'Set' if os.getenv('AI_API_KEY') else 'NOT SET'}")
    else:
        logger.info(f"  Device: {config.device}")
    logger.info(f"  Stories to generate: {config.generation_settings.num_stories}")
    logger.info(f"  Batch size: {config.generation.batch_size}")
    logger.info(f"  K-shot examples: {config.generation_settings.use_k_shot} (count: {config.generation_settings.k_shot_count})")
    logger.info(f"  Word diversity: {config.generation_settings.ensure_diversity}")
    logger.info(f"  Output path: {config.output_settings.output_path}")
    
    try:
        # Initialize story generator
        logger.info("Initializing story generator...")
        generator = StoryGenerator(
            model_name=config.model_name,
            vocabulary_path=config.data_paths.vocabulary_path,
            story_features_path=config.data_paths.story_features_path,
            conversation_examples_path=config.data_paths.conversation_examples_path,
            generation_config=config.generation,
            k_shot_count=config.generation_settings.k_shot_count,
            device=config.device,
            use_mock_provider=args.mock_provider,
            use_openai_provider=args.openai_provider,
            api_base_url=args.api_base_url,
            validation_settings=config.validation_settings.model_dump()
        )
        
        # Generate stories
        logger.info("Starting story generation...")
        stats = await generator.generate_stories(
            num_stories=config.generation_settings.num_stories,
            output_path=config.output_settings.output_path,
            use_k_shot=config.generation_settings.use_k_shot,
            ensure_diversity=config.generation_settings.ensure_diversity,
            save_intermediate=config.output_settings.save_intermediate,
            intermediate_save_interval=config.output_settings.intermediate_save_interval
        )
        
        # Print summary
        logger.info("Generation completed successfully!")
        logger.info("Summary:")
        logger.info(f"  Total stories: {stats['generation_summary']['total_stories']}")
        logger.info(f"  Total words: {stats['generation_summary']['total_words']}")
        logger.info(f"  Success rate: {stats['generation_summary']['success_rate']:.1%}")
        logger.info(f"  Average words per story: {stats['quality_metrics']['average_word_count']:.1f}")
        logger.info(f"  Generation speed: {stats['quality_metrics']['stories_per_minute']:.1f} stories/min")
        logger.info(f"  Output saved to: {config.output_settings.output_path}")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


def run_generation():
    """Wrapper function for running generation."""
    asyncio.run(main())


if __name__ == "__main__":
    run_generation()
