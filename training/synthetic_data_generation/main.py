"""Main CLI entry point for story generation."""

import asyncio
import argparse
import sys
from pathlib import Path
import logging

# Add parent directories to path for imports
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from training.common.utils import setup_logging
from training.synthetic_data_generation.config import load_config, create_config_file, StoryGenerationConfig
from training.synthetic_data_generation.story_generator import StoryGenerator


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
        config = StoryGenerationConfig(
            vocabulary_path="training/synthetic_data_generation/vocabulary.json",
            output_path="generated_stories.jsonl"
        )
    
    # Apply command line overrides
    if args.model_name:
        config.model_name = args.model_name
    if args.num_stories:
        config.num_stories = args.num_stories
    if args.output_path:
        config.output_path = args.output_path
    if args.batch_size:
        config.generation.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    if args.no_k_shot:
        config.use_k_shot = False
    if args.no_diversity:
        config.ensure_diversity = False
    
    # Validate required paths
    if not Path(config.vocabulary_path).exists():
        logger.error(f"Vocabulary file not found: {config.vocabulary_path}")
        sys.exit(1)
    
    if config.story_features_path and not Path(config.story_features_path).exists():
        logger.warning(f"Story features file not found: {config.story_features_path}")
        config.story_features_path = None
    
    if config.conversation_examples_path and not Path(config.conversation_examples_path).exists():
        logger.warning(f"Conversation examples file not found: {config.conversation_examples_path}")
        config.conversation_examples_path = None
    
    # Log configuration
    logger.info("Generation Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Stories to generate: {config.num_stories}")
    logger.info(f"  Batch size: {config.generation.batch_size}")
    logger.info(f"  K-shot examples: {config.use_k_shot} (count: {config.k_shot_count})")
    logger.info(f"  Word diversity: {config.ensure_diversity}")
    logger.info(f"  Output path: {config.output_path}")
    
    try:
        # Initialize story generator
        logger.info("Initializing story generator...")
        generator = StoryGenerator(
            model_name=config.model_name,
            vocabulary_path=config.vocabulary_path,
            story_features_path=config.story_features_path,
            conversation_examples_path=config.conversation_examples_path,
            generation_config=config.generation,
            k_shot_count=config.k_shot_count,
            device=config.device
        )
        
        # Generate stories
        logger.info("Starting story generation...")
        stats = await generator.generate_stories(
            num_stories=config.num_stories,
            output_path=config.output_path,
            use_k_shot=config.use_k_shot,
            ensure_diversity=config.ensure_diversity,
            save_intermediate=config.save_intermediate,
            intermediate_save_interval=config.intermediate_save_interval
        )
        
        # Print summary
        logger.info("Generation completed successfully!")
        logger.info("Summary:")
        logger.info(f"  Total stories: {stats['generation_summary']['total_stories']}")
        logger.info(f"  Total words: {stats['generation_summary']['total_words']}")
        logger.info(f"  Success rate: {stats['generation_summary']['success_rate']:.1%}")
        logger.info(f"  Average words per story: {stats['quality_metrics']['average_word_count']:.1f}")
        logger.info(f"  Generation speed: {stats['quality_metrics']['stories_per_minute']:.1f} stories/min")
        logger.info(f"  Output saved to: {config.output_path}")
        
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
