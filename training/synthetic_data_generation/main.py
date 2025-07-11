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
    parser = argparse.ArgumentParser(
        description="""
Generate synthetic bedtime stories using various AI providers.

PROVIDERS:
  - TransformersProvider (default): Local HuggingFace models
  - OpenAICompatibleProvider: OpenAI-compatible APIs (OpenAI, Together AI, local servers)
  - MockProvider: Testing without dependencies

BASIC USAGE:
  # Quick test with mock provider
  python -m training.synthetic_data_generation.main --mock-provider --num-stories 5

  # Use OpenAI API
  export AI_API_KEY=your_key
  python -m training.synthetic_data_generation.main --openai-provider --model-name gpt-3.5-turbo

  # Use configuration file
  python -m training.synthetic_data_generation.main --config config/example_config.json

  # Override config settings
  python -m training.synthetic_data_generation.main --config config/example_config.json --num-stories 100

CONFIGURATION:
  Configuration files work with all providers. Use --config to specify a file,
  then add provider flags (--mock-provider, --openai-provider) as needed.
  Command line arguments override config file settings.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (JSON). Works with all providers. "
             "Example configs: config/example_config.json (general), "
             "config/openai_config.json (API), config/mock_config.json (testing)"
    )

    # Quick setup options
    parser.add_argument(
        "--create-config",
        type=str,
        help="Create a default configuration file at the specified path. "
             "Creates a template JSON file that you can customize for your needs."
    )
    
    # Override options
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name from config. For TransformersProvider: HuggingFace model name "
             "(e.g., 'Qwen/Qwen2.5-3B-Instruct'). For OpenAI: API model name (e.g., 'gpt-3.5-turbo'). "
             "For MockProvider: any string (used for logging only)."
    )

    parser.add_argument(
        "--num-stories",
        type=int,
        help="Override number of stories to generate from config. "
             "Total stories to create in this run. Will be processed in batches."
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="Override output path from config. Path where generated stories will be saved "
             "in JSONL format. Metadata will be saved to {output_path}.metadata.json"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config. Number of stories to process simultaneously. "
             "Larger batches are more efficient but use more memory. Recommended: 8-16 for local models, "
             "3-5 for API providers to avoid rate limits."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        help="Override device from config. Only applies to TransformersProvider. "
             "'auto': automatically detect best device, 'cuda': force GPU, 'cpu': force CPU. "
             "Ignored for OpenAI and Mock providers."
    )

    parser.add_argument(
        "--no-k-shot",
        action="store_true",
        help="Disable k-shot examples. When enabled, stories are generated without example "
             "conversations for context. This may reduce quality but speeds up generation."
    )

    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Disable word diversity enforcement. When enabled, the same word combinations "
             "may be reused across stories. Useful for testing or when you want repeated patterns."
    )

    parser.add_argument(
        "--mock-provider",
        action="store_true",
        help="Use mock LLM provider for testing. Generates fake stories without requiring "
             "torch/transformers or API keys. Perfect for testing the pipeline, configs, "
             "and data processing without computational overhead."
    )

    parser.add_argument(
        "--openai-provider",
        action="store_true",
        help="Use OpenAI-compatible API provider. Requires AI_API_KEY environment variable. "
             "Works with OpenAI, Together AI, Anyscale, local servers (vLLM, text-generation-webui), "
             "and any service implementing OpenAI's chat completions API."
    )

    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for OpenAI-compatible API. Examples: "
             "OpenRouter:https://openrouter.ai/api/v1, "
             "Together AI: https://api.together.xyz/v1, "
             "Local server: http://localhost:8000/v1"
    )

    # Data file overrides
    parser.add_argument(
        "--vocabulary-path",
        type=str,
        help="Override vocabulary file path from config. JSON file containing word lists "
             "(nouns, verbs, adjectives) used for story generation. "
             "Default: training/synthetic_data_generation/config/vocabulary.json"
    )

    parser.add_argument(
        "--story-features-path",
        type=str,
        help="Override story features file path from config. JSON file containing additional "
             "story conditions (e.g., 'make sure story has dialogue'). These are randomly "
             "selected for each story. Default: docs/story_features.json"
    )

    parser.add_argument(
        "--conversation-examples-path",
        type=str,
        help="Override conversation examples file path from config. Text file containing "
             "example conversations for k-shot prompting. Improves story quality by providing "
             "context. Default: training/synthetic_data_generation/config/example_conversation.txt"
    )

    parser.add_argument(
        "--k-shot-config-file",
        type=str,
        help="Path to JSON file containing k-shot configurations. This is the preferred "
             "method for k-shot prompting as it provides structured, maintainable examples "
             "with metadata. Example: docs/k_shot_prompting_samples.json"
    )

    parser.add_argument(
        "--k-shot-config-name",
        type=str,
        help="Name of specific k-shot configuration to use from the JSON file. "
             "If not specified, the first configuration will be used. "
             "Use --list-k-shot-configs to see available configurations."
    )

    parser.add_argument(
        "--list-k-shot-configs",
        action="store_true",
        help="List available k-shot configurations from the specified JSON file and exit. "
             "Requires --k-shot-config-file to be specified."
    )

    parser.add_argument(
        "--require-k-shot",
        action="store_true",
        help="Fail if k-shot data is missing instead of continuing without examples. "
             "Use this for strict validation when k-shot prompting is essential."
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level. DEBUG: detailed execution info, INFO: progress updates, "
             "WARNING: only warnings and errors, ERROR: only errors. "
             "Use DEBUG for troubleshooting, INFO for normal operation."
    )
    
    args = parser.parse_args()

    # Show help if no meaningful arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nQuick start examples:")
        print("  # Test with mock provider (no dependencies)")
        print("  python -m training.synthetic_data_generation.main --mock-provider --num-stories 5")
        print("  # Use OpenAI API")
        print("  export AI_API_KEY=your_key")
        print("  python -m training.synthetic_data_generation.main --openai-provider --model-name gpt-3.5-turbo")
        print("  # Use configuration file")
        print("  python -m training.synthetic_data_generation.main --config config/example_config.json")
        return

    # Handle config file creation
    if args.create_config:
        create_config_file(args.create_config)
        return
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Handle k-shot configuration listing
    if args.list_k_shot_configs:
        if not args.k_shot_config_file:
            print("Error: --k-shot-config-file must be specified when using --list-k-shot-configs")
            sys.exit(1)

        try:
            from training.common.k_shot_loader import KShotLoader
            loader = KShotLoader()
            loader.load_from_json(args.k_shot_config_file)

            print(f"\nAvailable k-shot configurations in {args.k_shot_config_file}:")
            for i, config_name in enumerate(loader.list_configurations(), 1):
                config = loader.get_configuration(config_name)
                print(f"  {i}. {config_name} (k-shot count: {config.k_shot_count})")
            print()

        except Exception as e:
            print(f"Error loading k-shot configurations: {e}")
            sys.exit(1)

        return
    
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

    # Apply k-shot configuration overrides
    if args.k_shot_config_file:
        config.data_paths.k_shot_config_file = args.k_shot_config_file
    if args.k_shot_config_name:
        config.data_paths.k_shot_config_name = args.k_shot_config_name

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

    # Validate k-shot configuration file
    if config.data_paths.k_shot_config_file:
        if not Path(config.data_paths.k_shot_config_file).exists():
            if args.require_k_shot:
                logger.error(f"K-shot configuration file not found: {config.data_paths.k_shot_config_file}")
                sys.exit(1)
            else:
                logger.warning(f"K-shot configuration file not found: {config.data_paths.k_shot_config_file}")
                config.data_paths.k_shot_config_file = None
                config.data_paths.k_shot_config_name = None

    # Check if k-shot is required but no sources are available
    if args.require_k_shot and config.generation_settings.use_k_shot:
        has_k_shot_source = (
            config.data_paths.k_shot_config_file or
            config.data_paths.conversation_examples_path
        )
        if not has_k_shot_source:
            logger.error("K-shot examples are required (--require-k-shot) but no k-shot sources are available")
            logger.error("Provide either --k-shot-config-file or --conversation-examples-path")
            sys.exit(1)
    
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
    logger.info(f"  Max new tokens: {config.generation.max_new_tokens}")
    logger.info(f"  K-shot examples: {config.generation_settings.use_k_shot} (count: {config.generation_settings.k_shot_count})")
    logger.info(f"  Word diversity: {config.generation_settings.ensure_diversity}")
    logger.info(f"  Output path: {config.output_settings.output_path}")

    # Log validation configuration
    logger.info("Validation Configuration:")
    logger.info(f"  Basic validation: {config.validation_settings.validate_stories}")
    logger.info(f"  Word limits: {config.validation_settings.min_words}-{config.validation_settings.max_words}")
    if config.validation_settings.custom_validation:
        cv = config.validation_settings.custom_validation
        logger.info(f"  Custom validation: {cv.validator_class}")
        logger.info(f"  Validation model: {cv.model_name}")
        logger.info(f"  Validation provider: {cv.provider}")
    else:
        logger.info(f"  Custom validation: Disabled")
    
    try:
        # Initialize story generator
        logger.info("Initializing story generator...")
        generator = StoryGenerator(
            model_name=config.model_name,
            vocabulary_path=config.data_paths.vocabulary_path,
            story_features_path=config.data_paths.story_features_path,
            conversation_examples_path=config.data_paths.conversation_examples_path,
            k_shot_config_file=config.data_paths.k_shot_config_file,
            k_shot_config_name=config.data_paths.k_shot_config_name,
            generation_config=config.generation,
            k_shot_count=config.generation_settings.k_shot_count,
            k_shot_settings=config.k_shot_settings.model_dump(),
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
