"""Main story generation orchestrator."""

import time
import json
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ...common.data_models import (
    Vocabulary, GenerationConfig, GeneratedStory, GenerationResult
)
from ...common.llm_providers import TransformersProvider, MockLLMProvider, OpenAICompatibleProvider, LLMProvider
from ...common.utils import load_vocabulary, save_stories_jsonl
from .template_manager import TemplateManager
from .prompt_generator import PromptGenerator
from .batch_processor import BatchProcessor
from ..validation.base import BaseValidator

logger = logging.getLogger(__name__)


class StoryGenerator:
    """Main orchestrator for synthetic story generation."""
    
    def __init__(self,
                 model_name: str,
                 vocabulary_path: str,
                 story_features_path: Optional[str] = None,
                 conversation_examples_path: Optional[str] = None,
                 k_shot_config_file: Optional[str] = None,
                 k_shot_config_name: Optional[str] = None,
                 generation_config: Optional[GenerationConfig] = None,
                 k_shot_count: int = 2,
                 k_shot_settings: Optional[Dict[str, Any]] = None,
                 device: str = "auto",
                 use_mock_provider: bool = False,
                 use_openai_provider: bool = False,
                 api_base_url: str = "https://openrouter.ai/api/v1",
                 validation_settings: Optional[Dict[str, Any]] = None):
        """Initialize story generator.

        Args:
            model_name: Name of the model to use
            vocabulary_path: Path to vocabulary JSON file
            story_features_path: Path to story features JSON file
            conversation_examples_path: Path to legacy text conversation examples file
            k_shot_config_file: Path to JSON k-shot configuration file
            k_shot_config_name: Name of specific k-shot configuration to use
            generation_config: Generation configuration
            k_shot_count: Number of k-shot examples to use
            k_shot_settings: Dictionary with k-shot configuration settings
            device: Device to use for model
            use_mock_provider: Whether to use mock provider for testing
            use_openai_provider: Whether to use OpenAI-compatible API provider
            api_base_url: Base URL for OpenAI-compatible API
            validation_settings: Optional validation settings override
        """
        self.model_name = model_name
        self.vocabulary_path = vocabulary_path
        self.story_features_path = story_features_path
        self.conversation_examples_path = conversation_examples_path
        self.k_shot_config_file = k_shot_config_file
        self.k_shot_config_name = k_shot_config_name
        self.k_shot_count = k_shot_count
        self.k_shot_settings = k_shot_settings
        self.device = device
        self.use_mock_provider = use_mock_provider
        self.use_openai_provider = use_openai_provider
        self.api_base_url = api_base_url
        self.validation_settings = validation_settings or {}

        # Use default config if not provided
        self.generation_config = generation_config or GenerationConfig()

        # Track intermediate saves
        self._last_saved_interval = 0

        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components."""
        logger.info("Initializing story generation components...")
        
        # Load vocabulary
        self.vocabulary = load_vocabulary(self.vocabulary_path)
        logger.info(f"Loaded vocabulary: {len(self.vocabulary.nouns)} nouns, "
                   f"{len(self.vocabulary.verbs)} verbs, {len(self.vocabulary.adjectives)} adjectives")
        
        # Initialize LLM provider
        if self.use_mock_provider:
            self.llm_provider = MockLLMProvider(model_name=self.model_name)
            logger.info("Using mock LLM provider for testing")
        elif self.use_openai_provider:
            self.llm_provider = OpenAICompatibleProvider(
                model_name=self.model_name,
                api_base_url=self.api_base_url
            )
            logger.info(f"Using OpenAI-compatible API provider with model: {self.model_name}")
        else:
            self.llm_provider = TransformersProvider(
                model_name=self.model_name,
                device=self.device
            )
        
        # Initialize template manager
        self.template_manager = TemplateManager(
            story_features_path=self.story_features_path
        )
        
        # Initialize prompt generator
        self.prompt_generator = PromptGenerator(
            vocabulary=self.vocabulary,
            template_manager=self.template_manager,
            conversation_examples_path=self.conversation_examples_path,
            k_shot_config_file=self.k_shot_config_file,
            k_shot_config_name=self.k_shot_config_name,
            k_shot_count=self.k_shot_count,
            k_shot_settings=self.k_shot_settings
        )
        
        # Initialize custom validator if configured
        custom_validator = self._create_custom_validator()

        # Initialize batch processor with validation settings
        self.batch_processor = BatchProcessor(
            llm_provider=self.llm_provider,
            generation_config=self.generation_config,
            validate_stories=self.validation_settings.get('validate_stories', True),
            min_words=self.validation_settings.get('min_words', 50),
            max_words=self.validation_settings.get('max_words', 300),
            custom_validator=custom_validator
        )

        logger.info("All components initialized successfully")

    def _create_custom_validator(self) -> Optional[BaseValidator]:
        """Create custom validator if configured.

        Returns:
            BaseValidator instance or None if not configured
        """
        custom_validation_config = self.validation_settings.get('custom_validation')
        if not custom_validation_config:
            return None

        try:
            # Extract configuration
            model_name = custom_validation_config['model_name']
            provider_type = custom_validation_config['provider']
            validator_class_path = custom_validation_config['validator_class']
            generation_config = custom_validation_config.get('generation', {})

            # Create validation provider
            validation_provider = self._create_validation_provider(model_name, provider_type)

            # Import and instantiate validator class
            module_path, class_name = validator_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            validator_class = getattr(module, class_name)

            # Create validator instance
            validator_config = {
                'generation': generation_config
            }
            validator = validator_class(validation_provider, validator_config)

            logger.info(f"Custom validator initialized: {validator_class_path} with model {model_name}")
            return validator

        except Exception as e:
            logger.error(f"Failed to initialize custom validator: {e}")
            return None

    def _create_validation_provider(self, model_name: str, provider_type: str) -> LLMProvider:
        """Create LLM provider for validation.

        Args:
            model_name: Model name for validation
            provider_type: Provider type

        Returns:
            LLMProvider instance
        """
        if provider_type == "MockProvider":
            return MockLLMProvider(model_name=model_name)
        elif provider_type == "OpenAICompatible":
            return OpenAICompatibleProvider(
                model_name=model_name,
                api_base_url=self.api_base_url
            )
        elif provider_type == "TransformersProvider":
            return TransformersProvider(
                model_name=model_name,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown provider type for validation: {provider_type}")

    def _get_diverse_words(self, used_combinations: set) -> Dict[str, str]:
        """Get diverse word combination avoiding previously used combinations.

        Args:
            used_combinations: Set of previously used word combinations

        Returns:
            Dictionary with word1, word2, word3
        """
        max_attempts = 100

        for _ in range(max_attempts):
            words = self.vocabulary.get_random_words()
            combination = tuple(words.values())

            if combination not in used_combinations:
                return words

        # If we can't find a unique combination, return random words
        logger.warning("Could not find unique word combination, using random words")
        return self.vocabulary.get_random_words()

    async def generate_stories(self,
                             num_stories: int,
                             output_path: str,
                             use_k_shot: bool = True,
                             ensure_diversity: bool = True,
                             save_intermediate: bool = True,
                             intermediate_save_interval: int = 100) -> Dict[str, Any]:
        """Generate a dataset of stories.
        
        Args:
            num_stories: Number of stories to generate
            output_path: Path to save generated stories
            use_k_shot: Whether to use k-shot examples
            ensure_diversity: Whether to ensure word diversity
            save_intermediate: Whether to save intermediate results
            intermediate_save_interval: How often to save intermediate results
            
        Returns:
            Dictionary with generation statistics
        """
        logger.info(f"Starting generation of {num_stories} stories")
        start_time = time.time()

        # Reset intermediate save tracking for this generation run
        self._last_saved_interval = 0

        # Process in batches with proper batch IDs
        all_stories = []
        all_results = []
        batch_size = self.generation_config.batch_size
        total_batches = (num_stories + batch_size - 1) // batch_size
        used_word_combinations = set()

        def progress_callback(completed: int, total: int):
            logger.info(f"Progress: {completed}/{total} prompts processed ({completed/total*100:.1f}%)")

        logger.info(f"Processing {num_stories} stories in {total_batches} batches of size {batch_size}")

        for batch_idx in range(total_batches):
            # Calculate stories for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_stories)
            stories_in_batch = end_idx - start_idx

            logger.info(f"Generating prompts for batch {batch_idx + 1}/{total_batches} ({stories_in_batch} stories)")

            # Generate prompts for this specific batch
            batch_prompts = []
            for pos_in_batch in range(stories_in_batch):
                # Generate diverse words if required
                if ensure_diversity:
                    selected_words = self._get_diverse_words(used_word_combinations)
                    used_word_combinations.add(tuple(selected_words.values()))
                else:
                    selected_words = self.vocabulary.get_random_words()

                # Select k-shot examples if requested
                k_shot_examples = []
                if use_k_shot and self.prompt_generator.k_shot_count > 0:
                    k_shot_examples = self.prompt_generator._select_k_shot_examples(selected_words)

                # Create prompt with batch and position info
                prompt_id = f"batch_{batch_idx:03d}_pos_{pos_in_batch:03d}"
                prompt = self.template_manager.create_k_shot_prompt(
                    selected_words=selected_words,
                    k_shot_examples=k_shot_examples,
                    additional_condition=None,  # Let template manager randomly select
                    prompt_id=prompt_id
                )

                batch_prompts.append(prompt)

            # Process this batch with retry logic
            success = False
            max_retries = 3
            retry_delay = 5.0

            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"Processing batch {batch_idx + 1}/{total_batches}" +
                               (f" (attempt {attempt + 1})" if attempt > 0 else ""))

                    # Use adaptive batch processing for better memory handling
                    result = await self.batch_processor.process_batch_with_adaptive_size(
                        batch_prompts, batch_idx
                    )
                    all_results.append(result)
                    all_stories.extend(result.stories)
                    success = True

                    # Progress callback
                    progress_callback(len(all_stories), num_stories)

                    # Save intermediate results if requested
                    if save_intermediate:
                        self._handle_intermediate_save(all_stories, output_path, intermediate_save_interval)

                    # Clear memory between batches
                    self.llm_provider.clear_memory()

                    # Force garbage collection
                    import gc
                    gc.collect()

                    break

                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Batch {batch_idx + 1} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying batch {batch_idx + 1} in {retry_delay} seconds...")

                        # Clear memory before retry
                        self.llm_provider.clear_memory()

                        # Wait before retry
                        import asyncio
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Batch {batch_idx + 1} failed after {max_retries + 1} attempts: {e}")

            if not success:
                logger.error(f"Skipping batch {batch_idx + 1} after all retry attempts failed")
        
        # Final processing is already done in the loop above
        
        # Save final results
        self._save_stories(all_stories, output_path)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        stats = self._calculate_statistics(all_stories, all_results, total_time)
        
        # Save metadata
        metadata_path = Path(output_path).with_suffix('.metadata.json')
        self._save_metadata(stats, metadata_path)
        
        logger.info(f"Generation completed: {len(all_stories)} stories in {total_time:.2f}s")
        return stats
    
    def _save_stories(self, stories: List[GeneratedStory], output_path: str):
        """Save stories to file."""
        # Add timestamp to filename before extension
        path_obj = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = path_obj.with_stem(f"{path_obj.stem}_{timestamp}")

        # Convert to dictionaries for JSON serialization
        story_dicts = []
        correct_story_dicts = []

        for story in stories:
            story_dict = story.dict()
            # Convert datetime to string for JSON serialization
            if 'created_at' in story_dict:
                story_dict['created_at'] = story_dict['created_at'].isoformat()
            story_dicts.append(story_dict)

            # Add to correct stories if word_count > 0
            if story.word_count > 0:
                correct_story_dicts.append(story_dict)

        # Save all stories (existing behavior)
        save_stories_jsonl(story_dicts, str(timestamped_path))

        # Save only correct stories to a separate file
        correct_path = path_obj.with_stem(f"{path_obj.stem}_{timestamp}_correct")
        save_stories_jsonl(correct_story_dicts, str(correct_path))

        logger.info(f"Saved {len(story_dicts)} total stories to {timestamped_path}")
        logger.info(f"Saved {len(correct_story_dicts)} correct stories to {correct_path}")

    def _handle_intermediate_save(self, all_stories: List[GeneratedStory], output_path: str, intermediate_save_interval: int):
        """Handle intermediate saving logic to ensure exactly one save per interval.

        This method tracks intervals and ensures that for each intermediate_save_interval
        exactly one file is created, avoiding the modulo bug where saves could be skipped
        if story counts don't align with interval boundaries.

        Args:
            all_stories: Current list of all generated stories
            output_path: Base output path for intermediate files
            intermediate_save_interval: Number of stories per save interval
        """
        current_story_count = len(all_stories)
        current_interval = current_story_count // intermediate_save_interval

        # Check if we've crossed into a new interval that hasn't been saved yet
        if current_interval > self._last_saved_interval and current_story_count >= intermediate_save_interval:
            # Calculate the exact count for this interval save
            save_count = current_interval * intermediate_save_interval
            intermediate_path = f"{output_path}.intermediate_{save_count}.jsonl"

            logger.info(f"Saving intermediate results: {current_story_count} stories generated, "
                       f"saving at interval {current_interval} (target: {save_count} stories)")

            self._save_stories(all_stories, intermediate_path)
            self._last_saved_interval = current_interval
    
    def _save_metadata(self, metadata: Dict[str, Any], output_path: Path):
        """Save generation metadata."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved metadata to {output_path}")
    
    def _calculate_statistics(self, 
                            stories: List[GeneratedStory],
                            results: List[GenerationResult],
                            total_time: float) -> Dict[str, Any]:
        """Calculate comprehensive statistics."""
        if not stories:
            return {"error": "No stories generated"}
        
        # Basic statistics
        total_words = sum(story.word_count for story in stories)
        total_tokens = sum(story.tokens_generated for story in stories)
        
        # Quality statistics
        avg_word_count = total_words / len(stories)
        avg_generation_time = sum(story.generation_time for story in stories) / len(stories)
        avg_tokens_per_second = sum(story.tokens_per_second for story in stories) / len(stories)
        
        # Success rate
        total_prompts = sum(len(result.stories) for result in results) + sum(
            result.metadata.get("failed_generations", 0) for result in results
        )
        success_rate = len(stories) / total_prompts if total_prompts > 0 else 0.0
        
        # Word usage statistics
        word_usage = self._analyze_word_usage(stories)
        
        return {
            "generation_summary": {
                "total_stories": len(stories),
                "total_words": total_words,
                "total_tokens": total_tokens,
                "total_time_seconds": total_time,
                "success_rate": success_rate
            },
            "quality_metrics": {
                "average_word_count": avg_word_count,
                "average_generation_time": avg_generation_time,
                "average_tokens_per_second": avg_tokens_per_second,
                "stories_per_minute": len(stories) / (total_time / 60) if total_time > 0 else 0
            },
            "word_usage": word_usage,
            "configuration": {
                "model_name": self.model_name,
                "generation_config": self.generation_config.dict(),
                "k_shot_count": self.k_shot_count,
                "vocabulary_size": {
                    "nouns": len(self.vocabulary.nouns),
                    "verbs": len(self.vocabulary.verbs),
                    "adjectives": len(self.vocabulary.adjectives)
                }
            },
            "batch_results": [result.dict() for result in results]
        }
    
    def _analyze_word_usage(self, stories: List[GeneratedStory]) -> Dict[str, Any]:
        """Analyze word usage patterns in generated stories."""
        word_counts = {}
        feature_counts = {}
        
        for story in stories:
            # Count selected words
            selected_words = story.metadata.get("selected_words", {})
            for word in selected_words.values():
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Count features
            condition = story.metadata.get("additional_condition", "")
            if condition:
                feature_counts[condition] = feature_counts.get(condition, 0) + 1
        
        return {
            "most_used_words": sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20],
            "feature_distribution": feature_counts,
            "unique_words_used": len(word_counts),
            "total_word_usage": sum(word_counts.values())
        }
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Get information about the generator configuration."""
        return {
            "model_name": self.model_name,
            "vocabulary_path": self.vocabulary_path,
            "story_features_path": self.story_features_path,
            "conversation_examples_path": self.conversation_examples_path,
            "k_shot_count": self.k_shot_count,
            "device": self.device,
            "generation_config": self.generation_config.dict(),
            "prompt_generator_stats": self.prompt_generator.get_statistics(),
            "template_manager_stats": self.template_manager.get_template_stats(),
            "processor_stats": self.batch_processor.get_processor_stats()
        }
