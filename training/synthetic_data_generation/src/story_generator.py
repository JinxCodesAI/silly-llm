"""Main story generation orchestrator."""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from ...common.data_models import (
    Vocabulary, GenerationConfig, GeneratedStory, GenerationResult
)
from ...common.llm_providers import TransformersProvider, MockLLMProvider, OpenAICompatibleProvider, LLMProvider
from ...common.utils import load_vocabulary, save_stories_jsonl
from .template_manager import TemplateManager
from .prompt_generator import PromptGenerator
from .batch_processor import BatchProcessor

logger = logging.getLogger(__name__)


class StoryGenerator:
    """Main orchestrator for synthetic story generation."""
    
    def __init__(self,
                 model_name: str,
                 vocabulary_path: str,
                 story_features_path: Optional[str] = None,
                 conversation_examples_path: Optional[str] = None,
                 generation_config: Optional[GenerationConfig] = None,
                 k_shot_count: int = 2,
                 device: str = "auto",
                 use_mock_provider: bool = False,
                 use_openai_provider: bool = False,
                 api_base_url: str = "https://api.openai.com/v1"):
        """Initialize story generator.
        
        Args:
            model_name: Name of the model to use
            vocabulary_path: Path to vocabulary JSON file
            story_features_path: Path to story features JSON file
            conversation_examples_path: Path to conversation examples file
            generation_config: Generation configuration
            k_shot_count: Number of k-shot examples to use
            device: Device to use for model
            use_mock_provider: Whether to use mock provider for testing
            use_openai_provider: Whether to use OpenAI-compatible API provider
            api_base_url: Base URL for OpenAI-compatible API
        """
        self.model_name = model_name
        self.vocabulary_path = vocabulary_path
        self.story_features_path = story_features_path
        self.conversation_examples_path = conversation_examples_path
        self.k_shot_count = k_shot_count
        self.device = device
        self.use_mock_provider = use_mock_provider
        self.use_openai_provider = use_openai_provider
        self.api_base_url = api_base_url
        
        # Use default config if not provided
        self.generation_config = generation_config or GenerationConfig()
        
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
            k_shot_count=self.k_shot_count
        )
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            llm_provider=self.llm_provider,
            generation_config=self.generation_config
        )
        
        logger.info("All components initialized successfully")

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
                if use_k_shot and self.prompt_generator.k_shot_count > 0 and self.prompt_generator.conversation_examples:
                    k_shot_examples = self.prompt_generator._select_k_shot_examples()

                # Create prompt with batch and position info
                prompt_id = f"batch_{batch_idx:03d}_pos_{pos_in_batch:03d}"
                prompt = self.template_manager.create_k_shot_prompt(
                    selected_words=selected_words,
                    k_shot_examples=k_shot_examples,
                    additional_condition=None,  # Let template manager randomly select
                    prompt_id=prompt_id
                )

                batch_prompts.append(prompt)

            # Process this batch
            try:
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
                
                result = await self.batch_processor.process_batch_with_ids(
                    batch_prompts, batch_idx
                )
                all_results.append(result)
                all_stories.extend(result.stories)

                # Progress callback
                progress_callback(len(all_stories), num_stories)

                # Save intermediate results if requested
                if save_intermediate and len(all_stories) % intermediate_save_interval == 0:
                    intermediate_path = f"{output_path}.intermediate_{len(all_stories)}.jsonl"
                    self._save_stories(all_stories, intermediate_path)

                # Clear memory between batches
                self.llm_provider.clear_memory()

            except Exception as e:
                logger.error(f"Failed to process batch {batch_idx + 1}: {e}")
                # Continue with next batch
                continue
        
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
        # Convert to dictionaries for JSON serialization
        story_dicts = []
        for story in stories:
            story_dict = story.dict()
            # Convert datetime to string for JSON serialization
            if 'created_at' in story_dict:
                story_dict['created_at'] = story_dict['created_at'].isoformat()
            story_dicts.append(story_dict)
        save_stories_jsonl(story_dicts, output_path)
    
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
