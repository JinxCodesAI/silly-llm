"""Main story generation orchestrator."""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from ..common.data_models import (
    Vocabulary, GenerationConfig, GeneratedStory, GenerationResult
)
from ..common.llm_providers import TransformersProvider
from ..common.utils import load_vocabulary, save_stories_jsonl
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
                 device: str = "auto"):
        """Initialize story generator.
        
        Args:
            model_name: Name of the model to use
            vocabulary_path: Path to vocabulary JSON file
            story_features_path: Path to story features JSON file
            conversation_examples_path: Path to conversation examples file
            generation_config: Generation configuration
            k_shot_count: Number of k-shot examples to use
            device: Device to use for model
        """
        self.model_name = model_name
        self.vocabulary_path = vocabulary_path
        self.story_features_path = story_features_path
        self.conversation_examples_path = conversation_examples_path
        self.k_shot_count = k_shot_count
        self.device = device
        
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
        
        # Generate prompts
        logger.info("Generating prompts...")
        prompts = self.prompt_generator.generate_prompts(
            count=num_stories,
            use_k_shot=use_k_shot,
            ensure_diversity=ensure_diversity
        )
        
        # Process in batches
        all_stories = []
        all_results = []
        
        def progress_callback(completed: int, total: int):
            logger.info(f"Progress: {completed}/{total} prompts processed ({completed/total*100:.1f}%)")
        
        batch_results = await self.batch_processor.process_multiple_batches(
            prompts, progress_callback=progress_callback
        )
        
        # Collect all stories
        for result in batch_results:
            all_stories.extend(result.stories)
            all_results.append(result)
            
            # Save intermediate results if requested
            if save_intermediate and len(all_stories) % intermediate_save_interval == 0:
                intermediate_path = f"{output_path}.intermediate_{len(all_stories)}.jsonl"
                self._save_stories(all_stories, intermediate_path)
        
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
        story_dicts = [story.dict() for story in stories]
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
