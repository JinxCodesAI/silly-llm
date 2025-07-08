"""Efficient batch processing for story generation."""

import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
import logging

from ..common.data_models import (
    StoryPrompt, GeneratedStory, GenerationConfig, 
    GenerationResult, ValidationResult
)
from ..common.llm_providers import TransformersProvider
from ..common.utils import validate_story, clean_generated_text, count_words

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes batches of prompts efficiently using transformers."""
    
    def __init__(self, 
                 llm_provider: TransformersProvider,
                 generation_config: GenerationConfig,
                 validate_stories: bool = True,
                 min_words: int = 50,
                 max_words: int = 300):
        """Initialize batch processor.
        
        Args:
            llm_provider: TransformersProvider instance
            generation_config: Configuration for generation
            validate_stories: Whether to validate generated stories
            min_words: Minimum words for validation
            max_words: Maximum words for validation
        """
        self.llm_provider = llm_provider
        self.generation_config = generation_config
        self.validate_stories = validate_stories
        self.min_words = min_words
        self.max_words = max_words
    
    async def process_batch(self, prompts: List[StoryPrompt]) -> GenerationResult:
        """Process a batch of prompts and generate stories.
        
        Args:
            prompts: List of StoryPrompt objects
            
        Returns:
            GenerationResult with generated stories and metadata
        """
        if not prompts:
            return GenerationResult(
                stories=[],
                total_generation_time=0.0,
                average_tokens_per_second=0.0,
                success_rate=0.0
            )
        
        logger.info(f"Processing batch of {len(prompts)} prompts")
        
        # Prepare messages for chat template
        messages_batch = []
        for prompt in prompts:
            if prompt.k_shot_examples:
                # Convert k-shot examples to message format
                messages = []
                for example in prompt.k_shot_examples:
                    messages.append({"role": example.role, "content": example.content})
                # Add the actual prompt
                messages.append({"role": "user", "content": prompt.full_prompt})

                # Apply chat template
                formatted_text = self.llm_provider.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                messages_batch.append(formatted_text)
            else:
                # No k-shot examples, use prompt directly
                messages_batch.append(prompt.full_prompt)
        
        # Record start time and memory
        start_time = time.time()
        start_memory = self.llm_provider.get_memory_usage()
        
        try:
            # Generate stories
            generated_texts = await self.llm_provider.generate_batch(
                messages_batch, self.generation_config
            )
            
            generation_time = time.time() - start_time
            memory_used = self.llm_provider.get_memory_usage() - start_memory
            
            # Process results
            stories = []
            total_tokens = 0
            
            for i, (prompt, generated_text) in enumerate(zip(prompts, generated_texts)):
                try:
                    story = self._create_story_from_generation(
                        prompt, generated_text, generation_time / len(prompts), 
                        memory_used / len(prompts)
                    )
                    
                    if story:
                        stories.append(story)
                        total_tokens += story.tokens_generated
                
                except Exception as e:
                    logger.error(f"Failed to process story {i}: {e}")
            
            # Calculate metrics
            success_rate = len(stories) / len(prompts) if prompts else 0.0
            avg_tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0.0
            
            result = GenerationResult(
                stories=stories,
                total_generation_time=generation_time,
                average_tokens_per_second=avg_tokens_per_second,
                success_rate=success_rate,
                metadata={
                    "batch_size": len(prompts),
                    "successful_generations": len(stories),
                    "failed_generations": len(prompts) - len(stories),
                    "memory_used_gb": memory_used,
                    "model_name": self.llm_provider.model_name
                }
            )
            
            logger.info(f"Batch completed: {len(stories)}/{len(prompts)} successful, "
                       f"{avg_tokens_per_second:.1f} tokens/sec")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _create_story_from_generation(self, 
                                    prompt: StoryPrompt,
                                    generated_text: str,
                                    generation_time: float,
                                    memory_used: float) -> Optional[GeneratedStory]:
        """Create a GeneratedStory object from generation results.
        
        Args:
            prompt: Original prompt
            generated_text: Generated text
            generation_time: Time taken for generation
            memory_used: Memory used during generation
            
        Returns:
            GeneratedStory object or None if validation fails
        """
        try:
            # Clean the generated text
            cleaned_text = clean_generated_text(generated_text)
            
            if not cleaned_text.strip():
                logger.warning(f"Empty generation for prompt {prompt.prompt_id}")
                return None
            
            # Count words and tokens
            word_count = count_words(cleaned_text)
            tokens_generated = len(self.llm_provider.tokenizer.encode(cleaned_text))
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0.0
            
            # Validate story if enabled
            if self.validate_stories:
                required_words = list(prompt.selected_words.values())
                validation = validate_story(
                    cleaned_text, required_words, self.min_words, self.max_words
                )
                
                if not validation.is_valid:
                    logger.debug(f"Story validation failed for {prompt.prompt_id}: {validation.issues}")
                    return None
            
            # Create story object
            story = GeneratedStory(
                story_id=f"story_{uuid.uuid4().hex[:8]}",
                prompt_id=prompt.prompt_id,
                content=cleaned_text,
                word_count=word_count,
                generation_time=generation_time,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                memory_used_gb=memory_used,
                metadata={
                    "selected_words": prompt.selected_words,
                    "additional_condition": prompt.additional_condition,
                    "k_shot_count": len(prompt.k_shot_examples),
                    "template_version": prompt.metadata.get("template_version", "unknown")
                }
            )
            
            return story
            
        except Exception as e:
            logger.error(f"Failed to create story from generation: {e}")
            return None
    
    async def process_multiple_batches(self, 
                                     all_prompts: List[StoryPrompt],
                                     progress_callback: Optional[callable] = None) -> List[GenerationResult]:
        """Process multiple batches of prompts.
        
        Args:
            all_prompts: All prompts to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of GenerationResult objects
        """
        batch_size = self.generation_config.batch_size
        results = []
        
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_prompts) + batch_size - 1)//batch_size}")
            
            try:
                result = await self.process_batch(batch_prompts)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + len(batch_prompts), len(all_prompts))
                
                # Clear memory between batches
                self.llm_provider.clear_memory()
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        return results
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics and configuration.
        
        Returns:
            Dictionary with processor information
        """
        return {
            "generation_config": self.generation_config.dict(),
            "validation_enabled": self.validate_stories,
            "word_limits": {
                "min_words": self.min_words,
                "max_words": self.max_words
            },
            "llm_provider": self.llm_provider.get_capabilities()
        }
