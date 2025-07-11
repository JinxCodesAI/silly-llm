"""Efficient batch processing for story generation."""

import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
import logging

from ...common.data_models import (
    StoryPrompt, GeneratedStory, GenerationConfig,
    GenerationResult, ValidationResult, LLMRequest
)
from ...common.llm_providers import LLMProvider
from ...common.utils import validate_story, clean_generated_text, count_words
from ..validation.base import BaseValidator

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes batches of prompts efficiently using transformers."""
    
    def __init__(self,
                 llm_provider: LLMProvider,
                 generation_config: GenerationConfig,
                 validate_stories: bool = True,
                 min_words: int = 50,
                 max_words: int = 300,
                 custom_validator: Optional[BaseValidator] = None):
        """Initialize batch processor.

        Args:
            llm_provider: LLMProvider instance
            generation_config: Configuration for generation
            validate_stories: Whether to validate generated stories
            min_words: Minimum words for validation
            max_words: Maximum words for validation
            custom_validator: Optional custom validator instance
        """
        self.llm_provider = llm_provider
        self.generation_config = generation_config
        self.validate_stories = validate_stories
        self.min_words = min_words
        self.max_words = max_words
        self.custom_validator = custom_validator
    
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

        # Convert StoryPrompts to LLMRequests - clean, no provider-specific hacks!
        requests = [LLMRequest.from_story_prompt(prompt) for prompt in prompts]
        
        # Record start time and memory
        start_time = time.time()
        start_memory = self.llm_provider.get_memory_usage()
        
        try:
            # Generate stories using the new LLMRequest interface
            generated_texts = await self.llm_provider.generate_batch(
                requests, self.generation_config
            )
            
            generation_time = time.time() - start_time
            memory_used = self.llm_provider.get_memory_usage() - start_memory
            
            # Process results with batch validation support
            stories = []
            total_tokens = 0

            # Check if we should use batch validation
            if self.custom_validator and hasattr(self.custom_validator, 'validate_batch'):
                stories, total_tokens = await self._process_batch_with_validation(
                    prompts, generated_texts, generation_time, memory_used
                )
            else:
                # Use individual processing (legacy behavior)
                for i, (prompt, generated_text) in enumerate(zip(prompts, generated_texts)):
                    try:
                        story = await self._create_story_from_generation(
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

    def _generate_error_story(self,  prompt: StoryPrompt, generated_text: str, rejection_details: Any):
        return GeneratedStory(
                story_id=f"story_{uuid.uuid4().hex[:8]}",
                prompt_id=prompt.prompt_id,
                content=generated_text,
                word_count=0,
                generation_time=0,
                tokens_generated=0,
                tokens_per_second=0,
                memory_used_gb=0,
                metadata=rejection_details
            )

    async def _create_story_from_generation(self,
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
                return self._generate_error_story(prompt, generated_text, { "error":"generated text empty"})
            
            # Count words and tokens
            word_count = count_words(cleaned_text)

            # Count tokens if tokenizer is available, otherwise estimate
            if hasattr(self.llm_provider, 'tokenizer') and self.llm_provider.tokenizer:
                try:
                    tokens_generated = len(self.llm_provider.tokenizer.encode(cleaned_text))
                except Exception:
                    # Fallback estimation: roughly 4 characters per token
                    tokens_generated = len(cleaned_text) // 4
            else:
                # Fallback estimation: roughly 4 characters per token
                tokens_generated = len(cleaned_text) // 4

            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0.0
            
            # Validate story if enabled
            if self.validate_stories:
                # Traditional validation (word count, required words)
                required_words = list(prompt.selected_words.values())
                traditional_validation = validate_story(
                    cleaned_text, required_words, self.min_words, self.max_words
                )

                if not traditional_validation.is_valid:
                    logger.debug(f"Traditional validation failed for {prompt.prompt_id}: {traditional_validation.issues}")
                    return self._generate_error_story(prompt, generated_text, { "prompt_id":prompt.prompt_id, "issues":traditional_validation.issues })

                # Custom validation if configured
                if self.custom_validator:
                    try:
                        custom_validation = await self.custom_validator.validate(cleaned_text)
                        if not custom_validation.is_valid:
                            logger.debug(f"Custom validation failed for {prompt.prompt_id}: {custom_validation.reasoning}")
                            return self._generate_error_story(prompt, generated_text, { "prompt_id":prompt.prompt_id, "reasoning":custom_validation.reasoning })
                    except Exception as e:
                        logger.warning(f"Custom validation error for {prompt.prompt_id}: {e}")
                        # Continue with story if custom validation fails due to error
            
            # Prepare k-shot examples for metadata
            k_shot_examples_data = []
            for example in prompt.k_shot_examples:
                k_shot_examples_data.append({
                    "role": example.role,
                    "content": example.content
                })

            # Create story object with enhanced metadata
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
                    "template_version": prompt.metadata.get("template_version", "unknown"),
                    "full_prompt": prompt.full_prompt,
                    "template": prompt.template,
                    "k_shot_examples": k_shot_examples_data,
                    "multi_turn_conversation": {
                        "messages": k_shot_examples_data + [{
                            "role": "user",
                            "content": prompt.full_prompt
                        }],
                        "total_messages": len(k_shot_examples_data) + 1,
                        "conversation_format": "openai_chat_format"
                    }
                }
            )
            
            return story
            
        except Exception as e:
            logger.error(f"Failed to create story from generation: {e}")
            return None

    async def _process_batch_with_validation(self, prompts, generated_texts, generation_time, memory_used):
        """Process a batch of generated texts with batch validation support.

        Args:
            prompts: List of StoryPrompt objects
            generated_texts: List of generated text strings
            generation_time: Total generation time for the batch
            memory_used: Memory used for the batch

        Returns:
            Tuple of (stories list, total_tokens)
        """
        from ...common.utils import clean_generated_text, validate_story

        stories = []
        total_tokens = 0

        # First pass: clean texts and perform traditional validation
        cleaned_texts = []
        valid_indices = []

        for i, (prompt, generated_text) in enumerate(zip(prompts, generated_texts)):
            try:
                # Clean the generated text
                cleaned_text = clean_generated_text(generated_text)

                if not cleaned_text.strip():
                    logger.warning(f"Empty generation for prompt {prompt.prompt_id}")
                    continue

                # Traditional validation (word count, required words)
                if self.validate_stories:
                    required_words = list(prompt.selected_words.values())
                    traditional_validation = validate_story(
                        cleaned_text, required_words, self.min_words, self.max_words
                    )

                    if not traditional_validation.is_valid:
                        logger.debug(f"Traditional validation failed for {prompt.prompt_id}: {traditional_validation.issues}")
                        continue

                cleaned_texts.append(cleaned_text)
                valid_indices.append(i)

            except Exception as e:
                logger.error(f"Failed to process story {i} in batch validation: {e}")
                continue

        # Second pass: batch custom validation if we have valid texts
        custom_validation_results = []
        if cleaned_texts and self.custom_validator:
            try:
                logger.debug(f"Running batch validation on {len(cleaned_texts)} stories")
                custom_validation_results = await self.custom_validator.validate_batch(cleaned_texts)
            except Exception as e:
                logger.warning(f"Batch custom validation failed, falling back to individual: {e}")
                # Fallback to individual validation
                for cleaned_text in cleaned_texts:
                    try:
                        result = await self.custom_validator.validate(cleaned_text)
                        custom_validation_results.append(result)
                    except Exception as individual_e:
                        logger.warning(f"Individual validation also failed: {individual_e}")
                        # Create a failed validation result
                        from ..validation.base import CustomValidationResult
                        failed_result = CustomValidationResult(
                            is_valid=False,
                            score=0.0,
                            details={"error": str(individual_e)},
                            reasoning=f"Validation failed: {str(individual_e)}"
                        )
                        custom_validation_results.append(failed_result)

        # Third pass: create stories for texts that passed all validation
        if self.custom_validator and custom_validation_results:
            # Process with custom validation results
            for i, (cleaned_text, validation_result) in enumerate(zip(cleaned_texts, custom_validation_results)):
                original_index = valid_indices[i]
                prompt = prompts[original_index]
                generated_text = generated_texts[original_index]

                # Check custom validation result
                if not validation_result.is_valid:
                    logger.debug(f"Custom validation failed for {prompt.prompt_id}: {validation_result.reasoning}")
                    continue

                try:
                    # Create the story object
                    story = await self._create_story_object(
                        prompt, generated_text, cleaned_text,
                        generation_time / len(prompts),
                        memory_used / len(prompts)
                    )

                    if story:
                        stories.append(story)
                        total_tokens += story.tokens_generated

                except Exception as e:
                    logger.error(f"Failed to create story object for {prompt.prompt_id}: {e}")
        else:
            # Process without custom validation (only traditional validation was applied)
            for i, cleaned_text in enumerate(cleaned_texts):
                original_index = valid_indices[i]
                prompt = prompts[original_index]
                generated_text = generated_texts[original_index]

                try:
                    # Create the story object
                    story = await self._create_story_object(
                        prompt, generated_text, cleaned_text,
                        generation_time / len(prompts),
                        memory_used / len(prompts)
                    )

                    if story:
                        stories.append(story)
                        total_tokens += story.tokens_generated

                except Exception as e:
                    logger.error(f"Failed to create story object for {prompt.prompt_id}: {e}")

        logger.info(f"Batch validation completed: {len(stories)}/{len(prompts)} stories passed all validation")
        return stories, total_tokens

    async def _create_story_object(self, prompt, generated_text, cleaned_text, generation_time, memory_used):
        """Create a GeneratedStory object from validated content.

        Args:
            prompt: StoryPrompt object
            generated_text: Original generated text
            cleaned_text: Cleaned text that passed validation
            generation_time: Generation time for this story
            memory_used: Memory used for this story

        Returns:
            GeneratedStory object or None if creation fails
        """
        try:
            import uuid
            from ...common.utils import count_words

            # Count words and tokens
            word_count = count_words(cleaned_text)

            # Count tokens if tokenizer is available, otherwise estimate
            if hasattr(self.llm_provider, 'tokenizer') and self.llm_provider.tokenizer:
                try:
                    tokens_generated = len(self.llm_provider.tokenizer.encode(cleaned_text))
                except Exception:
                    # Fallback estimation: roughly 4 characters per token
                    tokens_generated = len(cleaned_text) // 4
            else:
                # Fallback estimation: roughly 4 characters per token
                tokens_generated = len(cleaned_text) // 4

            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0.0

            # Prepare k-shot examples for metadata
            k_shot_examples_data = []
            for example in prompt.k_shot_examples:
                k_shot_examples_data.append({
                    "role": example.role,
                    "content": example.content
                })

            # Create story object with enhanced metadata
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
                    "template_version": prompt.metadata.get("template_version", "unknown"),
                    "full_prompt": prompt.full_prompt,
                    "template": prompt.template,
                    "k_shot_examples": k_shot_examples_data,
                    "multi_turn_conversation": {
                        "messages": k_shot_examples_data + [{
                            "role": "user",
                            "content": prompt.full_prompt
                        }],
                        "total_messages": len(k_shot_examples_data) + 1,
                        "conversation_format": "openai_chat_format"
                    }
                }
            )

            return story

        except Exception as e:
            logger.error(f"Failed to create story object: {e}")
            return None
    
    async def process_multiple_batches(self,
                                     all_prompts: List[StoryPrompt],
                                     progress_callback: Optional[callable] = None,
                                     max_retries: int = 3,
                                     retry_delay: float = 5.0) -> List[GenerationResult]:
        """Process multiple batches of prompts with retry logic.

        Args:
            all_prompts: All prompts to process
            progress_callback: Optional callback for progress updates
            max_retries: Maximum number of retries for failed batches
            retry_delay: Delay in seconds between retries

        Returns:
            List of GenerationResult objects
        """
        batch_size = self.generation_config.batch_size
        results = []
        failed_batches = []

        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(all_prompts) + batch_size - 1)//batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches}")

            success = False
            for attempt in range(max_retries + 1):
                try:
                    result = await self.process_batch(batch_prompts)
                    results.append(result)
                    success = True

                    if progress_callback:
                        progress_callback(i + len(batch_prompts), len(all_prompts))

                    # Clear memory between batches
                    self.llm_provider.clear_memory()
                    break

                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Batch {batch_num} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying batch {batch_num} in {retry_delay} seconds...")

                        # Clear memory before retry
                        self.llm_provider.clear_memory()

                        # Wait before retry
                        import asyncio
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Batch {batch_num} failed after {max_retries + 1} attempts: {e}")
                        failed_batches.append({
                            'batch_num': batch_num,
                            'prompts': batch_prompts,
                            'error': str(e)
                        })

            if not success:
                logger.error(f"Skipping batch {batch_num} after all retry attempts failed")

        if failed_batches:
            logger.warning(f"Total failed batches: {len(failed_batches)}")
            for failed_batch in failed_batches:
                logger.warning(f"Failed batch {failed_batch['batch_num']}: {failed_batch['error']}")

        return results

    async def process_batch_with_ids(self, prompts: List[StoryPrompt], batch_id: int) -> GenerationResult:
        """Process a batch of prompts with batch ID for story naming.

        Args:
            prompts: List of StoryPrompt objects
            batch_id: Batch identifier

        Returns:
            GenerationResult with generated stories and metadata
        """
        # Use the regular process_batch but update story IDs afterwards
        result = await self.process_batch(prompts)

        # Update story IDs to include batch and position information
        for i, story in enumerate(result.stories):
            # Extract position from prompt_id if available
            if "pos_" in story.prompt_id:
                # Keep the existing format from prompt_id
                story_id = f"story_{story.prompt_id}"
            else:
                # Fallback format
                story_id = f"story_batch_{batch_id:03d}_pos_{i:03d}_{story.story_id.split('_')[-1]}"

            # Update the story_id
            story.story_id = story_id

        return result

    async def process_batch_with_adaptive_size(self, prompts: List[StoryPrompt], batch_id: int = 0) -> GenerationResult:
        """Process batch with adaptive batch size to handle memory constraints.

        Args:
            prompts: List of StoryPrompt objects
            batch_id: Batch identifier for story IDs

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

        # Try with full batch first
        try:
            return await self.process_batch_with_ids(prompts, batch_id)
        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's a memory-related error
            if any(keyword in error_msg for keyword in ['cuda out of memory', 'out of memory', 'memory']):
                logger.warning(f"Memory error detected, trying with smaller batch sizes: {e}")

                # Clear memory before retrying
                self.llm_provider.clear_memory()

                # Try with progressively smaller batch sizes
                for reduction_factor in [2, 4, 8]:
                    smaller_batch_size = max(1, len(prompts) // reduction_factor)
                    logger.info(f"Retrying with batch size {smaller_batch_size} (reduced by factor of {reduction_factor})")

                    try:
                        # Process in smaller sub-batches
                        all_stories = []
                        total_time = 0.0
                        total_tokens = 0

                        for i in range(0, len(prompts), smaller_batch_size):
                            sub_batch = prompts[i:i + smaller_batch_size]
                            sub_result = await self.process_batch_with_ids(sub_batch, batch_id)

                            all_stories.extend(sub_result.stories)
                            total_time += sub_result.total_generation_time
                            total_tokens += sum(story.tokens_generated for story in sub_result.stories)

                            # Clear memory between sub-batches
                            self.llm_provider.clear_memory()

                        # Calculate combined metrics
                        success_rate = len(all_stories) / len(prompts) if prompts else 0.0
                        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0.0

                        return GenerationResult(
                            stories=all_stories,
                            total_generation_time=total_time,
                            average_tokens_per_second=avg_tokens_per_second,
                            success_rate=success_rate,
                            metadata={
                                "batch_size": len(prompts),
                                "successful_generations": len(all_stories),
                                "failed_generations": len(prompts) - len(all_stories),
                                "adaptive_batch_size": smaller_batch_size,
                                "reduction_factor": reduction_factor,
                                "model_name": self.llm_provider.model_name
                            }
                        )

                    except Exception as sub_e:
                        logger.warning(f"Batch size {smaller_batch_size} also failed: {sub_e}")
                        continue

                # If all smaller batch sizes failed, raise the original error
                logger.error("All adaptive batch sizes failed")
                raise e
            else:
                # Not a memory error, re-raise
                raise e

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
