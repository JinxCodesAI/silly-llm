"""Prompt generation with k-shot examples and vocabulary selection."""

import random
from typing import List, Dict, Any, Optional
import logging

from ...common.data_models import Vocabulary, StoryPrompt, ConversationExample, KShotExample
from ...common.utils import parse_conversation_examples
from .template_manager import TemplateManager

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generates prompts for story generation with k-shot examples."""
    
    def __init__(self, 
                 vocabulary: Vocabulary,
                 template_manager: TemplateManager,
                 conversation_examples_path: Optional[str] = None,
                 k_shot_count: int = 2):
        """Initialize prompt generator.
        
        Args:
            vocabulary: Vocabulary object with words
            template_manager: Template manager for formatting prompts
            conversation_examples_path: Path to conversation examples file
            k_shot_count: Number of k-shot examples to include (0 for no examples)
        """
        self.vocabulary = vocabulary
        self.template_manager = template_manager
        self.k_shot_count = k_shot_count
        self.conversation_examples = []
        
        if conversation_examples_path:
            try:
                self.conversation_examples = parse_conversation_examples(conversation_examples_path)
                logger.info(f"Loaded {len(self.conversation_examples)} conversation examples")
            except Exception as e:
                logger.warning(f"Failed to load conversation examples: {e}")
    
    def generate_prompts(self, count: int,
                        use_k_shot: bool = True,
                        ensure_diversity: bool = True,
                        batch_id: Optional[int] = None) -> List[StoryPrompt]:
        """Generate a batch of story prompts.

        Args:
            count: Number of prompts to generate
            use_k_shot: Whether to include k-shot examples
            ensure_diversity: Whether to ensure word diversity across prompts
            batch_id: Optional batch identifier for prompt IDs

        Returns:
            List of StoryPrompt objects
        """
        prompts = []
        used_word_combinations = set()
        
        for i in range(count):
            # Generate unique word combination if diversity is required
            if ensure_diversity:
                selected_words = self._get_diverse_words(used_word_combinations)
                used_word_combinations.add(tuple(selected_words.values()))
            else:
                selected_words = self.vocabulary.get_random_words()
            
            # Select k-shot examples if requested
            k_shot_examples = []
            if use_k_shot and self.k_shot_count > 0 and self.conversation_examples:
                k_shot_examples = self._select_k_shot_examples()
            
            # Create prompt with proper ID format and random additional condition
            if batch_id is not None:
                prompt_id = f"batch_{batch_id:03d}_prompt_{i:03d}"
            else:
                prompt_id = f"prompt_{i:06d}"

            prompt = self.template_manager.create_k_shot_prompt(
                selected_words=selected_words,
                k_shot_examples=k_shot_examples,
                additional_condition=None,  # Let template manager randomly select
                prompt_id=prompt_id
            )
            
            prompts.append(prompt)
        
        logger.info(f"Generated {len(prompts)} prompts with k-shot={use_k_shot}")
        return prompts
    
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
    
    def _select_k_shot_examples(self) -> List[KShotExample]:
        """Select k-shot examples from conversation examples.
        
        Returns:
            List of KShotExample objects
        """
        if not self.conversation_examples:
            return []
        
        # Select random conversation examples
        num_conversations = min(self.k_shot_count, len(self.conversation_examples))
        selected_conversations = random.sample(self.conversation_examples, num_conversations)
        
        # Flatten to individual examples
        examples = []
        for conversation in selected_conversations:
            examples.extend(conversation.messages)
        
        return examples
    
    def create_single_prompt(self, 
                           word1: str, word2: str, word3: str,
                           additional_condition: Optional[str] = None,
                           use_k_shot: bool = True) -> StoryPrompt:
        """Create a single prompt with specific words.
        
        Args:
            word1: First word
            word2: Second word  
            word3: Third word
            additional_condition: Optional additional condition
            use_k_shot: Whether to include k-shot examples
            
        Returns:
            StoryPrompt object
        """
        selected_words = {
            "word1": word1,
            "word2": word2,
            "word3": word3
        }
        
        # Select k-shot examples if requested
        k_shot_examples = []
        if use_k_shot and self.k_shot_count > 0 and self.conversation_examples:
            k_shot_examples = self._select_k_shot_examples()
        
        return self.template_manager.create_k_shot_prompt(
            selected_words=selected_words,
            k_shot_examples=k_shot_examples,
            additional_condition=additional_condition
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the prompt generator.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "vocabulary_size": {
                "nouns": len(self.vocabulary.nouns),
                "verbs": len(self.vocabulary.verbs),
                "adjectives": len(self.vocabulary.adjectives),
                "total": len(self.vocabulary.nouns) + len(self.vocabulary.verbs) + len(self.vocabulary.adjectives)
            },
            "conversation_examples": len(self.conversation_examples),
            "k_shot_count": self.k_shot_count,
            "available_features": len(self.template_manager.get_available_features()),
            "total_possible_combinations": len(self.vocabulary.nouns) * len(self.vocabulary.verbs) * len(self.vocabulary.adjectives)
        }
