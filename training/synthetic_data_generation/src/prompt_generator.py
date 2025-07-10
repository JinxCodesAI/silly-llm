"""Prompt generation with k-shot examples and vocabulary selection."""

import random
from typing import List, Dict, Any, Optional, Callable
import logging

from ...common.data_models import Vocabulary, StoryPrompt, ConversationExample, KShotExample, KShotConfiguration
from ...common.utils import parse_conversation_examples
from ...common.k_shot_loader import KShotLoader
from .template_manager import TemplateManager

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generates prompts for story generation with k-shot examples."""

    def __init__(self,
                 vocabulary: Vocabulary,
                 template_manager: TemplateManager,
                 conversation_examples_path: Optional[str] = None,
                 k_shot_config_file: Optional[str] = None,
                 k_shot_config_name: Optional[str] = None,
                 k_shot_count: int = 2,
                 sample_selector: Optional[Callable] = None,
                 k_shot_settings: Optional[Dict[str, Any]] = None):
        """Initialize prompt generator.

        Args:
            vocabulary: Vocabulary object with words
            template_manager: Template manager for formatting prompts
            conversation_examples_path: Path to legacy text conversation examples file
            k_shot_config_file: Path to JSON k-shot configuration file
            k_shot_config_name: Name of specific k-shot configuration to use
            k_shot_count: Number of k-shot examples to include (0 for no examples)
            sample_selector: Custom function for selecting k-shot samples
            k_shot_settings: Dictionary with k-shot configuration settings
        """
        self.vocabulary = vocabulary
        self.template_manager = template_manager
        self.k_shot_count = k_shot_count
        self.conversation_examples = []
        self.k_shot_loader = KShotLoader()
        self.k_shot_config_name = k_shot_config_name

        # Set custom sample selector if provided
        if sample_selector:
            self.k_shot_loader.set_sample_selector(sample_selector)
        elif k_shot_settings:
            # Load selector from k_shot_settings
            selector = self._load_selector_from_settings(k_shot_settings)
            if selector:
                self.k_shot_loader.set_sample_selector(selector)

        # Load k-shot examples from JSON file (preferred)
        if k_shot_config_file:
            try:
                self.k_shot_loader.load_from_json(k_shot_config_file)
                logger.info(f"Loaded k-shot configurations from JSON: {k_shot_config_file}")
                if k_shot_config_name:
                    config = self.k_shot_loader.get_configuration(k_shot_config_name)
                    if config:
                        logger.info(f"Using k-shot configuration: {k_shot_config_name}")
                    else:
                        logger.warning(f"Configuration '{k_shot_config_name}' not found, will use default")
            except Exception as e:
                logger.error(f"Failed to load JSON k-shot configuration: {e}")

        # Fallback to legacy text format
        elif conversation_examples_path:
            try:
                self.conversation_examples = parse_conversation_examples(conversation_examples_path)
                logger.info(f"Loaded {len(self.conversation_examples)} conversation examples from text file")
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
            if use_k_shot and self.k_shot_count > 0:
                k_shot_examples = self._select_k_shot_examples(selected_words)
            
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
    
    def _select_k_shot_examples(self, selected_words: Dict[str, str]) -> List[KShotExample]:
        """Select k-shot examples using JSON configuration or legacy text format.

        Args:
            selected_words: Dictionary with selected words for context

        Returns:
            List of KShotExample objects
        """
        # Try JSON-based k-shot configuration first
        if self.k_shot_loader.configurations:
            # Create a simple prompt for sample selection
            prompt_text = f"Story with words: {', '.join(selected_words.values())}"

            # Select configuration using custom selector or default
            config = self.k_shot_loader.select_sample_for_prompt(prompt_text)
            if config:
                # Use the selected configuration's messages
                return config.messages[:self.k_shot_count * 2]  # Limit to requested count (user+assistant pairs)

        # Fallback to legacy text-based conversation examples
        if self.conversation_examples:
            # Select random conversation examples
            num_conversations = min(self.k_shot_count, len(self.conversation_examples))
            selected_conversations = random.sample(self.conversation_examples, num_conversations)

            # Flatten to individual examples
            examples = []
            for conversation in selected_conversations:
                examples.extend(conversation.messages)

            return examples

        # No k-shot examples available
        return []

    def _load_selector_from_settings(self, k_shot_settings: Dict[str, Any]) -> Optional[Callable]:
        """Load sample selector function from k_shot_settings configuration.

        Args:
            k_shot_settings: Dictionary with k-shot configuration settings

        Returns:
            Selector function or None if loading fails
        """
        selector_type = k_shot_settings.get("selector_type", "default")

        try:
            if selector_type == "default":
                from ...common.k_shot_loader import default_sample_selector
                return default_sample_selector
            elif selector_type == "random":
                from ...common.k_shot_loader import random_sample_selector
                return random_sample_selector
            elif selector_type == "keyword":
                from ...common.k_shot_loader import keyword_based_selector
                keyword_mappings = k_shot_settings.get("keyword_mappings", {})
                if keyword_mappings:
                    return keyword_based_selector(keyword_mappings)
                else:
                    logger.warning("Keyword selector requested but no keyword_mappings provided")
                    return None
            elif selector_type == "custom":
                # Load custom selector function
                selector_module = k_shot_settings.get("selector_module")
                selector_function = k_shot_settings.get("selector_function")

                if not selector_module or not selector_function:
                    logger.error("Custom selector requires both selector_module and selector_function")
                    return None

                # Import the module and get the function
                import importlib
                module = importlib.import_module(selector_module)
                selector_func = getattr(module, selector_function)

                logger.info(f"Loaded custom selector: {selector_module}.{selector_function}")
                return selector_func
            else:
                logger.warning(f"Unknown selector type: {selector_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to load selector function: {e}")
            return None
    
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
        if use_k_shot and self.k_shot_count > 0:
            k_shot_examples = self._select_k_shot_examples(selected_words)
        
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
            "k_shot_configurations": len(self.k_shot_loader.configurations),
            "k_shot_config_names": self.k_shot_loader.list_configurations(),
            "k_shot_count": self.k_shot_count,
            "available_features": len(self.template_manager.get_available_features()),
            "total_possible_combinations": len(self.vocabulary.nouns) * len(self.vocabulary.verbs) * len(self.vocabulary.adjectives)
        }
