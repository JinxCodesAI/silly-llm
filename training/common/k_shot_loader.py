"""K-shot example loader supporting JSON and text formats."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

from .data_models import KShotConfiguration, KShotSource, KShotExample, ConversationExample
from .utils import parse_conversation_examples

logger = logging.getLogger(__name__)


class KShotLoader:
    """Unified loader for k-shot examples from multiple sources."""
    
    def __init__(self):
        """Initialize the k-shot loader."""
        self.configurations: List[KShotConfiguration] = []
        self.sample_selector: Optional[Callable] = None
        
    def load_from_json(self, file_path: str) -> List[KShotConfiguration]:
        """Load k-shot configurations from JSON file.
        
        Args:
            file_path: Path to JSON file containing k-shot configurations
            
        Returns:
            List of KShotConfiguration objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"K-shot configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse the JSON structure directly
            configurations = []
            for sample in data.get('samples', []):
                # Convert messages to KShotExample objects
                k_shot_examples = []
                for msg in sample.get('messages', []):
                    k_shot_examples.append(KShotExample(
                        role=msg.get('role', ''),
                        content=msg.get('content', '')
                    ))

                config = KShotConfiguration(
                    name=sample.get('name', ''),
                    k_shot_count=sample.get('k_shot_count', 0),
                    messages=k_shot_examples,
                    metadata=sample.get('metadata', {})
                )
                configurations.append(config)
                
            self.configurations = configurations
            logger.info(f"Loaded {len(configurations)} k-shot configurations from JSON: {file_path}")
            return configurations
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse k-shot configuration from {file_path}: {e}")
    
    def load_from_text(self, file_path: str) -> List[KShotConfiguration]:
        """Load k-shot configurations from legacy text format.
        
        Args:
            file_path: Path to text file with PROMPT:/RESPONSE: format
            
        Returns:
            List of KShotConfiguration objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If text format is invalid
        """
        try:
            # Use existing parser from utils
            conversation_examples = parse_conversation_examples(file_path)
            
            configurations = []
            for i, example in enumerate(conversation_examples):
                config = KShotConfiguration(
                    name=f"text_example_{i+1}",
                    k_shot_count=len(example.messages) // 2,  # Pairs of user/assistant
                    messages=example.messages,
                    metadata={"source": "text_file", "file_path": file_path}
                )
                configurations.append(config)
                
            self.configurations = configurations
            logger.info(f"Loaded {len(configurations)} k-shot configurations from text: {file_path}")
            return configurations
            
        except Exception as e:
            raise ValueError(f"Failed to parse text k-shot configuration from {file_path}: {e}")
    
    def get_configuration(self, name: Optional[str] = None) -> Optional[KShotConfiguration]:
        """Get specific configuration by name or default.
        
        Args:
            name: Configuration name to retrieve, or None for first available
            
        Returns:
            KShotConfiguration object or None if not found
        """
        if not self.configurations:
            logger.warning("No k-shot configurations loaded")
            return None
            
        if name is None:
            # Return first configuration as default
            return self.configurations[0]
            
        # Search for configuration by name
        for config in self.configurations:
            if config.name == name:
                return config
                
        logger.warning(f"K-shot configuration '{name}' not found. Available: {[c.name for c in self.configurations]}")
        return None
    
    def list_configurations(self) -> List[str]:
        """List all available configuration names.
        
        Returns:
            List of configuration names
        """
        return [config.name for config in self.configurations]
    
    def set_sample_selector(self, selector_func: Callable[[str, List[KShotConfiguration]], KShotConfiguration]):
        """Set custom sample selector function.
        
        Args:
            selector_func: Function that takes (prompt, configurations) and returns selected configuration
        """
        self.sample_selector = selector_func
        logger.info("Custom sample selector function set")
    
    def select_sample_for_prompt(self, prompt: str, configurations: Optional[List[KShotConfiguration]] = None) -> Optional[KShotConfiguration]:
        """Select k-shot sample based on current prompt.
        
        Args:
            prompt: Current prompt text
            configurations: Optional list of configurations to choose from
            
        Returns:
            Selected KShotConfiguration or None
        """
        configs_to_use = configurations or self.configurations
        
        if not configs_to_use:
            logger.warning("No k-shot configurations available for selection")
            return None
            
        # Use custom selector if available
        if self.sample_selector:
            try:
                return self.sample_selector(prompt, configs_to_use)
            except Exception as e:
                logger.error(f"Custom sample selector failed: {e}, falling back to default")
        
        # Default implementation: always pick first sample
        return configs_to_use[0]


def default_sample_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Default sample selector - always returns first configuration.
    
    Args:
        prompt: Current prompt text (unused in default implementation)
        configurations: Available configurations
        
    Returns:
        First configuration in the list
    """
    return configurations[0]


def random_sample_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Random sample selector - returns random configuration.
    
    Args:
        prompt: Current prompt text (unused)
        configurations: Available configurations
        
    Returns:
        Randomly selected configuration
    """
    import random
    return random.choice(configurations)


def keyword_based_selector(keywords_map: Dict[str, str]) -> Callable[[str, List[KShotConfiguration]], KShotConfiguration]:
    """Create keyword-based sample selector.
    
    Args:
        keywords_map: Dictionary mapping keywords to configuration names
        
    Returns:
        Selector function that chooses based on keywords in prompt
    """
    def selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
        prompt_lower = prompt.lower()
        
        # Check for keywords in prompt
        for keyword, config_name in keywords_map.items():
            if keyword.lower() in prompt_lower:
                # Find configuration by name
                for config in configurations:
                    if config.name == config_name:
                        return config
        
        # Fallback to first configuration
        return configurations[0]
    
    return selector
