"""Template management for story generation."""

import random
from typing import List, Dict, Any, Optional
from pathlib import Path

from ...common.data_models import StoryPrompt, KShotExample
from ...common.utils import load_story_features


class TemplateManager:
    """Manages story generation templates and formatting."""
    
    def __init__(self, story_features_path: Optional[str] = None):
        """Initialize template manager.
        
        Args:
            story_features_path: Path to story features JSON file
        """
        self.base_template = (
            "Generate simple, short (up to 150 words) bed time story written entirely in English, easy to understand and follow by 3 years old who knows only English\n"
            "containing 3 English words {word1} {word2} {word3}\n\n"
            "{additional_condition}\n\n"
            "keep story coherent and gramatically correct, write full content of the story and nothing else (no commentary, title, etc) start with 'STORY:\n'"
        )
        
        self.story_features = []
        if story_features_path and Path(story_features_path).exists():
            self.story_features = load_story_features(story_features_path)
    
    def create_prompt(self, 
                     selected_words: Dict[str, str],
                     additional_condition: Optional[str] = None,
                     prompt_id: Optional[str] = None) -> StoryPrompt:
        """Create a story prompt from template and words.
        
        Args:
            selected_words: Dictionary with word1, word2, word3
            additional_condition: Optional additional condition for the story
            prompt_id: Optional prompt identifier
            
        Returns:
            StoryPrompt object
        """
        if prompt_id is None:
            prompt_id = f"prompt_{random.randint(100000, 999999)}"
        
        # Select random additional condition if not provided
        if additional_condition is None:
            if self.story_features:
                additional_condition = random.choice(self.story_features)
            else:
                additional_condition = ""
        
        # Format the template
        full_prompt = self.base_template.format(
            word1=selected_words.get("word1", ""),
            word2=selected_words.get("word2", ""),
            word3=selected_words.get("word3", ""),
            additional_condition=additional_condition
        )
        
        return StoryPrompt(
            prompt_id=prompt_id,
            template=self.base_template,
            selected_words=selected_words,
            additional_condition=additional_condition,
            full_prompt=full_prompt,
            metadata={
                "template_version": "1.0",
                "has_additional_condition": bool(additional_condition.strip())
            }
        )
    
    def create_k_shot_prompt(self,
                           selected_words: Dict[str, str],
                           k_shot_examples: List[KShotExample],
                           additional_condition: Optional[str] = None,
                           prompt_id: Optional[str] = None) -> StoryPrompt:
        """Create a k-shot prompt with examples.
        
        Args:
            selected_words: Dictionary with word1, word2, word3
            k_shot_examples: List of k-shot examples
            additional_condition: Optional additional condition
            prompt_id: Optional prompt identifier
            
        Returns:
            StoryPrompt object with k-shot examples
        """
        # Create base prompt
        prompt = self.create_prompt(selected_words, additional_condition, prompt_id)
        
        # Add k-shot examples
        prompt.k_shot_examples = k_shot_examples
        prompt.metadata["k_shot_count"] = len(k_shot_examples)
        
        return prompt
    
    def format_for_chat_template(self, prompt: StoryPrompt) -> List[Dict[str, str]]:
        """Format prompt for chat template usage.
        
        Args:
            prompt: StoryPrompt object
            
        Returns:
            List of message dictionaries for chat template
        """
        messages = []
        
        # Add k-shot examples if present
        for example in prompt.k_shot_examples:
            messages.append({
                "role": example.role,
                "content": example.content
            })
        
        # Add the actual prompt
        messages.append({
            "role": "user",
            "content": prompt.full_prompt
        })
        
        return messages
    
    def get_available_features(self) -> List[str]:
        """Get list of available story features."""
        return self.story_features.copy()
    
    def add_custom_feature(self, feature: str):
        """Add a custom story feature."""
        if feature not in self.story_features:
            self.story_features.append(feature)
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about the template manager."""
        return {
            "base_template": self.base_template,
            "available_features": len(self.story_features),
            "features": self.story_features
        }
