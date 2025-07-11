"""Custom sample selector functions for k-shot prompting.

This module provides example implementations of custom sample selector functions
that can be used with the JSON k-shot configuration system.
"""

from typing import List
from training.common.data_models import KShotConfiguration


def theme_based_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Select k-shot configuration based on story theme detected in prompt.
    
    Args:
        prompt: The current prompt text
        configurations: Available k-shot configurations
        
    Returns:
        Selected KShotConfiguration
    """
    prompt_lower = prompt.lower()
    
    # Define theme keywords
    theme_keywords = {
        "animals": ["animal", "rabbit", "cat", "dog", "bird", "zoo", "farm", "pet"],
        "adventure": ["adventure", "treasure", "map", "journey", "explore", "brave", "quest"],
        "magic": ["magic", "wizard", "fairy", "spell", "enchanted", "magical", "wand"],
        "family": ["family", "mom", "dad", "sister", "brother", "grandma", "grandpa", "parent"],
        "friendship": ["friend", "friendship", "together", "help", "share", "kind"],
        "nature": ["forest", "tree", "flower", "garden", "river", "mountain", "nature"]
    }
    
    # Check for theme keywords in prompt
    for theme, keywords in theme_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            # Look for configuration with matching theme
            for config in configurations:
                if config.metadata.get("theme") == theme:
                    return config
    
    # Fallback to first configuration
    return configurations[0]


def difficulty_based_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Select k-shot configuration based on story complexity.
    
    Args:
        prompt: The current prompt text
        configurations: Available k-shot configurations
        
    Returns:
        Selected KShotConfiguration
    """
    prompt_lower = prompt.lower()
    
    # Detect complexity indicators
    if any(word in prompt_lower for word in ["simple", "easy", "basic", "short", "young"]):
        difficulty = "easy"
    elif any(word in prompt_lower for word in ["complex", "detailed", "long", "advanced", "older"]):
        difficulty = "hard"
    else:
        difficulty = "medium"
    
    # Find matching configuration
    for config in configurations:
        if config.metadata.get("difficulty") == difficulty:
            return config
    
    # Fallback to first configuration
    return configurations[0]


def length_based_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Select k-shot configuration based on desired story length.
    
    Args:
        prompt: The current prompt text
        configurations: Available k-shot configurations
        
    Returns:
        Selected KShotConfiguration
    """
    import re
    
    # Extract word count from prompt
    word_count_match = re.search(r'(\d+)\s*words?', prompt.lower())
    if word_count_match:
        target_words = int(word_count_match.group(1))
        
        if target_words <= 100:
            target_length = "short"
        elif target_words <= 200:
            target_length = "medium"
        else:
            target_length = "long"
        
        # Find matching configuration
        for config in configurations:
            if config.metadata.get("story_length") == target_length:
                return config
    
    # Check for length keywords
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["short", "brief", "quick"]):
        target_length = "short"
    elif any(word in prompt_lower for word in ["long", "detailed", "extended"]):
        target_length = "long"
    else:
        target_length = "medium"
    
    # Find matching configuration
    for config in configurations:
        if config.metadata.get("story_length") == target_length:
            return config
    
    # Fallback to first configuration
    return configurations[0]


def age_based_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Select k-shot configuration based on target age group.
    
    Args:
        prompt: The current prompt text
        configurations: Available k-shot configurations
        
    Returns:
        Selected KShotConfiguration
    """
    import re
    prompt_lower = prompt.lower()
    
    # Extract age from prompt
    age_match = re.search(r'(\d+)\s*years?\s*old', prompt_lower)
    if age_match:
        age = int(age_match.group(1))
        
        if age <= 3:
            target_age = "toddler"
        elif age <= 6:
            target_age = "preschool"
        elif age <= 10:
            target_age = "elementary"
        else:
            target_age = "older"
    else:
        # Check for age-related keywords
        if any(word in prompt_lower for word in ["toddler", "baby", "very young"]):
            target_age = "toddler"
        elif any(word in prompt_lower for word in ["preschool", "kindergarten", "young"]):
            target_age = "preschool"
        elif any(word in prompt_lower for word in ["elementary", "school age"]):
            target_age = "elementary"
        else:
            target_age = "general"
    
    # Find matching configuration
    for config in configurations:
        config_age = config.metadata.get("target_age", "")
        if target_age in config_age or config_age in target_age:
            return config
    
    # Fallback to first configuration
    return configurations[0]


def combined_selector(prompt: str, configurations: List[KShotConfiguration]) -> KShotConfiguration:
    """Advanced selector that combines multiple criteria.
    
    Args:
        prompt: The current prompt text
        configurations: Available k-shot configurations
        
    Returns:
        Selected KShotConfiguration
    """
    prompt_lower = prompt.lower()
    
    # Score each configuration based on multiple factors
    scores = {}
    
    for config in configurations:
        score = 0
        metadata = config.metadata
        
        # Theme matching (highest priority)
        theme = metadata.get("theme", "")
        if theme and theme in prompt_lower:
            score += 10
        
        # Difficulty matching
        difficulty = metadata.get("difficulty", "")
        if difficulty == "easy" and any(word in prompt_lower for word in ["simple", "easy", "young"]):
            score += 5
        elif difficulty == "hard" and any(word in prompt_lower for word in ["complex", "detailed", "older"]):
            score += 5
        elif difficulty == "medium":
            score += 3  # Medium is a good default
        
        # Length matching
        story_length = metadata.get("story_length", "")
        if story_length == "short" and any(word in prompt_lower for word in ["short", "brief"]):
            score += 3
        elif story_length == "long" and any(word in prompt_lower for word in ["long", "detailed"]):
            score += 3
        
        # Keyword matching
        keywords = metadata.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in prompt_lower:
                score += 2
        
        scores[config.name] = score
    
    # Select configuration with highest score
    if scores:
        best_config_name = max(scores, key=scores.get)
        for config in configurations:
            if config.name == best_config_name:
                return config
    
    # Fallback to first configuration
    return configurations[0]
