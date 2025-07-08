"""Common utilities for the training pipeline."""

import json
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .data_models import Vocabulary, ConversationExample, KShotExample, ValidationResult

logger = logging.getLogger(__name__)


def load_vocabulary(file_path: str) -> Vocabulary:
    """Load vocabulary from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Vocabulary(**data)
    except Exception as e:
        logger.error(f"Failed to load vocabulary from {file_path}: {e}")
        raise


def load_story_features(file_path: str) -> List[str]:
    """Load story features from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        if not isinstance(features, list):
            raise ValueError("Story features must be a list")
        
        return features
    except Exception as e:
        logger.error(f"Failed to load story features from {file_path}: {e}")
        raise


def parse_conversation_examples(file_path: str) -> List[ConversationExample]:
    """Parse conversation examples from text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        examples = []
        
        # Split by PROMPT/RESPONSE pairs
        sections = re.split(r'\n(?=PROMPT:|RESPONSE:)', content.strip())
        
        current_conversation = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if section.startswith('PROMPT:'):
                # If we have a previous conversation, save it
                if current_conversation and len(current_conversation) >= 2:
                    examples.append(ConversationExample(messages=current_conversation))
                
                # Start new conversation
                prompt_content = section[7:].strip()  # Remove "PROMPT:"
                current_conversation = [KShotExample(role="user", content=prompt_content)]
                
            elif section.startswith('RESPONSE:'):
                response_content = section[9:].strip()  # Remove "RESPONSE:"
                if current_conversation:
                    current_conversation.append(KShotExample(role="assistant", content=response_content))
        
        # Add the last conversation if complete
        if current_conversation and len(current_conversation) >= 2:
            examples.append(ConversationExample(messages=current_conversation))
        
        logger.info(f"Parsed {len(examples)} conversation examples from {file_path}")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to parse conversation examples from {file_path}: {e}")
        raise


def count_words(text: str) -> int:
    """Count words in text."""
    # Simple word counting - split by whitespace and filter empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)


def validate_story(story_content: str, required_words: List[str], 
                  min_words: int = 50, max_words: int = 300) -> ValidationResult:
    """Validate a generated story."""
    issues = []
    
    # Count words
    word_count = count_words(story_content)
    
    # Check word count
    if word_count < min_words:
        issues.append(f"Story too short: {word_count} words (minimum: {min_words})")
    elif word_count > max_words:
        issues.append(f"Story too long: {word_count} words (maximum: {max_words})")
    
    # Check for required words
    story_lower = story_content.lower()
    missing_words = []
    
    for word in required_words:
        if word.lower() not in story_lower:
            missing_words.append(word)
    
    contains_required_words = len(missing_words) == 0
    
    if missing_words:
        issues.append(f"Missing required words: {', '.join(missing_words)}")
    
    # Calculate overall score
    score = 1.0
    if word_count < min_words or word_count > max_words:
        score -= 0.3
    if missing_words:
        score -= 0.5 * (len(missing_words) / len(required_words))
    
    score = max(0.0, score)
    
    is_valid = len(issues) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        word_count=word_count,
        contains_required_words=contains_required_words,
        issues=issues,
        score=score
    )


def clean_generated_text(text: str) -> str:
    """Clean and normalize generated text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove any potential artifacts
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML-like tags
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
    
    return text.strip()


def save_stories_jsonl(stories: List[Dict[str, Any]], output_path: str):
    """Save stories to JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for story in stories:
            f.write(json.dumps(story, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(stories)} stories to {output_path}")


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
