"""Data models for the synthetic data generation pipeline."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class WordCategory(Enum):
    """Categories of words in vocabulary."""
    NOUN = "nouns"
    VERB = "verbs" 
    ADJECTIVE = "adjectives"


class GenerationConfig(BaseModel):
    """Configuration for story generation."""
    batch_size: int = Field(default=8, description="Batch size for generation")
    max_new_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    use_cache: bool = Field(default=True, description="Whether to use KV cache")


class StoryPrompt(BaseModel):
    """Represents a story generation prompt."""
    prompt_id: str = Field(description="Unique identifier for the prompt")
    template: str = Field(description="Template used for generation")
    selected_words: Dict[str, str] = Field(description="Selected words for the story")
    additional_condition: Optional[str] = Field(default=None, description="Additional story condition")
    full_prompt: str = Field(description="Complete formatted prompt")
    k_shot_examples: List['KShotExample'] = Field(default_factory=list, description="K-shot examples")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GeneratedStory(BaseModel):
    """Represents a generated story."""
    story_id: str = Field(description="Unique identifier for the story")
    prompt_id: str = Field(description="ID of the prompt used")
    content: str = Field(description="Generated story content")
    word_count: int = Field(description="Number of words in the story")
    generation_time: float = Field(description="Time taken to generate (seconds)")
    tokens_generated: int = Field(description="Number of tokens generated")
    tokens_per_second: float = Field(description="Generation speed")
    memory_used_gb: float = Field(description="Memory used during generation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KShotExample(BaseModel):
    """Represents a k-shot example for prompting."""
    role: str = Field(description="Role (user/assistant)")
    content: str = Field(description="Content of the message")


class LLMMessage(BaseModel):
    """Individual message in a conversation."""
    role: str = Field(description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")


class LLMRequest(BaseModel):
    """Request structure for LLM providers."""
    messages: List[LLMMessage] = Field(description="Conversation messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")

    @classmethod
    def from_story_prompt(cls, prompt: 'StoryPrompt') -> "LLMRequest":
        """Create LLMRequest from StoryPrompt with k-shot examples."""
        messages = []

        # Add k-shot examples in order
        for example in prompt.k_shot_examples:
            messages.append(LLMMessage(role=example.role, content=example.content))

        # Add the current prompt as user message
        messages.append(LLMMessage(role="user", content=prompt.full_prompt))

        return cls(
            messages=messages,
            metadata={
                "prompt_id": prompt.prompt_id,
                "selected_words": prompt.selected_words,
                "additional_condition": prompt.additional_condition,
                "k_shot_count": len(prompt.k_shot_examples)
            }
        )

    def to_simple_prompt(self) -> str:
        """Fallback conversion to simple string for legacy support."""
        if len(self.messages) == 1 and self.messages[0].role == "user":
            return self.messages[0].content

        # Format as simple conversation
        parts = []
        for msg in self.messages:
            if msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            elif msg.role == "system":
                parts.append(f"System: {msg.content}")

        return "\n\n".join(parts)


class ConversationExample(BaseModel):
    """Represents a complete conversation example."""
    messages: List[KShotExample] = Field(description="List of messages in the conversation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Example metadata")


class Vocabulary(BaseModel):
    """Represents the vocabulary for story generation."""
    nouns: List[str] = Field(description="List of nouns")
    verbs: List[str] = Field(description="List of verbs") 
    adjectives: List[str] = Field(description="List of adjectives")
    
    def get_words_by_category(self, category: WordCategory) -> List[str]:
        """Get words by category."""
        return getattr(self, category.value)
    
    def get_random_words(self, count: int = 3) -> Dict[str, str]:
        """Get random words for story generation."""
        import random
        
        # Select one word from each category
        word1 = random.choice(self.nouns)
        word2 = random.choice(self.verbs) 
        word3 = random.choice(self.adjectives)
        
        return {
            "word1": word1,
            "word2": word2, 
            "word3": word3
        }


class GenerationResult(BaseModel):
    """Result of a generation batch."""
    stories: List[GeneratedStory] = Field(description="Generated stories")
    total_generation_time: float = Field(description="Total time for batch")
    average_tokens_per_second: float = Field(description="Average generation speed")
    success_rate: float = Field(description="Percentage of successful generations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")


class ValidationResult(BaseModel):
    """Result of story validation."""
    is_valid: bool = Field(description="Whether the story passes validation")
    word_count: int = Field(description="Number of words in story")
    contains_required_words: bool = Field(description="Whether required words are present")
    issues: List[str] = Field(default_factory=list, description="List of validation issues")
    score: float = Field(description="Overall quality score (0-1)")
