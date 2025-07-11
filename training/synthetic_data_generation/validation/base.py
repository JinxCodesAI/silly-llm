"""Base classes for custom validation system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ...common.llm_providers import LLMProvider
from ...common.data_models import GenerationConfig


class CustomValidationResult(BaseModel):
    """Result of custom validation."""
    is_valid: bool = Field(description="Whether the story passes validation")
    score: float = Field(description="Validation score (0-1)", ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed validation results")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for the validation result")


class BaseValidator(ABC):
    """Abstract base class for custom validators."""
    
    def __init__(self, provider: LLMProvider, config: Dict[str, Any]):
        """Initialize validator with LLM provider and configuration.

        Args:
            provider: LLM provider instance for validation
            config: Validator-specific configuration
        """
        self.provider = provider
        self.config = config

        # Extract generation config or use defaults
        generation_params = config.get('generation', {})
        self.generation_config = GenerationConfig(
            max_new_tokens=generation_params.get('max_new_tokens', 128),
            temperature=generation_params.get('temperature', 0.1),
            top_p=generation_params.get('top_p', 0.9),
            do_sample=generation_params.get('do_sample', True),
            repetition_penalty=generation_params.get('repetition_penalty', 1.0)
        )
        
    @abstractmethod
    def get_validation_prompt(self, story_content: str) -> str:
        """Generate the validation prompt for the LLM.

        Args:
            story_content: The story content to validate

        Returns:
            Formatted prompt for validation
        """
        pass

    @abstractmethod
    def parse_validation_response(self, response: str) -> CustomValidationResult:
        """Parse the LLM response into a CustomValidationResult.

        This method should be implemented by each validator to handle
        their specific response format and parsing logic.

        Args:
            response: Raw response from LLM

        Returns:
            CustomValidationResult parsed from response
        """
        pass

    async def validate(self, story_content: str) -> CustomValidationResult:
        """Validate a story using the LLM provider.

        This method handles the common provider interaction logic.
        Subclasses should implement get_validation_prompt() and parse_validation_response().

        Args:
            story_content: The generated story content to validate

        Returns:
            CustomValidationResult with validation outcome
        """
        try:
            # Create validation prompt using subclass implementation
            validation_prompt = self.get_validation_prompt(story_content)

            # Create LLM request
            from ...common.data_models import LLMRequest, LLMMessage
            request = LLMRequest(
                messages=[
                    LLMMessage(role="user", content=validation_prompt)
                ]
            )

            # Generate validation response
            responses = await self.provider.generate_batch([request], self.generation_config)

            if not responses:
                return CustomValidationResult(
                    is_valid=False,
                    score=0.0,
                    details={"error": "No response from validation provider"},
                    reasoning="Validation failed: no response received"
                )

            response = responses[0]

            # Parse the response using subclass implementation
            result = self.parse_validation_response(response)

            # Add common details
            result.details.update({
                "story_length": len(story_content),
                "validation_prompt": validation_prompt,
                "provider_model": self.provider.model_name
            })

            return result

        except Exception as e:
            return CustomValidationResult(
                is_valid=False,
                score=0.0,
                details={"error": str(e)},
                reasoning=f"Validation failed with error: {str(e)}"
            )
