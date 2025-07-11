"""Quality validator implementation for English language validation."""

import asyncio
from typing import Dict, Any

from .base import BaseValidator, CustomValidationResult


class QualityValidator(BaseValidator):
    """Validator that checks if stories contain only English words."""
    
    def __init__(self, provider, config: Dict[str, Any]):
        """Initialize quality validator.

        Args:
            provider: LLM provider instance
            config: Configuration dictionary with generation parameters
        """
        super().__init__(provider, config)
    
    def get_validation_prompt(self, story_content: str) -> str:
        """Generate validation prompt for English language checking.
        
        Args:
            story_content: Story content to validate
            
        Returns:
            Formatted validation prompt
        """
        prompt = (
            "Does provided story contain only English words, think step by step and "
            "finish generation with either ANSWER:NO or ANSWER:YES\n\n"
            f"STORY TO ANALYZE: {story_content}"
        )
        return prompt
    

    
    def parse_validation_response(self, response: str) -> CustomValidationResult:
        """Parse validation response with enhanced logic for quality checking.

        Args:
            response: Raw LLM response

        Returns:
            CustomValidationResult with parsed outcome
        """
        response_clean = response.strip()
        response_upper = response_clean.upper()
        
        # Look for explicit ANSWER pattern first
        if "ANSWER:YES" in response_upper:
            return CustomValidationResult(
                is_valid=True,
                score=1.0,
                details={"pattern_found": "ANSWER:YES", "raw_response": response_clean},
                reasoning=response_clean
            )
        elif "ANSWER:NO" in response_upper:
            return CustomValidationResult(
                is_valid=False,
                score=0.0,
                details={"pattern_found": "ANSWER:NO", "raw_response": response_clean},
                reasoning=response_clean
            )
        
        # Fallback parsing for less structured responses
        positive_indicators = ["YES", "ENGLISH", "VALID", "CORRECT", "GOOD", "PASS"]
        negative_indicators = ["NO", "NON-ENGLISH", "INVALID", "INCORRECT", "BAD", "FAIL", "FOREIGN"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in response_upper)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_upper)
        
        if positive_count > negative_count:
            return CustomValidationResult(
                is_valid=True,
                score=0.7,  # Lower confidence for unclear responses
                details={
                    "pattern_found": "positive_indicators",
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "raw_response": response_clean
                },
                reasoning=response_clean
            )
        elif negative_count > positive_count:
            return CustomValidationResult(
                is_valid=False,
                score=0.3,
                details={
                    "pattern_found": "negative_indicators",
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "raw_response": response_clean
                },
                reasoning=response_clean
            )
        else:
            # Ambiguous response - default to invalid for safety
            return CustomValidationResult(
                is_valid=False,
                score=0.1,
                details={
                    "pattern_found": "ambiguous",
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "raw_response": response_clean
                },
                reasoning=f"Ambiguous validation response: {response_clean}"
            )
