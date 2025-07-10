#!/usr/bin/env python3
"""Demo script showing the refactored validation system architecture."""

import asyncio
import logging
from typing import Dict, Any

from .validation.base import BaseValidator, CustomValidationResult
from .validation.quality_validator import QualityValidator
from ..common.llm_providers import MockLLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleYesNoValidator(BaseValidator):
    """Example of a simple custom validator that looks for yes/no responses."""
    
    def get_validation_prompt(self, story_content: str) -> str:
        """Generate a simple yes/no validation prompt."""
        return f"Is this a good bedtime story? Answer YES or NO.\n\nStory: {story_content}"
    
    def parse_validation_response(self, response: str) -> CustomValidationResult:
        """Parse yes/no response."""
        response_upper = response.upper().strip()
        
        if "YES" in response_upper:
            return CustomValidationResult(
                is_valid=True,
                score=1.0,
                details={"response_type": "yes", "raw_response": response},
                reasoning="Validator responded with YES"
            )
        elif "NO" in response_upper:
            return CustomValidationResult(
                is_valid=False,
                score=0.0,
                details={"response_type": "no", "raw_response": response},
                reasoning="Validator responded with NO"
            )
        else:
            return CustomValidationResult(
                is_valid=False,
                score=0.5,
                details={"response_type": "unclear", "raw_response": response},
                reasoning="Unclear response from validator"
            )


class AdvancedValidator(BaseValidator):
    """Example of an advanced validator with custom scoring."""
    
    def get_validation_prompt(self, story_content: str) -> str:
        """Generate an advanced validation prompt."""
        return (
            "Rate this bedtime story on a scale of 1-10 for appropriateness for 3-year-olds. "
            "Consider language complexity, content, and educational value. "
            "Format your response as: SCORE: X (where X is 1-10)\n\n"
            f"Story: {story_content}"
        )
    
    def parse_validation_response(self, response: str) -> CustomValidationResult:
        """Parse scored response."""
        response_upper = response.upper().strip()
        
        # Look for SCORE: pattern
        if "SCORE:" in response_upper:
            try:
                # Extract score
                score_part = response_upper.split("SCORE:")[1].strip()
                score_num = int(score_part.split()[0])
                
                # Convert 1-10 scale to 0-1 scale
                normalized_score = max(0.0, min(1.0, (score_num - 1) / 9))
                is_valid = score_num >= 6  # Consider 6+ as valid
                
                return CustomValidationResult(
                    is_valid=is_valid,
                    score=normalized_score,
                    details={
                        "raw_score": score_num,
                        "normalized_score": normalized_score,
                        "raw_response": response
                    },
                    reasoning=f"Story scored {score_num}/10"
                )
            except (ValueError, IndexError):
                pass
        
        # Fallback parsing
        return CustomValidationResult(
            is_valid=False,
            score=0.1,
            details={"error": "Could not parse score", "raw_response": response},
            reasoning="Failed to parse validation score"
        )


async def demo_validators():
    """Demonstrate different validator implementations."""
    logger.info("=== Validation System Architecture Demo ===")
    
    # Create mock provider
    provider = MockLLMProvider(model_name="demo-validator")
    config = {"generation": {"max_new_tokens": 64, "temperature": 0.1}}
    
    # Test stories
    test_stories = [
        "Once upon a time, there was a happy cat who loved to play.",
        "The quick brown fox jumps over the lazy dog.",
        "A very short story."
    ]
    
    # Test different validators
    validators = [
        ("QualityValidator (English check)", QualityValidator(provider, config)),
        ("SimpleYesNoValidator", SimpleYesNoValidator(provider, config)),
        ("AdvancedValidator (1-10 scoring)", AdvancedValidator(provider, config))
    ]
    
    for story_idx, story in enumerate(test_stories, 1):
        logger.info(f"\n--- Testing Story {story_idx}: '{story}' ---")
        
        for validator_name, validator in validators:
            logger.info(f"\n{validator_name}:")
            
            try:
                # Show the prompt that would be generated
                prompt = validator.get_validation_prompt(story)
                logger.info(f"  Prompt: {prompt[:80]}...")
                
                # Run validation
                result = await validator.validate(story)
                
                logger.info(f"  Result: Valid={result.is_valid}, Score={result.score:.2f}")
                logger.info(f"  Reasoning: {result.reasoning[:60]}...")
                
            except Exception as e:
                logger.error(f"  Error: {e}")
    
    logger.info("\n=== Demo completed ===")
    logger.info("\nKey Architecture Benefits:")
    logger.info("1. BaseValidator handles common LLM provider interaction")
    logger.info("2. Subclasses focus on prompt formatting and response parsing")
    logger.info("3. Easy to create new validators with different logic")
    logger.info("4. Consistent error handling and result structure")


def main():
    """Main entry point."""
    try:
        asyncio.run(demo_validators())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
