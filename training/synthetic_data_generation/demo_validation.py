#!/usr/bin/env python3
"""Demo script for custom validation system."""

import asyncio
import logging
from pathlib import Path

from .validation.quality_validator import QualityValidator
from ..common.llm_providers import MockLLMProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_validation():
    """Demonstrate the custom validation system."""
    logger.info("Starting validation system demo")
    
    # Create a mock provider for testing
    provider = MockLLMProvider(model_name="demo-validator")
    
    # Configure the validator
    config = {
        "generation": {
            "max_new_tokens": 64,
            "temperature": 0.1
        }
    }
    
    # Create validator
    validator = QualityValidator(provider, config)
    
    # Test stories
    test_stories = [
        "Once upon a time, there was a happy cat who loved to play in the garden.",
        "The quick brown fox jumps over the lazy dog every morning.",
        "Children love to read bedtime stories before going to sleep.",
        "A short story.",  # This might be too short
        "Hola amigo, como estas? This story mixes languages.",  # Mixed languages
    ]
    
    logger.info(f"Testing {len(test_stories)} stories")
    
    for i, story in enumerate(test_stories, 1):
        logger.info(f"\n--- Testing Story {i} ---")
        logger.info(f"Story: {story}")
        
        try:
            # Validate the story
            result = await validator.validate(story)
            
            logger.info(f"Valid: {result.is_valid}")
            logger.info(f"Score: {result.score:.2f}")
            logger.info(f"Reasoning: {result.reasoning}")
            
            if result.details:
                logger.info(f"Details: {result.details}")
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
    
    logger.info("\nValidation demo completed")


def main():
    """Main entry point."""
    try:
        asyncio.run(demo_validation())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
