"""Test script for the story generation pipeline."""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add the workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from training.common.utils import setup_logging, load_vocabulary
from training.common.data_models import GenerationConfig
from training.synthetic_data_generation.template_manager import TemplateManager
from training.synthetic_data_generation.prompt_generator import PromptGenerator


async def test_components():
    """Test individual components of the pipeline."""
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Testing story generation pipeline components...")
    
    # Test 1: Load vocabulary
    try:
        vocab_path = "training/synthetic_data_generation/vocabulary.json"
        vocabulary = load_vocabulary(vocab_path)
        logger.info(f"✓ Vocabulary loaded: {len(vocabulary.nouns)} nouns, "
                   f"{len(vocabulary.verbs)} verbs, {len(vocabulary.adjectives)} adjectives")
        
        # Test random word selection
        random_words = vocabulary.get_random_words()
        logger.info(f"✓ Random words: {random_words}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load vocabulary: {e}")
        return False
    
    # Test 2: Template Manager
    try:
        features_path = "docs/story_features.json"
        template_manager = TemplateManager(story_features_path=features_path)
        
        # Create a test prompt
        test_words = {"word1": "moon", "word2": "dance", "word3": "happy"}
        prompt = template_manager.create_prompt(test_words)
        
        logger.info(f"✓ Template manager created prompt: {prompt.prompt_id}")
        logger.info(f"  Selected words: {prompt.selected_words}")
        logger.info(f"  Additional condition: {prompt.additional_condition}")
        logger.info(f"  Full prompt preview: {prompt.full_prompt[:100]}...")
        
    except Exception as e:
        logger.error(f"✗ Failed to test template manager: {e}")
        return False
    
    # Test 3: Prompt Generator
    try:
        examples_path = "training/synthetic_data_generation/example_conversation.txt"
        prompt_generator = PromptGenerator(
            vocabulary=vocabulary,
            template_manager=template_manager,
            conversation_examples_path=examples_path,
            k_shot_count=2
        )
        
        # Generate test prompts
        test_prompts = prompt_generator.generate_prompts(
            count=3, use_k_shot=True, ensure_diversity=True
        )
        
        logger.info(f"✓ Generated {len(test_prompts)} test prompts")
        for i, prompt in enumerate(test_prompts):
            logger.info(f"  Prompt {i+1}: words={list(prompt.selected_words.values())}, "
                       f"k-shot examples={len(prompt.k_shot_examples)}")
        
        # Test statistics
        stats = prompt_generator.get_statistics()
        logger.info(f"✓ Prompt generator stats: {stats}")
        
    except Exception as e:
        logger.error(f"✗ Failed to test prompt generator: {e}")
        return False
    
    # Test 4: Chat template formatting
    try:
        if test_prompts:
            chat_messages = template_manager.format_for_chat_template(test_prompts[0])
            logger.info(f"✓ Chat template formatting: {len(chat_messages)} messages")
            for msg in chat_messages:
                role = msg['role']
                content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                logger.info(f"  {role}: {content_preview}")
        
    except Exception as e:
        logger.error(f"✗ Failed to test chat template formatting: {e}")
        return False
    
    logger.info("✓ All component tests passed!")
    return True


async def test_mock_generation():
    """Test generation with mock LLM provider."""
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Testing mock generation...")
    
    try:
        # Create mock responses
        mock_responses = [
            "Once upon a time, there was a happy bunny who loved to dance under the moon. "
            "The bunny would hop and skip all night long, bringing joy to all the forest animals. "
            "When morning came, the bunny would sleep peacefully, dreaming of more dancing adventures.",
            
            "In a magical garden, a little bird found a shiny star that had fallen from the sky. "
            "The bird was very gentle and decided to help the star find its way back home. "
            "Together they flew high into the night, and the star returned to its place, twinkling brightly.",
            
            "A small teddy bear sat by the window, watching the rain fall softly outside. "
            "The bear felt a bit sad, but then remembered all the fun games to play indoors. "
            "Soon the bear was building blocks and reading books, feeling happy and cozy again."
        ]
        
        # Test validation
        from training.common.utils import validate_story
        
        for i, story in enumerate(mock_responses):
            required_words = ["moon", "dance", "happy"] if i == 0 else ["star", "gentle", "bird"] if i == 1 else ["teddy", "sad", "happy"]
            validation = validate_story(story, required_words)
            
            logger.info(f"✓ Mock story {i+1} validation:")
            logger.info(f"  Valid: {validation.is_valid}")
            logger.info(f"  Word count: {validation.word_count}")
            logger.info(f"  Contains required words: {validation.contains_required_words}")
            logger.info(f"  Score: {validation.score:.2f}")
            if validation.issues:
                logger.info(f"  Issues: {validation.issues}")
        
        logger.info("✓ Mock generation test completed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Mock generation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Story Generation Pipeline Test Suite")
    print("=" * 60)
    
    # Run component tests
    component_success = asyncio.run(test_components())
    
    print("\n" + "-" * 40)
    
    # Run mock generation tests
    mock_success = asyncio.run(test_mock_generation())
    
    print("\n" + "=" * 60)
    if component_success and mock_success:
        print("✓ ALL TESTS PASSED!")
        print("\nThe pipeline is ready for use. You can now:")
        print("1. Create a configuration file:")
        print("   python -m training.synthetic_data_generation.main --create-config my_config.yaml")
        print("2. Run generation:")
        print("   python -m training.synthetic_data_generation.main --config my_config.yaml")
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please check the error messages above and fix any issues.")
    print("=" * 60)


if __name__ == "__main__":
    main()
