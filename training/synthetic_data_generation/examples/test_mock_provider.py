"""Test script to verify mock provider works without torch/transformers."""

import sys
import asyncio
from pathlib import Path

# Add the workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root))

from training.common.utils import setup_logging
from training.common.llm_providers import MockLLMProvider
from training.common.data_models import GenerationConfig


async def test_mock_provider():
    """Test the mock LLM provider."""
    print("=" * 60)
    print("Mock LLM Provider Test")
    print("=" * 60)
    
    setup_logging("INFO")
    
    # Create mock provider
    provider = MockLLMProvider(model_name="test-mock-model")
    
    # Test capabilities
    capabilities = provider.get_capabilities()
    print(f"\nProvider capabilities:")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    
    # Test generation
    test_prompts = [
        'Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old\ncontaining words "cat" "jump" "happy"\n\nkeep story coherent and gramatically correct',
        'Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old\ncontaining words "moon" "dance" "bright"\n\nmake sure story has sad elements but ends well\n\nkeep story coherent and gramatically correct',
        'Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old\ncontaining words "teddy" "sleep" "cozy"\n\nkeep story coherent and gramatically correct'
    ]
    
    config = GenerationConfig(
        batch_size=3,
        max_new_tokens=150,
        temperature=0.8
    )
    
    print(f"\nGenerating {len(test_prompts)} test stories...")
    
    # Generate stories
    stories = await provider.generate_batch(test_prompts, config)
    
    print(f"\nGenerated {len(stories)} stories:")
    for i, story in enumerate(stories, 1):
        print(f"\n--- Story {i} ---")
        print(story)
        print(f"Word count: {len(story.split())}")
    
    # Test capabilities after generation
    final_capabilities = provider.get_capabilities()
    print(f"\nFinal generation count: {final_capabilities['generation_count']}")
    
    print("\n" + "=" * 60)
    print("✅ Mock provider test completed successfully!")
    print("This demonstrates that the pipeline can work without torch/transformers.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_mock_provider())
