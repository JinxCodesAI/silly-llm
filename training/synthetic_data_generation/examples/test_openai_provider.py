"""Test script to verify OpenAI-compatible provider works."""

import sys
import os
import asyncio
from pathlib import Path

# Add the workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root))

from training.common.utils import setup_logging
from training.common.llm_providers import OpenAICompatibleProvider
from training.common.data_models import GenerationConfig


async def test_openai_provider():
    """Test the OpenAI-compatible provider."""
    print("=" * 60)
    print("OpenAI-Compatible Provider Test")
    print("=" * 60)
    
    setup_logging("INFO")
    
    # Check for API key
    api_key = os.getenv('AI_API_KEY')
    if not api_key:
        print("\n❌ ERROR: AI_API_KEY environment variable not set!")
        print("Please set it with: export AI_API_KEY=your_api_key")
        print("=" * 60)
        return False
    
    print(f"\n✅ API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '***'}")
    
    try:
        # Create provider
        provider = OpenAICompatibleProvider(
            model_name="gpt-3.5-turbo",  # Default model for testing
            api_base_url="https://api.openai.com/v1"
        )
        
        # Test capabilities
        capabilities = provider.get_capabilities()
        print(f"\nProvider capabilities:")
        for key, value in capabilities.items():
            print(f"  {key}: {value}")
        
        # Test generation
        test_prompts = [
            'Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old\ncontaining words "cat" "jump" "happy"\n\nkeep story coherent and gramatically correct',
            'Generate simple, short (up to 150 words) bed time story easy to understand and follow by 3 years old\ncontaining words "moon" "dance" "bright"\n\nmake sure story has sad elements but ends well\n\nkeep story coherent and gramatically correct'
        ]
        
        config = GenerationConfig(
            batch_size=2,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9
        )
        
        print(f"\nGenerating {len(test_prompts)} test stories...")
        print("This may take a moment as we make API calls...")
        
        # Generate stories
        stories = await provider.generate_batch(test_prompts, config)
        
        print(f"\nGenerated {len(stories)} stories:")
        for i, story in enumerate(stories, 1):
            if story.strip():
                print(f"\n--- Story {i} ---")
                print(story)
                print(f"Word count: {len(story.split())}")
            else:
                print(f"\n--- Story {i} ---")
                print("❌ Generation failed or empty response")
        
        # Test capabilities after generation
        final_capabilities = provider.get_capabilities()
        print(f"\nFinal generation count: {final_capabilities['generation_count']}")
        
        print("\n" + "=" * 60)
        print("✅ OpenAI-compatible provider test completed successfully!")
        print("The provider can now be used with the main pipeline.")
        print("=" * 60)
        return True
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Install httpx with: pip install httpx")
        print("=" * 60)
        return False
        
    except Exception as e:
        print(f"\n❌ Provider test failed: {e}")
        print("Please check your API key and network connection.")
        print("=" * 60)
        return False


def test_environment():
    """Test environment setup."""
    print("Environment Check:")
    print(f"  AI_API_KEY: {'✅ Set' if os.getenv('AI_API_KEY') else '❌ Not set'}")
    
    try:
        import httpx
        print(f"  httpx: ✅ Available (version: {httpx.__version__})")
    except ImportError:
        print(f"  httpx: ❌ Not available (install with: pip install httpx)")
    
    print()


if __name__ == "__main__":
    test_environment()
    asyncio.run(test_openai_provider())
