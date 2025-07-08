"""Demo script showing the data generation pipeline without requiring model loading."""

import sys
from pathlib import Path
import json
import random

# Add the workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from training.common.utils import setup_logging, load_vocabulary
from training.common.data_models import GenerationConfig
from training.synthetic_data_generation.template_manager import TemplateManager
from training.synthetic_data_generation.prompt_generator import PromptGenerator


def demo_prompt_generation():
    """Demonstrate prompt generation with k-shot examples."""
    print("=" * 60)
    print("Data Generation Pipeline Demo")
    print("=" * 60)
    
    setup_logging("INFO")
    
    # Load vocabulary
    print("\n1. Loading vocabulary...")
    vocabulary = load_vocabulary("training/synthetic_data_generation/vocabulary.json")
    print(f"   Loaded: {len(vocabulary.nouns)} nouns, {len(vocabulary.verbs)} verbs, {len(vocabulary.adjectives)} adjectives")
    
    # Initialize template manager
    print("\n2. Initializing template manager...")
    template_manager = TemplateManager(story_features_path="docs/story_features.json")
    features = template_manager.get_available_features()
    print(f"   Available story features: {len(features)}")
    for feature in features:
        print(f"   - {feature}")
    
    # Initialize prompt generator
    print("\n3. Initializing prompt generator...")
    prompt_generator = PromptGenerator(
        vocabulary=vocabulary,
        template_manager=template_manager,
        conversation_examples_path="training/synthetic_data_generation/example_conversation.txt",
        k_shot_count=2
    )
    
    stats = prompt_generator.get_statistics()
    print(f"   Conversation examples loaded: {stats['conversation_examples']}")
    print(f"   Total possible word combinations: {stats['total_possible_combinations']:,}")
    
    # Generate sample prompts
    print("\n4. Generating sample prompts...")
    prompts = prompt_generator.generate_prompts(count=5, use_k_shot=True, ensure_diversity=True)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   Prompt {i}:")
        print(f"   ID: {prompt.prompt_id}")
        print(f"   Selected words: {prompt.selected_words}")
        print(f"   Additional condition: {prompt.additional_condition}")
        print(f"   K-shot examples: {len(prompt.k_shot_examples)}")
        print(f"   Full prompt preview:")
        print(f"   {prompt.full_prompt[:200]}...")
    
    # Show chat template formatting
    print("\n5. Chat template formatting example...")
    sample_prompt = prompts[0]
    chat_messages = template_manager.format_for_chat_template(sample_prompt)
    
    print(f"   Formatted as {len(chat_messages)} messages:")
    for j, msg in enumerate(chat_messages):
        role = msg['role']
        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"   Message {j+1} ({role}): {content}")
    
    # Save sample prompts
    print("\n6. Saving sample prompts...")
    output_data = []
    for prompt in prompts:
        output_data.append({
            "prompt_id": prompt.prompt_id,
            "selected_words": prompt.selected_words,
            "additional_condition": prompt.additional_condition,
            "full_prompt": prompt.full_prompt,
            "k_shot_examples_count": len(prompt.k_shot_examples),
            "chat_messages": template_manager.format_for_chat_template(prompt)
        })
    
    with open("sample_prompts.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"   Saved {len(output_data)} sample prompts to sample_prompts.json")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("\nNext steps:")
    print("1. Install torch and transformers: pip install torch transformers")
    print("2. Create config: python -m training.synthetic_data_generation.main --create-config my_config.yaml")
    print("3. Run generation: python -m training.synthetic_data_generation.main --config my_config.yaml")
    print("=" * 60)


def demo_template_variations():
    """Show different template variations and word combinations."""
    print("\n" + "=" * 60)
    print("Template Variations Demo")
    print("=" * 60)
    
    vocabulary = load_vocabulary("training/synthetic_data_generation/vocabulary.json")
    template_manager = TemplateManager(story_features_path="docs/story_features.json")
    
    print("\nGenerating 10 different word combinations and conditions:")
    
    for i in range(10):
        words = vocabulary.get_random_words()
        features = template_manager.get_available_features()
        condition = random.choice(features) if features else ""
        
        prompt = template_manager.create_prompt(words, condition)
        
        print(f"\n{i+1}. Words: {list(words.values())}")
        print(f"   Condition: {condition}")
        print(f"   Template preview: {prompt.full_prompt[:150]}...")


if __name__ == "__main__":
    demo_prompt_generation()
    demo_template_variations()
