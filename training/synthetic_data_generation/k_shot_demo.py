"""Demo script showing k-shot prompting in action."""

import sys
import json
from pathlib import Path

# Add the workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from training.common.utils import setup_logging, load_vocabulary
from training.synthetic_data_generation.template_manager import TemplateManager
from training.synthetic_data_generation.prompt_generator import PromptGenerator


def demo_k_shot_prompting():
    """Demonstrate k-shot prompting with different configurations."""
    print("=" * 70)
    print("K-Shot Prompting Demo")
    print("=" * 70)
    
    setup_logging("INFO")
    
    # Load components
    vocabulary = load_vocabulary("training/synthetic_data_generation/vocabulary.json")
    template_manager = TemplateManager(story_features_path="docs/story_features.json")
    
    # Test different k-shot configurations
    k_shot_configs = [0, 1, 2, 3]
    
    for k_shot_count in k_shot_configs:
        print(f"\n{'='*50}")
        print(f"K-Shot Configuration: {k_shot_count} examples")
        print(f"{'='*50}")
        
        prompt_generator = PromptGenerator(
            vocabulary=vocabulary,
            template_manager=template_manager,
            conversation_examples_path="training/synthetic_data_generation/example_conversation.txt",
            k_shot_count=k_shot_count
        )
        
        # Generate a sample prompt
        prompts = prompt_generator.generate_prompts(count=1, use_k_shot=True)
        prompt = prompts[0]
        
        print(f"\nPrompt ID: {prompt.prompt_id}")
        print(f"Selected words: {prompt.selected_words}")
        print(f"Additional condition: {prompt.additional_condition}")
        print(f"K-shot examples count: {len(prompt.k_shot_examples)}")
        
        # Show chat template formatting
        chat_messages = template_manager.format_for_chat_template(prompt)
        print(f"\nChat template messages ({len(chat_messages)} total):")
        
        for i, message in enumerate(chat_messages):
            role = message['role']
            content = message['content']
            
            # Truncate long content for display
            if len(content) > 200:
                content_display = content[:200] + "..."
            else:
                content_display = content
            
            print(f"\n  Message {i+1} ({role}):")
            print(f"  {content_display}")
        
        # Save detailed example for the 2-shot case
        if k_shot_count == 2:
            save_detailed_example(prompt, chat_messages)


def save_detailed_example(prompt, chat_messages):
    """Save a detailed example of 2-shot prompting."""
    example_data = {
        "description": "Detailed example of 2-shot prompting for bedtime story generation",
        "prompt_metadata": {
            "prompt_id": prompt.prompt_id,
            "selected_words": prompt.selected_words,
            "additional_condition": prompt.additional_condition,
            "k_shot_examples_count": len(prompt.k_shot_examples)
        },
        "full_prompt_text": prompt.full_prompt,
        "chat_template_format": chat_messages,
        "explanation": {
            "structure": "The chat template contains k-shot examples followed by the actual prompt",
            "k_shot_examples": "Previous user prompts and assistant responses serve as examples",
            "final_prompt": "The last user message is the prompt to be completed",
            "model_task": "The model should generate a response similar to the examples"
        },
        "usage_in_pipeline": {
            "step_1": "PromptGenerator creates prompts with k-shot examples",
            "step_2": "TemplateManager formats them for chat template",
            "step_3": "BatchProcessor sends to model for generation",
            "step_4": "Model generates story following the pattern from examples"
        }
    }
    
    with open("k_shot_detailed_example.json", "w", encoding="utf-8") as f:
        json.dump(example_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Saved detailed example to: k_shot_detailed_example.json")


def show_conversation_examples():
    """Show the parsed conversation examples."""
    print(f"\n{'='*50}")
    print("Parsed Conversation Examples")
    print(f"{'='*50}")
    
    from training.common.utils import parse_conversation_examples
    
    examples = parse_conversation_examples("training/synthetic_data_generation/example_conversation.txt")
    
    print(f"\nFound {len(examples)} conversation examples:")
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Messages: {len(example.messages)}")
        
        for j, message in enumerate(example.messages):
            role = message.role
            content = message.content[:100] + "..." if len(message.content) > 100 else message.content
            print(f"  {j+1}. {role}: {content}")


def main():
    """Run the k-shot prompting demo."""
    demo_k_shot_prompting()
    show_conversation_examples()
    
    print(f"\n{'='*70}")
    print("Demo completed!")
    print("\nKey takeaways:")
    print("1. K-shot prompting uses previous examples to guide generation")
    print("2. More examples can improve consistency but increase token usage")
    print("3. The pipeline automatically formats examples for chat templates")
    print("4. Examples are randomly selected from the conversation file")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
