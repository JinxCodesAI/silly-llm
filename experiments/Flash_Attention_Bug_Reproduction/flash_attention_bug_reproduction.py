#!/usr/bin/env python3
"""
Minimal Flash Attention Bug Reproduction Script

This script demonstrates a potential issue with flash_attention_2 implementation
using the Qwen2.5-0.5B model. It runs the same inference with and without
flash_attention to help identify and reproduce bugs.

Usage:
    python flash_attention_bug_reproduction.py
"""

import torch
import time
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_inference(use_flash_attention=False, model_name="Qwen/Qwen3-0.6B"):
    """
    Test inference with or without flash attention
    
    Args:
        use_flash_attention (bool): Whether to use flash_attention_2
        model_name (str): Model to test with
    
    Returns:
        dict: Results including generated text, timing, and any errors
    """
    print(f"\n{'='*60}")
    print(f"Testing {'WITH' if use_flash_attention else 'WITHOUT'} Flash Attention")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    result = {
        "use_flash_attention": use_flash_attention,
        "model_name": model_name,
        "success": False,
        "generated_text": None,
        "generation_time": None,
        "error": None,
        "error_traceback": None
    }
    
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up model kwargs
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        
        # Add flash attention if requested
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using flash_attention_2 implementation")
        else:
            print("Using default attention implementation")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Prepare input - simple hardcoded prompt
        prompt = "Write a short story about a robot learning to paint:"
        
        # Apply chat template if available (following original inference_comparison.py pattern)
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Always use non-thinking mode as in original
            )
        except Exception as e:
            print(f"Chat template not available, using raw prompt: {e}")
            text = prompt
        
        print(f"Input text: {text[:100]}...")
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(f"Input tokens: {inputs['input_ids'].shape[1]}")
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True
        }
        
        print("Starting generation...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        generation_time = time.time() - start_time
        
        # Extract generated text
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Update result
        result.update({
            "success": True,
            "generated_text": generated_text,
            "generation_time": generation_time,
            "tokens_generated": len(generated_tokens),
            "tokens_per_second": len(generated_tokens) / generation_time if generation_time > 0 else 0
        })
        
        print(f"✅ Generation successful!")
        print(f"Time: {generation_time:.2f}s")
        print(f"Tokens generated: {len(generated_tokens)}")
        print(f"Tokens/second: {len(generated_tokens) / generation_time:.1f}")
        print(f"Generated text: {generated_text[:200]}...")
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        result.update({
            "error": error_msg,
            "error_traceback": error_traceback
        })
        
        print(f"❌ Error occurred: {error_msg}")
        print(f"Full traceback:\n{error_traceback}")
    
    finally:
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result

def compare_implementations(model_name="Qwen/Qwen3-0.6B"):
    """
    Compare inference with and without flash attention
    
    Args:
        model_name (str): Model to test with
    """
    print(f"🔬 Flash Attention Bug Reproduction Script")
    print(f"Testing model: {model_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test without flash attention first
    result_without = test_inference(use_flash_attention=False, model_name=model_name)
    
    # Test with flash attention
    result_with = test_inference(use_flash_attention=True, model_name=model_name)
    
    # Compare results
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"\nWithout Flash Attention:")
    print(f"  Success: {result_without['success']}")
    if result_without['success']:
        print(f"  Time: {result_without['generation_time']:.2f}s")
        print(f"  Tokens/sec: {result_without['tokens_per_second']:.1f}")
    else:
        print(f"  Error: {result_without['error']}")
    
    print(f"\nWith Flash Attention:")
    print(f"  Success: {result_with['success']}")
    if result_with['success']:
        print(f"  Time: {result_with['generation_time']:.2f}s")
        print(f"  Tokens/sec: {result_with['tokens_per_second']:.1f}")
        
        # Performance comparison if both succeeded
        if result_without['success']:
            speedup = result_without['generation_time'] / result_with['generation_time']
            print(f"  Speedup: {speedup:.2f}x")
    else:
        print(f"  Error: {result_with['error']}")
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    if result_without['success'] and result_with['success']:
        print("✅ Both implementations work correctly")
        if result_with['generation_time'] < result_without['generation_time']:
            print("🚀 Flash Attention provides performance improvement")
        else:
            print("⚠️  Flash Attention may not provide expected speedup")
    elif result_without['success'] and not result_with['success']:
        print("❌ Flash Attention implementation has issues!")
        print("🐛 This indicates a potential bug in flash_attention_2")
        print("\nError details for bug report:")
        print(f"Model: {model_name}")
        print(f"Error: {result_with['error']}")
        print("Full traceback:")
        print(result_with['error_traceback'])
    elif not result_without['success'] and result_with['success']:
        print("⚠️  Standard implementation failed but Flash Attention works")
        print("This is unexpected - check your environment")
    else:
        print("❌ Both implementations failed")
        print("Check your model installation and CUDA setup")
    
    return result_without, result_with

def main():
    """Main function"""
    # You can change the model here to test different models
    model_name = "Qwen/Qwen3-0.6B"
    
    try:
        result_without, result_with = compare_implementations(model_name)
        
        # Save results for bug report if needed
        if not result_with['success'] and result_without['success']:
            print(f"\n💾 Bug reproduction data:")
            print(f"Model: {model_name}")
            print(f"PyTorch: {torch.__version__}")
            print(f"Transformers: {__import__('transformers').__version__}")
            try:
                import flash_attn
                print(f"Flash Attention: {flash_attn.__version__}")
            except ImportError:
                print("Flash Attention: Not installed or not importable")
            
            print(f"\nTo reproduce:")
            print(f"1. Install requirements: pip install torch transformers flash-attn")
            print(f"2. Run this script")
            print(f"3. Observe the error when flash_attention_2 is enabled")
            
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
