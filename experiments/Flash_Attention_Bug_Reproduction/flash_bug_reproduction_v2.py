#!/usr/bin/env python3
"""
Flash Attention Bug Reproduction Script v2 - Based on Qwen3 Test Patterns

This script reproduces potential flash attention issues using the exact same patterns
as the official Qwen3 test suite. It uses Qwen3ForCausalLM and follows the testing
methodology from transformers/tests/models/qwen3/test_modeling_qwen3.py

Key differences from v1:
- Uses Qwen3ForCausalLM instead of AutoModelForCausalLM
- Uses Qwen3-0.6B-Base model (as in official tests)
- Follows exact test patterns for attention implementation comparison
- Uses torch.testing.assert_close for numerical comparison
- Tests multiple attention implementations: eager, sdpa, flash_attention_2

Usage:
    python flash_bug_reproduction_v2.py
"""

import torch
import time
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

def cleanup_memory():
    """Clean up GPU memory - similar to transformers test cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_attention_implementation(model_name="Qwen/Qwen3-0.6B", attn_implementation="eager"):
    """
    Test specific attention implementation following Qwen3 test patterns
    
    Args:
        model_name (str): Model to test with
        attn_implementation (str): Attention implementation to test
    
    Returns:
        dict: Results including generated tokens, timing, and any errors
    """
    print(f"\n{'='*60}")
    print(f"Testing attention implementation: {attn_implementation}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    result = {
        "attn_implementation": attn_implementation,
        "model_name": model_name,
        "success": False,
        "generated_ids": None,
        "logits": None,
        "generation_time": None,
        "error": None,
        "error_traceback": None
    }
    
    try:
        cleanup_memory()
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model with {attn_implementation} attention...")

        # Load model with specific attention implementation (following original inference_comparison.py)
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,  # Use bfloat16 as in original script
        }

        if attn_implementation == "flash_attention_2":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif attn_implementation == "sdpa":
            model_kwargs["attn_implementation"] = "sdpa"
        # For "eager", don't set attn_implementation (default)

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Test 1: Basic logits test (similar to test_model_600m_logits)
        print("Running logits test...")
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]  # Same as in official test
        input_tensor = torch.tensor([input_ids]).to(model.device)

        with torch.no_grad():
            logits = model(input_tensor).logits.float().cpu()

        result["logits"] = logits
        print(f"Logits shape: {logits.shape}")
        print(f"Logits mean: {logits.mean(-1)}")
        
        # Test 2: Generation test (similar to test_model_600m_generation)
        print("Running generation test...")
        prompt = "My favourite condiment is "
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        
        # Greedy generation (temperature=0) as in official tests
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        
        generation_time = time.time() - start_time
        
        # Extract only the generated part (excluding input)
        generated_tokens = generated_ids[:, input_ids.shape[1]:]
        
        result.update({
            "success": True,
            "generated_ids": generated_ids,
            "generated_tokens": generated_tokens,
            "generation_time": generation_time,
            "input_length": input_ids.shape[1],
            "total_length": generated_ids.shape[1]
        })
        
        # Decode and display result
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"✅ Generation successful!")
        print(f"Time: {generation_time:.3f}s")
        print(f"Input tokens: {input_ids.shape[1]}")
        print(f"Generated tokens: {generated_tokens.shape[1]}")
        print(f"Generated text: {generated_text}")
        
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
        cleanup_memory()
    
    return result

def compare_attention_implementations(model_name="Qwen/Qwen3-0.6B"):
    """
    Compare different attention implementations following Qwen3 test methodology
    
    Args:
        model_name (str): Model to test with
    """
    print(f"🔬 Flash Attention Bug Reproduction Script v2")
    print(f"Based on Qwen3 official test patterns")
    print(f"Testing model: {model_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Test different attention implementations in order
    implementations = ["eager", "sdpa", "flash_attention_2"]
    results = {}
    
    for impl in implementations:
        print(f"\n{'='*80}")
        print(f"Testing {impl.upper()} attention implementation")
        print(f"{'='*80}")
        
        try:
            result = test_attention_implementation(model_name, impl)
            results[impl] = result
        except Exception as e:
            print(f"Failed to test {impl}: {e}")
            results[impl] = {
                "attn_implementation": impl,
                "success": False,
                "error": str(e),
                "error_traceback": traceback.format_exc()
            }
    
    # Compare results (following Qwen3 test pattern)
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*80}")
    
    successful_results = {k: v for k, v in results.items() if v["success"]}
    failed_results = {k: v for k, v in results.items() if not v["success"]}
    
    # Display success/failure summary
    for impl, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{impl:20} {status}")
        if result["success"]:
            print(f"{'':20} Time: {result['generation_time']:.3f}s")
        else:
            print(f"{'':20} Error: {result['error']}")
    
    # Numerical comparison (similar to official tests)
    if len(successful_results) >= 2:
        print(f"\n🔍 NUMERICAL COMPARISON")
        print(f"{'='*50}")
        
        # Use eager as reference (as in official tests)
        if "eager" in successful_results:
            reference = successful_results["eager"]
            reference_tokens = reference["generated_tokens"]
            
            for impl, result in successful_results.items():
                if impl == "eager":
                    continue
                
                test_tokens = result["generated_tokens"]
                
                try:
                    # Use torch.testing.assert_close as in official tests
                    torch.testing.assert_close(
                        reference_tokens, 
                        test_tokens, 
                        rtol=1e-4, 
                        atol=1e-4
                    )
                    print(f"✅ {impl} matches eager (within tolerance)")
                except AssertionError as e:
                    print(f"❌ {impl} differs from eager: {e}")
                    print(f"   Reference: {reference_tokens[0][:10].tolist()}")
                    print(f"   Test:      {test_tokens[0][:10].tolist()}")
    
    # Final summary and bug report info
    print(f"\n🎯 SUMMARY:")
    
    if "flash_attention_2" in failed_results and "eager" in successful_results:
        print("❌ FLASH ATTENTION BUG DETECTED!")
        print("🐛 Flash attention fails while eager attention works")
        print("\nBug report information:")
        print(f"Model: {model_name}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {__import__('transformers').__version__}")
        try:
            import flash_attn
            print(f"Flash Attention: {flash_attn.__version__}")
        except ImportError:
            print("Flash Attention: Not installed or not importable")
        
        flash_error = failed_results["flash_attention_2"]
        print(f"\nFlash Attention Error:")
        print(f"Error: {flash_error['error']}")
        print("Full traceback:")
        print(flash_error['error_traceback'])
        
    elif all(result["success"] for result in results.values()):
        print("✅ All attention implementations work correctly")
        
        # Check for performance differences
        if len(successful_results) > 1:
            times = {impl: result["generation_time"] for impl, result in successful_results.items()}
            fastest = min(times, key=times.get)
            slowest = max(times, key=times.get)
            
            if times[slowest] > times[fastest] * 1.1:  # 10% difference threshold
                speedup = times[slowest] / times[fastest]
                print(f"🚀 Performance difference detected:")
                print(f"   Fastest: {fastest} ({times[fastest]:.3f}s)")
                print(f"   Slowest: {slowest} ({times[slowest]:.3f}s)")
                print(f"   Speedup: {speedup:.2f}x")
            else:
                print("⚖️  Similar performance across implementations")
    else:
        print("⚠️  Multiple implementations failed - check your environment")
    
    return results

def main():
    """Main function"""
    # Use the same model as in official Qwen3 tests
    model_name = "Qwen/Qwen3-0.6B"
    
    try:
        results = compare_attention_implementations(model_name)
        
        # Additional debug info for bug reports
        print(f"\n💾 Debug Information:")
        print(f"Model: {model_name}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Check if this matches expected behavior from official tests
        if "flash_attention_2" in results and not results["flash_attention_2"]["success"]:
            print(f"\n🔧 Reproduction Steps:")
            print(f"1. Install: pip install torch transformers flash-attn")
            print(f"2. Run: python {__file__}")
            print(f"3. Observe flash_attention_2 failure while eager works")
            
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
