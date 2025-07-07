#!/usr/bin/env python3
"""
Test script to verify the setup for inference comparison experiments.
This script checks if all required dependencies are available and models can be loaded.
"""

import sys
import torch
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers', 
        'accelerate',
        'bitsandbytes',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    # Check optional packages
    optional_packages = ['flash_attn', 'optimum']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            print(f"  ⚠️  {package} (optional) - not available")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required dependencies are available!")
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n🔍 Checking CUDA setup...")
    
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available")
        print(f"  🔧 CUDA version: {torch.version.cuda}")
        print(f"  🎮 GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  🎮 GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        print("  ⚠️  CUDA not available - will use CPU (slower)")
        return False

def test_model_loading():
    """Test if we can load a small model"""
    print("\n🔍 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"
        print(f"  📥 Loading test model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"  ✅ Model loaded successfully")
        print(f"  🔧 Model device: {next(model.parameters()).device}")
        print(f"  🔧 Model dtype: {next(model.parameters()).dtype}")
        
        # Test generation
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  ✅ Test generation successful")
        print(f"  📝 Input: {test_input}")
        print(f"  📝 Output: {response}")
        
        # Cleanup
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

def check_qwen_models():
    """Check if Qwen models are accessible"""
    print("\n🔍 Checking Qwen model accessibility...")
    
    try:
        from transformers import AutoTokenizer
        
        models_to_check = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"]
        
        for model_name in models_to_check:
            try:
                print(f"  📥 Checking {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"  ✅ {model_name} accessible")
                del tokenizer
            except Exception as e:
                print(f"  ❌ {model_name} not accessible: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking Qwen models: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Inference Comparison Setup Test")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check dependencies
    if not check_dependencies():
        all_checks_passed = False
    
    # Check CUDA
    cuda_available = check_cuda()
    
    # Test model loading
    if not test_model_loading():
        all_checks_passed = False
    
    # Check Qwen models (optional - they might not be cached)
    qwen_available = check_qwen_models()
    if not qwen_available:
        print("  ⚠️  Qwen models not cached - they will be downloaded on first use")
    
    # Final summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✅ Setup test PASSED!")
        print("🚀 You can run the inference comparison experiments.")
        
        if cuda_available:
            print("💡 CUDA is available - experiments will run on GPU")
        else:
            print("💡 CUDA not available - experiments will run on CPU (slower)")
            
        print("\nNext steps:")
        print("  python inference_comparison.py --quick")
        print("  python inference_comparison.py --interactive")
        
    else:
        print("❌ Setup test FAILED!")
        print("🔧 Please fix the issues above before running experiments.")
        sys.exit(1)

if __name__ == "__main__":
    main()
