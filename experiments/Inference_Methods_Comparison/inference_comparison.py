#!/usr/bin/env python3
"""
Interactive Inference Methods Comparison Script

This script compares different inference methods for transformer models:
- Speculative decoding
- Beam search with multiple sequences
- Batching strategies
- Different sampling methods
- Cache implementations

Usage:
    python inference_comparison.py
"""

import torch
import time
import json
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
import psutil
import gc
from pathlib import Path

@dataclass
class InferenceConfig:
    """Configuration for inference experiments"""
    # Model settings
    main_model: str = "Qwen/Qwen3-4B"
    assistant_model: str = "Qwen/Qwen3-0.6B"

    # Generation settings
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # Experiment settings
    num_completions: int = 3
    prompts: List[str] = None
    batch_size: int = None  # Auto-determined if None, used for batched methods

    # Performance settings
    use_quantization: bool = False
    use_flash_attention: bool = True
    use_speculative_decoding: bool = False
    cache_implementation: str = "dynamic"  # dynamic, static, offloaded, quantized
    
    def __post_init__(self):
        if self.prompts is None:
            self.prompts = [
                "generate short (up to 150 words) bed time story containing word elephant and cake",
                "generate short (up to 150 words) bed time story containing word dog and sausage"
            ]

@dataclass
class InferenceResult:
    """Results from an inference experiment"""
    method: str
    prompt_index: int
    completion_index: int
    generated_text: str
    generation_time: float
    memory_used_gb: float
    tokens_generated: int
    tokens_per_second: float
    prompt: str

class InferenceMethodsComparator:
    """Main class for comparing different inference methods"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.tokenizer = None
        self.main_model = None
        self.assistant_model = None
        self.results: List[InferenceResult] = []
        
    def setup_models(self):
        """Initialize models and tokenizer"""
        print(f"Loading tokenizer from {self.config.main_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.main_model)
        
        # Setup quantization if requested
        quantization_config = None
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        # Load main model
        print(f"Loading main model {self.config.main_model}...")
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "quantization_config": quantization_config
        }
        
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        self.main_model = AutoModelForCausalLM.from_pretrained(
            self.config.main_model, 
            **model_kwargs
        )
        
        # Load assistant model for speculative decoding
        print(f"Loading assistant model {self.config.assistant_model}...")
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            self.config.assistant_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Set cache implementation
        if self.config.cache_implementation == "static":
            self.main_model.generation_config.cache_implementation = "static"
            
        print("Models loaded successfully!")
        
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return psutil.virtual_memory().used / (1024**3)
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def prepare_input(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Prepare input for generation with chat template"""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Always use non-thinking mode as requested
        )
        
        return self.tokenizer(text, return_tensors="pt").to(self.main_model.device)
    
    def method_standard_generation(self, prompt: str, completion_idx: int) -> InferenceResult:
        """Standard autoregressive generation"""
        self.clear_memory()
        inputs = self.prepare_input(prompt)

        start_time = time.time()
        start_memory = self.get_memory_usage()

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }

        # Add speculative decoding if enabled
        if self.config.use_speculative_decoding and self.assistant_model is not None:
            generation_kwargs["assistant_model"] = self.assistant_model

        with torch.no_grad():
            outputs = self.main_model.generate(**inputs, **generation_kwargs)
        
        generation_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory
        
        # Extract generated text
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tokens_generated = len(generated_tokens)
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        return InferenceResult(
            method="standard",
            prompt_index=0,  # Will be set by caller
            completion_index=completion_idx,
            generated_text=generated_text,
            generation_time=generation_time,
            memory_used_gb=memory_used,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            prompt=prompt
        )
    


    def method_beam_search(self, prompt: str, num_sequences: int = 3) -> List[InferenceResult]:
        """Beam search with multiple return sequences"""
        self.clear_memory()
        inputs = self.prepare_input(prompt)

        start_time = time.time()
        start_memory = self.get_memory_usage()

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "num_beams": num_sequences,
            "num_return_sequences": num_sequences,
            "early_stopping": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }

        # Add speculative decoding if enabled
        if self.config.use_speculative_decoding and self.assistant_model is not None:
            generation_kwargs["assistant_model"] = self.assistant_model

        with torch.no_grad():
            outputs = self.main_model.generate(**inputs, **generation_kwargs)

        generation_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        results = []
        input_length = inputs["input_ids"].shape[1]

        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            tokens_generated = len(generated_tokens)
            # Divide time by number of sequences for per-sequence timing
            tokens_per_second = tokens_generated / (generation_time / num_sequences) if generation_time > 0 else 0

            results.append(InferenceResult(
                method="beam_search",
                prompt_index=0,  # Will be set by caller
                completion_index=i,
                generated_text=generated_text,
                generation_time=generation_time / num_sequences,
                memory_used_gb=memory_used / num_sequences,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                prompt=prompt
            ))

        return results

    def method_batched_generation(self, prompt: str, batch_size: int = None) -> List[InferenceResult]:
        """Batched generation (repeating same prompt)"""
        self.clear_memory()

        # Use configured batch_size or default to num_completions
        if batch_size is None:
            batch_size = self.config.batch_size or self.config.num_completions

        # Create batch by repeating the same prompt
        prompts_batch = [prompt] * batch_size

        # Prepare batch inputs
        messages_batch = [[{"role": "user", "content": p}] for p in prompts_batch]
        texts_batch = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            ) for messages in messages_batch
        ]

        inputs = self.tokenizer(
            texts_batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.main_model.device)

        start_time = time.time()
        start_memory = self.get_memory_usage()

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }

        # Add speculative decoding if enabled
        if self.config.use_speculative_decoding and self.assistant_model is not None:
            generation_kwargs["assistant_model"] = self.assistant_model

        with torch.no_grad():
            outputs = self.main_model.generate(**inputs, **generation_kwargs)

        generation_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        results = []

        for i, output in enumerate(outputs):
            # Find input length for this sequence
            input_length = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum().item()
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            tokens_generated = len(generated_tokens)
            tokens_per_second = tokens_generated / (generation_time / batch_size) if generation_time > 0 else 0

            results.append(InferenceResult(
                method="batched",
                prompt_index=0,  # Will be set by caller
                completion_index=i,
                generated_text=generated_text,
                generation_time=generation_time / batch_size,
                memory_used_gb=memory_used / batch_size,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                prompt=prompt
            ))

        return results

    def method_nucleus_sampling(self, prompt: str, completion_idx: int) -> InferenceResult:
        """Nucleus (top-p) sampling"""
        self.clear_memory()
        inputs = self.prepare_input(prompt)

        start_time = time.time()
        start_memory = self.get_memory_usage()

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.8,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }

        # Add speculative decoding if enabled
        if self.config.use_speculative_decoding and self.assistant_model is not None:
            generation_kwargs["assistant_model"] = self.assistant_model

        with torch.no_grad():
            outputs = self.main_model.generate(**inputs, **generation_kwargs)

        generation_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Extract generated text
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        tokens_generated = len(generated_tokens)
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

        return InferenceResult(
            method="nucleus_sampling",
            prompt_index=0,  # Will be set by caller
            completion_index=completion_idx,
            generated_text=generated_text,
            generation_time=generation_time,
            memory_used_gb=memory_used,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            prompt=prompt
        )

    def run_experiment(self, method: str, prompt: str, prompt_idx: int, num_completions: int) -> List[InferenceResult]:
        """Run experiment for a specific method and prompt"""
        speculative_suffix = " + speculative" if self.config.use_speculative_decoding else ""
        print(f"\n🔄 Running {method}{speculative_suffix} for prompt {prompt_idx + 1}: '{prompt[:50]}...'")

        results = []

        if method == "standard":
            for i in range(num_completions):
                print(f"  Completion {i + 1}/{num_completions}...")
                result = self.method_standard_generation(prompt, i)
                result.prompt_index = prompt_idx
                # Update method name to reflect speculative decoding usage
                if self.config.use_speculative_decoding:
                    result.method = "standard_speculative"
                results.append(result)

        elif method == "beam_search":
            print(f"  Generating {num_completions} sequences with beam search...")
            beam_results = self.method_beam_search(prompt, num_completions)
            for result in beam_results:
                result.prompt_index = prompt_idx
                # Update method name to reflect speculative decoding usage
                if self.config.use_speculative_decoding:
                    result.method = "beam_search_speculative"
            results.extend(beam_results)

        elif method == "batched":
            print(f"  Generating {num_completions} completions in batch...")
            batch_results = self.method_batched_generation(prompt, num_completions)
            for result in batch_results:
                result.prompt_index = prompt_idx
                # Update method name to reflect speculative decoding usage
                if self.config.use_speculative_decoding:
                    result.method = "batched_speculative"
            results.extend(batch_results)

        elif method == "nucleus_sampling":
            for i in range(num_completions):
                print(f"  Completion {i + 1}/{num_completions}...")
                result = self.method_nucleus_sampling(prompt, i)
                result.prompt_index = prompt_idx
                # Update method name to reflect speculative decoding usage
                if self.config.use_speculative_decoding:
                    result.method = "nucleus_sampling_speculative"
                results.append(result)

        return results

    def run_all_experiments(self) -> Dict[str, List[InferenceResult]]:
        """Run all experiments for all methods and prompts"""
        methods = ["standard", "beam_search", "batched", "nucleus_sampling"]
        all_results = {method: [] for method in methods}

        print(f"\n🚀 Starting inference comparison experiments...")
        print(f"📊 Configuration:")
        print(f"  - Main model: {self.config.main_model}")
        print(f"  - Assistant model: {self.config.assistant_model}")
        print(f"  - Number of prompts: {len(self.config.prompts)}")
        print(f"  - Completions per prompt: {self.config.num_completions}")
        print(f"  - Batch size: {self.config.batch_size or 'auto'}")
        print(f"  - Max new tokens: {self.config.max_new_tokens}")
        print(f"  - Speculative decoding: {self.config.use_speculative_decoding}")
        print(f"  - Cache implementation: {self.config.cache_implementation}")
        print(f"  - Quantization: {self.config.use_quantization}")

        for method in methods:
            print(f"\n{'='*60}")
            print(f"🔬 Testing method: {method.upper()}")
            print(f"{'='*60}")

            for prompt_idx, prompt in enumerate(self.config.prompts):
                results = self.run_experiment(method, prompt, prompt_idx, self.config.num_completions)
                all_results[method].extend(results)
                self.results.extend(results)

        return all_results

    def analyze_results(self, results: Dict[str, List[InferenceResult]]) -> Dict[str, Any]:
        """Analyze and summarize results"""
        analysis = {}

        for method, method_results in results.items():
            if not method_results:
                continue

            times = [r.generation_time for r in method_results]
            memories = [r.memory_used_gb for r in method_results]
            tokens_per_sec = [r.tokens_per_second for r in method_results]
            token_counts = [r.tokens_generated for r in method_results]

            analysis[method] = {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_memory": sum(memories) / len(memories),
                "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec),
                "avg_tokens_generated": sum(token_counts) / len(token_counts),
                "total_completions": len(method_results)
            }

        return analysis

    def print_results_summary(self, analysis: Dict[str, Any]):
        """Print a formatted summary of results"""
        print(f"\n{'='*80}")
        print(f"📈 RESULTS SUMMARY")
        print(f"{'='*80}")

        # Create table header
        print(f"{'Method':<20} {'Avg Time (s)':<12} {'Tokens/sec':<12} {'Memory (GB)':<12} {'Completions':<12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        # Sort methods by average tokens per second (descending)
        sorted_methods = sorted(analysis.items(), key=lambda x: x[1]['avg_tokens_per_sec'], reverse=True)

        for method, stats in sorted_methods:
            print(f"{method:<20} {stats['avg_time']:<12.2f} {stats['avg_tokens_per_sec']:<12.1f} "
                  f"{stats['avg_memory']:<12.2f} {stats['total_completions']:<12}")

        print(f"\n🏆 Performance Rankings:")
        print(f"1. Fastest (tokens/sec): {sorted_methods[0][0]} ({sorted_methods[0][1]['avg_tokens_per_sec']:.1f} tok/s)")

        # Find most memory efficient
        memory_sorted = sorted(analysis.items(), key=lambda x: x[1]['avg_memory'])
        print(f"2. Most memory efficient: {memory_sorted[0][0]} ({memory_sorted[0][1]['avg_memory']:.2f} GB)")

        # Find fastest total time
        time_sorted = sorted(analysis.items(), key=lambda x: x[1]['avg_time'])
        print(f"3. Fastest total time: {time_sorted[0][0]} ({time_sorted[0][1]['avg_time']:.2f} s)")

    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"inference_results_{timestamp}.json"

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append(asdict(result))

        output_data = {
            "config": asdict(self.config),
            "results": serializable_results,
            "timestamp": time.time()
        }

        output_path = Path("experiments") / filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")
        return output_path


def interactive_config() -> InferenceConfig:
    """Interactive configuration setup"""
    print("🔧 Interactive Configuration Setup")
    print("=" * 50)

    config = InferenceConfig()

    # Model selection
    print(f"\n1. Model Configuration")
    use_default_models = input(f"Use default models? (Qwen3-4B + Qwen3-0.6B) [Y/n]: ").strip().lower()
    if use_default_models not in ['', 'y', 'yes']:
        config.main_model = input("Main model name: ").strip()
        config.assistant_model = input("Assistant model name: ").strip()

    # Generation settings
    print(f"\n2. Generation Settings")
    try:
        max_tokens = input(f"Max new tokens [{config.max_new_tokens}]: ").strip()
        if max_tokens:
            config.max_new_tokens = int(max_tokens)

        temperature = input(f"Temperature [{config.temperature}]: ").strip()
        if temperature:
            config.temperature = float(temperature)
    except ValueError:
        print("Invalid input, using defaults")

    # Experiment settings
    print(f"\n3. Experiment Settings")
    try:
        num_completions = input(f"Number of completions per prompt [{config.num_completions}]: ").strip()
        if num_completions:
            config.num_completions = int(num_completions)

        batch_size = input(f"Batch size (auto if empty) [{config.batch_size or 'auto'}]: ").strip()
        if batch_size and batch_size.lower() != 'auto':
            config.batch_size = int(batch_size)
    except ValueError:
        print("Invalid input, using defaults")

    # Prompts
    print(f"\n4. Prompts Configuration")
    use_default_prompts = input("Use default prompts? [Y/n]: ").strip().lower()
    if use_default_prompts not in ['', 'y', 'yes']:
        prompts = []
        print("Enter prompts (empty line to finish):")
        while True:
            prompt = input(f"Prompt {len(prompts) + 1}: ").strip()
            if not prompt:
                break
            prompts.append(prompt)
        if prompts:
            config.prompts = prompts

    # Performance settings
    print(f"\n5. Performance Settings")
    config.use_quantization = input("Use 4-bit quantization? [y/N]: ").strip().lower() in ['y', 'yes']
    config.use_flash_attention = input("Use FlashAttention-2? [Y/n]: ").strip().lower() not in ['n', 'no']
    config.use_speculative_decoding = input("Use speculative decoding? [y/N]: ").strip().lower() in ['y', 'yes']

    cache_options = ["dynamic", "static", "offloaded", "quantized"]
    print(f"Cache implementation options: {', '.join(cache_options)}")
    cache_choice = input(f"Cache implementation [{config.cache_implementation}]: ").strip()
    if cache_choice in cache_options:
        config.cache_implementation = cache_choice

    return config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare transformer inference methods")
    parser.add_argument("--config", type=str, help="JSON config file path")
    parser.add_argument("--interactive", action="store_true", help="Interactive configuration")
    parser.add_argument("--quick", action="store_true", help="Quick test with minimal settings")
    parser.add_argument("--output", type=str, help="Output file name")

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = InferenceConfig(**config_dict)
    elif args.interactive:
        config = interactive_config()
    elif args.quick:
        config = InferenceConfig(
            num_completions=2,
            max_new_tokens=50,
            prompts=["Tell me a short story about a cat"]
        )
    else:
        config = InferenceConfig()

    print(f"\n🎯 Final Configuration:")
    print(f"  Main model: {config.main_model}")
    print(f"  Assistant model: {config.assistant_model}")
    print(f"  Completions per prompt: {config.num_completions}")
    print(f"  Batch size: {config.batch_size or 'auto'}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Number of prompts: {len(config.prompts)}")
    print(f"  Speculative decoding: {config.use_speculative_decoding}")
    print(f"  Quantization: {config.use_quantization}")
    print(f"  FlashAttention: {config.use_flash_attention}")
    print(f"  Cache: {config.cache_implementation}")

    # Initialize comparator
    comparator = InferenceMethodsComparator(config)

    try:
        # Setup models
        comparator.setup_models()

        # Run experiments
        results = comparator.run_all_experiments()

        # Analyze results
        analysis = comparator.analyze_results(results)
        comparator.print_results_summary(analysis)

        # Save results
        output_file = args.output if args.output else None
        comparator.save_results(output_file)

        # Show sample outputs
        print(f"\n📝 Sample Outputs:")
        print("=" * 50)
        for method in ["standard", "beam_search", "batched"]:
            method_results = results.get(method, [])
            if method_results:
                sample = method_results[0]
                method_display = sample.method.upper()
                print(f"\n{method_display}:")
                print(f"Prompt: {sample.prompt}")
                print(f"Output: {sample.generated_text[:200]}...")
                print(f"Time: {sample.generation_time:.2f}s, Tokens/s: {sample.tokens_per_second:.1f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
