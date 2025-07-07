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
from typing import List, Dict, Any, Tuple, Union
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
    num_completions: Union[int, List[int]] = 3  # Number of generations per prompt (int for all, list per prompt)
    prompts: List[str] = None
    batch_size: int = 4  # GPU capacity - how many prompts to process simultaneously
    methods_to_test: List[str] = None  # If None, test all methods

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

        # Validate num_completions
        if isinstance(self.num_completions, list):
            if len(self.num_completions) != len(self.prompts):
                raise ValueError(f"num_completions list length ({len(self.num_completions)}) must match prompts length ({len(self.prompts)})")

        # Set default batch_size if None
        if self.batch_size is None:
            self.batch_size = 4

        # Validate methods_to_test
        if self.methods_to_test is not None:
            valid_methods = {"standard", "simple", "beam_search", "batched", "nucleus_sampling"}
            invalid_methods = set(self.methods_to_test) - valid_methods
            if invalid_methods:
                raise ValueError(f"Invalid method(s) specified: {sorted(invalid_methods)}. "
                               f"Valid methods are: {sorted(valid_methods)}")

            # Check for duplicates
            if len(self.methods_to_test) != len(set(self.methods_to_test)):
                duplicates = [method for method in set(self.methods_to_test)
                            if self.methods_to_test.count(method) > 1]
                raise ValueError(f"Duplicate method(s) specified: {sorted(duplicates)}")

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
        
        # Load assistant model for speculative decoding (only if different from main model)
        if self.config.use_speculative_decoding and self.config.assistant_model != self.config.main_model:
            print(f"Loading assistant model {self.config.assistant_model}...")
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                self.config.assistant_model,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            print("Skipping assistant model (same as main model or speculative decoding disabled)")
            self.assistant_model = None
        
        # Set cache implementation
        if self.config.cache_implementation == "static":
            self.main_model.generation_config.cache_implementation = "static"
            
        print("Models loaded successfully!")

    def get_completions_for_prompt(self, prompt_idx: int) -> int:
        """Get number of completions for a specific prompt"""
        if isinstance(self.config.num_completions, list):
            return self.config.num_completions[prompt_idx]
        return self.config.num_completions

    def get_total_completions(self) -> int:
        """Get total number of completions across all prompts"""
        if isinstance(self.config.num_completions, list):
            return sum(self.config.num_completions)
        return self.config.num_completions * len(self.config.prompts)

    def create_generation_batches(self) -> List[Tuple[str, int, int]]:
        """Create batches for generation: (prompt, prompt_idx, completion_idx)"""
        batches = []
        current_batch = []

        # Create all (prompt, prompt_idx, completion_idx) tuples
        all_generations = []
        for prompt_idx, prompt in enumerate(self.config.prompts):
            num_comps = self.get_completions_for_prompt(prompt_idx)
            for comp_idx in range(num_comps):
                all_generations.append((prompt, prompt_idx, comp_idx))

        # Group into batches
        for i in range(0, len(all_generations), self.config.batch_size):
            batch = all_generations[i:i + self.config.batch_size]
            batches.append(batch)

        return batches

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            # Use max_memory_allocated to get peak memory usage during generation
            return torch.cuda.max_memory_allocated() / (1024**3)
        return psutil.virtual_memory().used / (1024**3)
    
    def clear_memory(self):
        """Clear GPU memory cache and reset memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking
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
    

    


    def method_beam_search(self, prompt: str, prompt_idx: int, num_sequences: int) -> List[InferenceResult]:
        """Beam search with multiple return sequences for a single prompt"""
        results = []

        # If num_sequences > batch_size, we need multiple beam search calls
        remaining_sequences = num_sequences
        completion_idx = 0

        while remaining_sequences > 0:
            current_sequences = min(remaining_sequences, self.config.batch_size)

            # Only clear memory at the start of the first iteration
            if remaining_sequences == num_sequences:
                self.clear_memory()

            inputs = self.prepare_input(prompt)

            start_time = time.time()

            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "num_beams": current_sequences,
                "num_return_sequences": current_sequences,
                "early_stopping": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }

            # Add speculative decoding if enabled
            if self.config.use_speculative_decoding and self.assistant_model is not None:
                generation_kwargs["assistant_model"] = self.assistant_model

            try:
                with torch.no_grad():
                    outputs = self.main_model.generate(**inputs, **generation_kwargs)
            except Exception as e:
                print(f"❌ Error during beam search generation: {e}")
                raise

            generation_time = time.time() - start_time
            memory_used = self.get_memory_usage()

            input_length = inputs["input_ids"].shape[1]

            for i, output in enumerate(outputs):
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                tokens_generated = len(generated_tokens)
                per_sequence_time = generation_time / current_sequences if generation_time > 0 else 0
                tokens_per_second = tokens_generated / per_sequence_time if per_sequence_time > 0 else 0

                results.append(InferenceResult(
                    method="beam_search",
                    prompt_index=prompt_idx,
                    completion_index=completion_idx,
                    generated_text=generated_text,
                    generation_time=per_sequence_time,
                    memory_used_gb=memory_used,  # Will be updated by run_experiment with method-wide memory
                    tokens_generated=tokens_generated,
                    tokens_per_second=tokens_per_second,
                    prompt=prompt
                ))

                completion_idx += 1

            remaining_sequences -= current_sequences

        return results

    def method_standard_generation(self) -> List[InferenceResult]:
        """Standard generation (one at a time, no memory reset between items)"""
        # Get all generation tasks
        all_generations = []
        for prompt_idx, prompt in enumerate(self.config.prompts):
            num_comps = self.get_completions_for_prompt(prompt_idx)
            for comp_idx in range(num_comps):
                all_generations.append((prompt, prompt_idx, comp_idx))

        results = []

        # Process each generation individually (but don't reset memory between them)
        for prompt, prompt_idx, completion_idx in all_generations:
            # Prepare single input
            inputs = self.prepare_input(prompt)

            start_time = time.time()

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

            try:
                with torch.no_grad():
                    outputs = self.main_model.generate(**inputs, **generation_kwargs)
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                raise

            generation_time = time.time() - start_time

            # Extract generated text
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            tokens_generated = len(generated_tokens)
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

            results.append(InferenceResult(
                method="standard",
                prompt_index=prompt_idx,
                completion_index=completion_idx,
                generated_text=generated_text,
                generation_time=generation_time,
                memory_used_gb=0.0,  # Will be updated by run_experiment with method-wide memory
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                prompt=prompt
            ))

        return results

    def method_batched_generation(self, batch: List[Tuple[str, int, int]]) -> List[InferenceResult]:
        """Batched generation with mixed prompts"""
        # Memory is managed by run_experiment method

        # Extract prompts from batch
        prompts_batch = [item[0] for item in batch]

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

        try:
            with torch.no_grad():
                outputs = self.main_model.generate(**inputs, **generation_kwargs)
        except Exception as e:
            print(f"❌ Error during batched generation: {e}")
            raise

        generation_time = time.time() - start_time
        memory_used = self.get_memory_usage()  # Get peak memory used during generation

        results = []

        for i, (output, (prompt, prompt_idx, completion_idx)) in enumerate(zip(outputs, batch)):
            # Find input length for this sequence
            input_length = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum().item()
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            tokens_generated = len(generated_tokens)
            # Calculate per-item timing
            per_item_time = generation_time / len(batch) if generation_time > 0 else 0
            tokens_per_second = tokens_generated / per_item_time if per_item_time > 0 else 0

            results.append(InferenceResult(
                method="batched",
                prompt_index=prompt_idx,
                completion_index=completion_idx,
                generated_text=generated_text,
                generation_time=per_item_time,
                memory_used_gb=memory_used,  # Will be updated by run_experiment with method-wide memory
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                prompt=prompt
            ))

        return results

    def method_nucleus_sampling(self) -> List[InferenceResult]:
        """Nucleus (top-p) sampling using batched generation with modified parameters"""
        # Create batches
        batches = self.create_generation_batches()
        results = []

        for batch in batches:
            # Temporarily modify generation parameters for nucleus sampling
            original_temp = self.config.temperature
            original_top_p = self.config.top_p

            self.config.temperature = 0.8
            self.config.top_p = 0.9

            try:
                batch_results = self.method_batched_generation(batch)
                # Update method name
                for result in batch_results:
                    result.method = "nucleus_sampling"
                results.extend(batch_results)
            finally:
                # Restore original parameters
                self.config.temperature = original_temp
                self.config.top_p = original_top_p

        return results

    def run_experiment(self, method: str) -> List[InferenceResult]:
        """Run experiment for a specific method across all prompts"""
        speculative_suffix = " + speculative" if self.config.use_speculative_decoding else ""
        print(f"\n🔄 Running {method}{speculative_suffix}...")

        # Start method-wide timing and memory tracking
        method_start_time = time.time()
        self.clear_memory()  # Reset memory tracking for this method

        results = []

        if method == "standard" or method == "simple":
            results = self.method_standard_generation()
            # Update method name to match config
            for result in results:
                result.method = method

        elif method == "beam_search":
            # Beam search processes prompts one by one
            for prompt_idx, prompt in enumerate(self.config.prompts):
                num_completions = self.get_completions_for_prompt(prompt_idx)
                print(f"  Beam search for prompt {prompt_idx + 1}: {num_completions} sequences")
                beam_results = self.method_beam_search(prompt, prompt_idx, num_completions)
                results.extend(beam_results)

        elif method == "batched":
            # Process all batches
            batches = self.create_generation_batches()
            results = []
            for batch_idx, batch in enumerate(batches):
                print(f"  Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} items)")
                batch_results = self.method_batched_generation(batch)
                results.extend(batch_results)

        elif method == "nucleus_sampling":
            results = self.method_nucleus_sampling()

        # Calculate method-wide metrics
        method_total_time = time.time() - method_start_time
        method_peak_memory = self.get_memory_usage()

        # Update all results with method-wide memory measurement
        for result in results:
            result.memory_used_gb = method_peak_memory

        print(f"  ✅ {method} completed in {method_total_time:.2f}s, peak memory: {method_peak_memory:.3f}GB")

        # Update method names for speculative decoding
        if self.config.use_speculative_decoding:
            for result in results:
                result.method = f"{result.method}_speculative"

        return results

    def run_all_experiments(self) -> Dict[str, List[InferenceResult]]:
        """Run all experiments for all methods and prompts"""
        all_methods = ["standard", "simple", "beam_search", "batched", "nucleus_sampling"]

        # If methods_to_test is explicitly set (not None), use it; otherwise use all methods
        if self.config.methods_to_test is not None:
            methods = self.config.methods_to_test
        else:
            methods = all_methods

        all_results = {method: [] for method in methods}

        print(f"\n🚀 Starting inference comparison experiments...")
        print(f"📊 Configuration:")
        print(f"  - Main model: {self.config.main_model}")
        print(f"  - Assistant model: {self.config.assistant_model}")
        print(f"  - Number of prompts: {len(self.config.prompts)}")
        print(f"  - Completions per prompt: {self.config.num_completions}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Max new tokens: {self.config.max_new_tokens}")
        print(f"  - Speculative decoding: {self.config.use_speculative_decoding}")
        print(f"  - Cache implementation: {self.config.cache_implementation}")
        print(f"  - Quantization: {self.config.use_quantization}")
        print(f"  - Methods to test: {methods}")

        # Calculate expected total results
        total_expected = len(methods) * self.get_total_completions()
        print(f"  - Expected total results: {total_expected}")
        print(f"    ({len(methods)} methods × {self.get_total_completions()} total completions)")

        for method in methods:
            print(f"\n{'='*60}")
            print(f"🔬 Testing method: {method.upper()}")
            print(f"{'='*60}")

            results = self.run_experiment(method)
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
        
        output_path = Path("/teamspace/studios/this_studio/silly-llm/experiments") / filename
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
    parser.add_argument("--methods", nargs="+", choices=["standard", "simple", "beam_search", "batched", "nucleus_sampling"],
                       help="Specific methods to test")

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)

            # Handle null values properly - convert None to empty list for methods_to_test
            if config_dict.get('methods_to_test') is None:
                config_dict['methods_to_test'] = None  # Keep as None to indicate "use default"

            config = InferenceConfig(**config_dict)
        except FileNotFoundError:
            print(f"❌ Error: Configuration file '{args.config}' not found.")
            return
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in configuration file '{args.config}': {e}")
            return
        except ValueError as e:
            print(f"❌ Configuration Error: {e}")
            return
    elif args.interactive:
        config = interactive_config()
    elif args.quick:
        config = InferenceConfig(
            main_model="Qwen/Qwen3-4B",
            assistant_model="Qwen/Qwen3-0.6B",
            num_completions=2,
            max_new_tokens=50,
            prompts=["Tell me a short story about a cat"],
            use_quantization=True,
            use_flash_attention=True,
            use_speculative_decoding=True,
            methods_to_test=["standard", "batched"]  # Test only 2 methods for speed
        )
    else:
        config = InferenceConfig()

    # Override methods if specified
    if args.methods:
        try:
            # Validate methods before setting
            valid_methods = {"standard", "simple", "beam_search", "batched", "nucleus_sampling"}
            invalid_methods = set(args.methods) - valid_methods
            if invalid_methods:
                print(f"❌ Error: Invalid method(s) specified: {sorted(invalid_methods)}")
                print(f"Valid methods are: {sorted(valid_methods)}")
                return
            config.methods_to_test = args.methods
        except Exception as e:
            print(f"❌ Error setting methods: {e}")
            return

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
