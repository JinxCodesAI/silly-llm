# Transformers Library Development Guide

## Creating Highly Efficient, Parallelizable, and Maintainable Code for Fast Inference and Training

This comprehensive guide provides best practices for developing efficient transformer models using the Hugging Face ecosystem, with a primary focus on inference optimization and insights drawn from the TinyStories methodology and modern optimization techniques.

## Table of Contents

1. [Inference Optimization](#inference-optimization)
2. [Generation Strategies and Decoding Methods](#generation-strategies-and-decoding-methods)
3. [Advanced Inference Techniques](#advanced-inference-techniques)
4. [KV Cache Strategies and Memory Management](#kv-cache-strategies-and-memory-management)
5. [Quantization and Model Compression](#quantization-and-model-compression)
6. [Architecture Design for Efficient Inference](#architecture-design-for-efficient-inference)
7. [Multi-GPU Training Strategies](#multi-gpu-training-strategies)
8. [Batch Processing and Data Loading](#batch-processing-and-data-loading)
9. [Code Organization and Modularity](#code-organization-and-modularity)
10. [Performance Monitoring and Debugging](#performance-monitoring-and-debugging)
11. [Production Deployment](#production-deployment)

## Inference Optimization

Inference optimization is crucial for deploying transformer models in production environments. Modern LLMs require sophisticated optimization techniques to achieve acceptable latency and memory usage while maintaining quality.

### 1. Static KV Cache and torch.compile

The combination of static KV cache with `torch.compile` provides up to 4x speed improvements for inference:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Prevent tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_optimized_model(model_name="microsoft/DialoGPT-medium"):
    """Setup model with static cache and torch.compile optimization"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Enable static cache
    model.generation_config.cache_implementation = "static"

    # Compile the forward pass
    model.forward = torch.compile(
        model.forward,
        mode="reduce-overhead",
        fullgraph=True
    )

    return model, tokenizer

# Usage example
model, tokenizer = setup_optimized_model()
input_text = "The theory of special relativity states"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

# Optimized generation
outputs = model.generate(**input_ids, max_new_tokens=50)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### 2. Memory-Efficient Attention Mechanisms

#### FlashAttention-2 Integration

FlashAttention-2 provides significant memory savings and speed improvements:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def create_flash_attention_model(model_name, use_quantization=True):
    """Create model with FlashAttention-2 and optional quantization"""

    config_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto"
    }

    if use_quantization:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        config_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **config_kwargs)
    return model

# Example usage
model = create_flash_attention_model("microsoft/DialoGPT-medium")
```

#### SDPA (Scaled Dot Product Attention)

PyTorch's native SDPA automatically chooses the best attention implementation:

```python
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

def generate_with_sdpa(model, tokenizer, prompt, backend="flash_attention"):
    """Generate text using specific SDPA backend"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Map backend names to SDPBackend enums
    backend_map = {
        "flash_attention": SDPBackend.FLASH_ATTENTION,
        "memory_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": SDPBackend.CUDNN_ATTENTION
    }

    with sdpa_kernel(backend_map[backend]):
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
result = generate_with_sdpa(model, tokenizer, "Hello, how are you?")
```

### 3. Advanced Memory Management

#### Dynamic Memory Allocation

```python
class MemoryOptimizedInference:
    def __init__(self, model, tokenizer, max_memory_gb=16):
        self.model = model
        self.tokenizer = tokenizer
        self.max_memory_gb = max_memory_gb
        self.memory_tracker = []

    def get_memory_usage(self):
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0

    def adaptive_batch_size(self, base_batch_size=8):
        """Dynamically adjust batch size based on memory usage"""
        current_memory = self.get_memory_usage()
        memory_ratio = current_memory / self.max_memory_gb

        if memory_ratio > 0.8:
            return max(1, base_batch_size // 2)
        elif memory_ratio < 0.4:
            return min(32, base_batch_size * 2)
        return base_batch_size

    def generate_with_memory_management(self, prompts, max_new_tokens=50):
        """Generate text with automatic memory management"""
        results = []
        batch_size = self.adaptive_batch_size()

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            # Clear cache if memory usage is high
            if self.get_memory_usage() > self.max_memory_gb * 0.7:
                torch.cuda.empty_cache()

            # Process batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            batch_results = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            results.extend(batch_results)

            # Track memory usage
            self.memory_tracker.append(self.get_memory_usage())

        return results

# Usage
memory_optimizer = MemoryOptimizedInference(model, tokenizer)
prompts = ["Hello", "How are you?", "What's the weather?"] * 10
results = memory_optimizer.generate_with_memory_management(prompts)
```

## Generation Strategies and Decoding Methods

### 1. Speculative Decoding

Speculative decoding uses a smaller assistant model to generate candidate tokens that are verified by the main model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    def __init__(self, main_model_name, assistant_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(main_model_name)

        # Load main model
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Load smaller assistant model
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def generate_with_speculation(self, prompt, max_new_tokens=100):
        """Generate text using speculative decoding"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.main_model.device)

        # Use assistant model for speculation
        outputs = self.main_model.generate(
            **inputs,
            assistant_model=self.assistant_model,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
speculative_decoder = SpeculativeDecoder(
    main_model_name="microsoft/DialoGPT-large",
    assistant_model_name="microsoft/DialoGPT-small"
)

result = speculative_decoder.generate_with_speculation(
    "The future of AI is",
    max_new_tokens=50
)
```

### 2. Prompt Lookup Decoding

Prompt lookup decoding is effective for input-grounded tasks where there's overlap between prompt and output:

```python
def generate_with_prompt_lookup(model, tokenizer, prompt, num_lookup_tokens=3):
    """Generate using prompt lookup decoding"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        prompt_lookup_num_tokens=num_lookup_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example for summarization task
prompt = """
Article: The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
Summary: The sentence
"""

result = generate_with_prompt_lookup(model, tokenizer, prompt)
```

### 3. Advanced Sampling Strategies

```python
class AdvancedSampler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def nucleus_sampling(self, prompt, top_p=0.9, temperature=0.8):
        """Generate using nucleus (top-p) sampling"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def contrastive_search(self, prompt, penalty_alpha=0.6, top_k=4):
        """Generate using contrastive search"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def beam_search_with_constraints(self, prompt, num_beams=4, constraints=None):
        """Generate using beam search with optional constraints"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": 100,
            "num_beams": num_beams,
            "early_stopping": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        if constraints:
            # Add constraint handling here
            pass

        outputs = self.model.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
sampler = AdvancedSampler(model, tokenizer)
result = sampler.nucleus_sampling("The future of technology")
```

## Advanced Inference Techniques

### 1. Iterative Generation and Chat Applications

For multi-turn conversations, efficient cache management is crucial:

```python
from transformers.cache_utils import DynamicCache, StaticCache

class ChatInferenceEngine:
    def __init__(self, model, tokenizer, cache_type="dynamic"):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_type = cache_type
        self.conversation_cache = None
        self.conversation_history = []

    def initialize_cache(self, max_length=2048):
        """Initialize appropriate cache type"""
        if self.cache_type == "static":
            return StaticCache(
                config=self.model.config,
                max_batch_size=1,
                max_cache_len=max_length,
                device=self.model.device,
                dtype=self.model.dtype
            )
        else:
            return DynamicCache()

    def chat_turn(self, user_message, max_new_tokens=100):
        """Process a single chat turn efficiently"""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            self.conversation_history,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        # Initialize cache if first turn
        if self.conversation_cache is None:
            self.conversation_cache = self.initialize_cache()

        # Generate response
        input_length = prompt["input_ids"].shape[1]
        outputs = self.model.generate(
            **prompt,
            max_new_tokens=max_new_tokens,
            past_key_values=self.conversation_cache,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )

        # Extract only the new tokens
        response_tokens = outputs[0, input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_history = []
        self.conversation_cache = None

# Usage example
chat_engine = ChatInferenceEngine(model, tokenizer, cache_type="static")

# Multi-turn conversation
response1 = chat_engine.chat_turn("Hello, how are you?")
response2 = chat_engine.chat_turn("Can you help me with Python?")
response3 = chat_engine.chat_turn("Write a simple function")
```

### 2. Batch Inference Optimization

```python
class BatchInferenceOptimizer:
    def __init__(self, model, tokenizer, max_batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size

    def smart_batching(self, prompts, max_length_diff=50):
        """Group prompts by similar lengths for efficient batching"""
        # Tokenize and get lengths
        tokenized = [
            self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            for prompt in prompts
        ]

        # Sort by length
        sorted_items = sorted(zip(prompts, tokenized), key=lambda x: x[1])

        batches = []
        current_batch = []
        current_length = 0

        for prompt, length in sorted_items:
            if (not current_batch or
                (length - current_length <= max_length_diff and
                 len(current_batch) < self.max_batch_size)):
                current_batch.append(prompt)
                current_length = max(current_length, length)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [prompt]
                current_length = length

        if current_batch:
            batches.append(current_batch)

        return batches

    def process_batch(self, batch, max_new_tokens=50):
        """Process a batch of prompts efficiently"""
        # Tokenize batch with padding
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # Generate for batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode results
        results = []
        for i, output in enumerate(outputs):
            # Skip input tokens
            input_length = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum()
            generated_tokens = output[input_length:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(result)

        return results

    def batch_generate(self, prompts, max_new_tokens=50):
        """Generate responses for multiple prompts efficiently"""
        batches = self.smart_batching(prompts)
        all_results = []

        for batch in batches:
            batch_results = self.process_batch(batch, max_new_tokens)
            all_results.extend(batch_results)

        # Restore original order
        prompt_to_result = dict(zip([p for batch in batches for p in batch], all_results))
        return [prompt_to_result[prompt] for prompt in prompts]

# Usage
batch_optimizer = BatchInferenceOptimizer(model, tokenizer)
prompts = [
    "Write a short story about",
    "Explain quantum physics",
    "Create a recipe for",
    "Describe the weather"
]
results = batch_optimizer.batch_generate(prompts)
```

### 3. Streaming Generation

```python
import asyncio
from typing import AsyncGenerator, Generator

class StreamingGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def stream_generate(self, prompt: str, max_new_tokens: int = 100) -> Generator[str, None, None]:
        """Generate text with streaming output"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Initialize generation
        generated_ids = inputs["input_ids"].clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Generate next token
            with torch.no_grad():
                if past_key_values is None:
                    # First iteration - use full input
                    outputs = self.model(**inputs, use_cache=True)
                else:
                    # Subsequent iterations - use only last token
                    last_token = generated_ids[:, -1:]
                    outputs = self.model(
                        input_ids=last_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )

                past_key_values = outputs.past_key_values

                # Sample next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits / 0.7, dim=-1),
                    num_samples=1
                )

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Decode and yield new token
                new_token_text = self.tokenizer.decode(
                    next_token[0],
                    skip_special_tokens=True
                )

                yield new_token_text

                # Check for end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

    async def async_stream_generate(self, prompt: str, max_new_tokens: int = 100) -> AsyncGenerator[str, None]:
        """Async version of streaming generation"""
        for token in self.stream_generate(prompt, max_new_tokens):
            yield token
            await asyncio.sleep(0)  # Allow other tasks to run

# Usage
streaming_gen = StreamingGenerator(model, tokenizer)

# Synchronous streaming
print("Streaming generation:")
for token in streaming_gen.stream_generate("The future of AI is"):
    print(token, end="", flush=True)

# Asynchronous streaming
async def demo_async_streaming():
    async for token in streaming_gen.async_stream_generate("The future of AI is"):
        print(token, end="", flush=True)

# asyncio.run(demo_async_streaming())
```

## KV Cache Strategies and Memory Management

Efficient KV cache management is critical for transformer inference performance. Different cache strategies offer trade-offs between memory usage, speed, and compatibility with optimization techniques.

### 1. Cache Type Selection Guide

| Cache Type | Memory Efficient | torch.compile Support | Initialization Required | Best Use Case |
|------------|------------------|----------------------|------------------------|---------------|
| DynamicCache | No | No | No | Development/Testing |
| StaticCache | No | Yes | Yes | Production Inference |
| OffloadedCache | Yes | No | No | Memory-Constrained GPU |
| QuantizedCache | Yes | No | No | Long Context Generation |
| SlidingWindowCache | No | Yes | Yes | Streaming Applications |

### 2. Dynamic Cache (Default)

```python
from transformers import DynamicCache

class DynamicCacheManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_with_dynamic_cache(self, prompt, max_new_tokens=50):
        """Standard generation with dynamic cache"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Dynamic cache grows automatically
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,  # Uses DynamicCache by default
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def manual_cache_management(self, prompt, max_new_tokens=50):
        """Manual cache initialization and management"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Initialize cache manually
        past_key_values = DynamicCache()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
cache_manager = DynamicCacheManager(model, tokenizer)
result = cache_manager.generate_with_dynamic_cache("Hello world")
```

### 3. Static Cache for Production

```python
from transformers import StaticCache

class StaticCacheOptimizer:
    def __init__(self, model, tokenizer, max_cache_length=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_cache_length = max_cache_length

        # Pre-compile model with static cache
        self.model.generation_config.cache_implementation = "static"
        self.model.forward = torch.compile(
            self.model.forward,
            mode="reduce-overhead",
            fullgraph=True
        )

    def optimized_generate(self, prompt, max_new_tokens=50):
        """Generate with static cache and torch.compile"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Pad inputs to avoid recompilation
        input_length = inputs["input_ids"].shape[1]
        if input_length < 64:  # Pad to common lengths
            pad_length = 64 - input_length
            inputs["input_ids"] = torch.cat([
                inputs["input_ids"],
                torch.full((1, pad_length), self.tokenizer.pad_token_id,
                          device=self.model.device)
            ], dim=1)
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"],
                torch.zeros((1, pad_length), device=self.model.device)
            ], dim=1)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            cache_implementation="static",
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate_static(self, prompts, max_new_tokens=50):
        """Batch generation with static cache"""
        # Pad all prompts to same length for efficiency
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            pad_to_multiple_of=64  # Avoid recompilation
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            cache_implementation="static",
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Usage
static_optimizer = StaticCacheOptimizer(model, tokenizer)
result = static_optimizer.optimized_generate("The future of AI")
```

### 4. Memory-Efficient Caches

#### Offloaded Cache for GPU Memory Constraints

```python
class OffloadedCacheManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_with_offloading(self, prompt, max_new_tokens=100):
        """Generate with CPU offloading for memory efficiency"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:
            # Try normal generation first
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        except torch.cuda.OutOfMemoryError:
            print("OOM detected, falling back to offloaded cache")
            torch.cuda.empty_cache()

            # Use offloaded cache
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                cache_implementation="offloaded",
                do_sample=True,
                temperature=0.7
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def resilient_batch_generate(self, prompts, max_new_tokens=50):
        """Batch generation with automatic fallback to offloaded cache"""
        results = []

        for prompt in prompts:
            try:
                result = self.generate_with_offloading(prompt, max_new_tokens)
                results.append(result)
            except Exception as e:
                print(f"Error processing prompt: {e}")
                results.append(f"Error: {str(e)}")

        return results

# Usage
offloaded_manager = OffloadedCacheManager(model, tokenizer)
result = offloaded_manager.generate_with_offloading("Long prompt here...")
```

#### Quantized Cache for Long Contexts

```python
from transformers import QuantizedCacheConfig, HQQQuantizedCache

class QuantizedCacheManager:
    def __init__(self, model, tokenizer, backend="quanto"):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend

    def setup_quantized_cache(self, quantization_bits=4):
        """Setup quantized cache configuration"""
        if self.backend == "hqq":
            cache_config = {
                "backend": "hqq",
                "nbits": quantization_bits,
                "axis-key": 1,
                "axis-value": 1
            }
        else:  # quanto
            cache_config = {
                "backend": "quanto",
                "nbits": quantization_bits
            }

        return cache_config

    def generate_with_quantized_cache(self, prompt, max_new_tokens=200, bits=4):
        """Generate with quantized cache for memory efficiency"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        cache_config = self.setup_quantized_cache(bits)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            cache_implementation="quantized",
            cache_config=cache_config,
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def long_context_generation(self, long_prompt, max_new_tokens=500):
        """Handle very long contexts with quantized cache"""
        # Use 2-bit quantization for maximum memory savings
        return self.generate_with_quantized_cache(
            long_prompt,
            max_new_tokens,
            bits=2
        )

# Usage
quantized_manager = QuantizedCacheManager(model, tokenizer, backend="quanto")
long_prompt = "Very long context here..." * 100
result = quantized_manager.long_context_generation(long_prompt)
```

### 5. Sliding Window Cache

```python
class SlidingWindowCacheManager:
    def __init__(self, model, tokenizer, window_size=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size

        # Ensure model supports sliding window attention
        if not hasattr(model.config, 'sliding_window'):
            print("Warning: Model may not support sliding window attention")

    def generate_with_sliding_window(self, prompt, max_new_tokens=100):
        """Generate with sliding window cache"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            cache_implementation="sliding_window",
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def streaming_with_sliding_window(self, prompt, max_new_tokens=500):
        """Streaming generation with sliding window for long sequences"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated_text = prompt

        # Generate in chunks to demonstrate sliding window
        chunk_size = 50
        for i in range(0, max_new_tokens, chunk_size):
            remaining_tokens = min(chunk_size, max_new_tokens - i)

            current_inputs = self.tokenizer(
                generated_text,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **current_inputs,
                max_new_tokens=remaining_tokens,
                cache_implementation="sliding_window",
                do_sample=True,
                temperature=0.7
            )

            new_text = self.tokenizer.decode(
                outputs[0][current_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            generated_text += new_text

            # Truncate if context gets too long
            if len(generated_text.split()) > self.window_size:
                words = generated_text.split()
                generated_text = " ".join(words[-self.window_size//2:])

        return generated_text

# Usage (requires model with sliding window support like Mistral)
# sliding_manager = SlidingWindowCacheManager(mistral_model, mistral_tokenizer)
# result = sliding_manager.generate_with_sliding_window("Long conversation...")
```

## Quantization and Model Compression

Quantization reduces model memory requirements by storing weights in lower precision formats. This section covers various quantization techniques and their trade-offs.

### 1. Memory Requirements by Precision

| Precision | Memory per Billion Parameters | Use Case |
|-----------|-------------------------------|----------|
| float32 | ~4 GB | Training (rarely used) |
| bfloat16/float16 | ~2 GB | Standard inference |
| int8 | ~1 GB | Memory-constrained inference |
| int4 | ~0.5 GB | Extreme memory constraints |

### 2. BitsAndBytes Quantization

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

class QuantizationManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_8bit_model(self):
        """Load model with 8-bit quantization"""
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        return model

    def load_4bit_model(self, use_double_quant=True, quant_type="nf4"):
        """Load model with 4-bit quantization"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,  # "fp4" or "nf4"
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        return model

    def compare_memory_usage(self):
        """Compare memory usage across different quantization levels"""
        results = {}

        # Full precision
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        results["fp16"] = torch.cuda.memory_allocated() / (1024**3)
        del model_fp16
        torch.cuda.empty_cache()

        # 8-bit
        model_8bit = self.load_8bit_model()
        results["int8"] = torch.cuda.memory_allocated() / (1024**3)
        del model_8bit
        torch.cuda.empty_cache()

        # 4-bit
        model_4bit = self.load_4bit_model()
        results["int4"] = torch.cuda.memory_allocated() / (1024**3)
        del model_4bit
        torch.cuda.empty_cache()

        return results

# Usage
quant_manager = QuantizationManager("microsoft/DialoGPT-medium")
model_8bit = quant_manager.load_8bit_model()
model_4bit = quant_manager.load_4bit_model()

# Compare memory usage
memory_comparison = quant_manager.compare_memory_usage()
print("Memory usage comparison (GB):", memory_comparison)
```

### 3. Advanced Quantization Techniques

#### GPTQ Quantization

```python
from transformers import GPTQConfig, AutoModelForCausalLM

class GPTQQuantizer:
    def __init__(self, model_name, calibration_dataset=None):
        self.model_name = model_name
        self.calibration_dataset = calibration_dataset

    def setup_gptq_config(self, bits=4, group_size=128, desc_act=False):
        """Setup GPTQ quantization configuration"""
        return GPTQConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            static_groups=False,
            sym=True,
            true_sequential=True,
            model_seqlen=2048,
            block_name_to_quantize="model.decoder.layers",
            module_name_preceding_first_block=["model.decoder.embed_tokens"],
            quantize_config={
                "zero_point": True,
                "q_group_size": group_size,
            }
        )

    def quantize_model(self, bits=4):
        """Quantize model using GPTQ"""
        gptq_config = self.setup_gptq_config(bits=bits)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=gptq_config,
            device_map="auto"
        )

        return model

    def benchmark_gptq_vs_bnb(self, prompt="Hello, how are you?"):
        """Compare GPTQ vs BitsAndBytes quantization"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(prompt, return_tensors="pt")

        results = {}

        # GPTQ 4-bit
        model_gptq = self.quantize_model(bits=4)
        start_time = time.time()
        with torch.no_grad():
            outputs_gptq = model_gptq.generate(**inputs, max_new_tokens=50)
        results["gptq_time"] = time.time() - start_time
        results["gptq_memory"] = torch.cuda.memory_allocated() / (1024**3)
        del model_gptq
        torch.cuda.empty_cache()

        # BitsAndBytes 4-bit
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model_bnb = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        start_time = time.time()
        with torch.no_grad():
            outputs_bnb = model_bnb.generate(**inputs, max_new_tokens=50)
        results["bnb_time"] = time.time() - start_time
        results["bnb_memory"] = torch.cuda.memory_allocated() / (1024**3)
        del model_bnb
        torch.cuda.empty_cache()

        return results

# Usage
# gptq_quantizer = GPTQQuantizer("microsoft/DialoGPT-medium")
# benchmark_results = gptq_quantizer.benchmark_gptq_vs_bnb()
```

#### AWQ (Activation-aware Weight Quantization)

```python
from transformers import AwqConfig

class AWQQuantizer:
    def __init__(self, model_name):
        self.model_name = model_name

    def setup_awq_config(self, bits=4, group_size=128, zero_point=True):
        """Setup AWQ quantization configuration"""
        return AwqConfig(
            bits=bits,
            group_size=group_size,
            zero_point=zero_point,
            version="GEMM",
            do_fuse=True,
            fuse_max_seq_len=2048,
            modules_to_fuse={
                "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "mlp": ["gate_proj", "up_proj", "down_proj"]
            }
        )

    def load_awq_model(self, bits=4):
        """Load model with AWQ quantization"""
        awq_config = self.setup_awq_config(bits=bits)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=awq_config,
            device_map="auto"
        )

        return model

    def fused_inference(self, model, prompt, max_new_tokens=50):
        """Perform inference with fused AWQ operations"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # AWQ supports fused operations for better performance
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                use_cache=True
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
# awq_quantizer = AWQQuantizer("microsoft/DialoGPT-medium")
# awq_model = awq_quantizer.load_awq_model()
# result = awq_quantizer.fused_inference(awq_model, "Hello world")
```

### 4. Dynamic Quantization

```python
class DynamicQuantizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def apply_dynamic_quantization(self):
        """Apply dynamic quantization to the model"""
        # Dynamic quantization for linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        return quantized_model

    def benchmark_quantization_impact(self, prompts, max_new_tokens=50):
        """Benchmark the impact of quantization on performance"""
        results = {"original": {}, "quantized": {}}

        # Original model
        start_time = time.time()
        original_outputs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            original_outputs.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

        results["original"]["time"] = time.time() - start_time
        results["original"]["memory"] = torch.cuda.memory_allocated() / (1024**3)

        # Quantized model
        quantized_model = self.apply_dynamic_quantization()
        torch.cuda.empty_cache()

        start_time = time.time()
        quantized_outputs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(quantized_model.device)
            with torch.no_grad():
                outputs = quantized_model.generate(**inputs, max_new_tokens=max_new_tokens)
            quantized_outputs.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

        results["quantized"]["time"] = time.time() - start_time
        results["quantized"]["memory"] = torch.cuda.memory_allocated() / (1024**3)

        # Calculate quality metrics (simple BLEU-like comparison)
        quality_scores = []
        for orig, quant in zip(original_outputs, quantized_outputs):
            # Simple token overlap ratio
            orig_tokens = set(orig.split())
            quant_tokens = set(quant.split())
            if orig_tokens:
                overlap = len(orig_tokens.intersection(quant_tokens)) / len(orig_tokens)
                quality_scores.append(overlap)

        results["quality_retention"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return results

# Usage
dynamic_quantizer = DynamicQuantizer(model, tokenizer)
test_prompts = ["Hello world", "How are you?", "Tell me a story"]
benchmark_results = dynamic_quantizer.benchmark_quantization_impact(test_prompts)
print("Quantization benchmark:", benchmark_results)
```

## Architecture Design for Efficient Inference

Modern transformer architectures incorporate several innovations specifically designed to improve inference efficiency, particularly for long sequences and autoregressive generation.

### 1. Positional Embeddings for Long Sequences

#### Rotary Position Embedding (RoPE)

RoPE enables better extrapolation to longer sequences than seen during training:

```python
import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        # Create position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None):
        """Apply rotary position embedding to query and key tensors"""
        if position_ids is not None:
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed

class RoPEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value
```

#### ALiBi (Attention with Linear Biases)

ALiBi provides excellent extrapolation capabilities with minimal computational overhead:

```python
class ALiBiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # ALiBi slopes
        self.register_buffer("slopes", self._get_alibi_slopes(self.num_heads))

    def _get_alibi_slopes(self, num_heads):
        """Compute ALiBi slopes for each attention head"""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:num_heads-closest_power_of_2])
            return torch.tensor(slopes)

    def _get_alibi_bias(self, seq_len, device):
        """Generate ALiBi bias matrix"""
        # Create position matrix
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory_position - context_position

        # Apply slopes to create bias
        alibi_bias = relative_position[None, :, :] * self.slopes[:, None, None]
        return alibi_bias

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle past key values
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v)
        kv_seq_len = k.shape[2]

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale

        # Add ALiBi bias
        alibi_bias = self._get_alibi_bias(kv_seq_len, hidden_states.device)
        if past_key_value is not None:
            # Adjust bias for cached keys
            alibi_bias = alibi_bias[:, :, -seq_len:, :]

        attn_weights = attn_weights + alibi_bias

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value
```

### 2. Multi-Query and Grouped-Query Attention

#### Multi-Query Attention (MQA)

MQA significantly reduces KV cache memory requirements:

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Multiple query heads, single key/value head
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)  # Single head
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)  # Single head
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.size()

        # Project queries (multiple heads)
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Project key and value (single head each)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)

        # Expand key and value to match query heads
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        # Handle past key values (much smaller cache!)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # Store only single head in cache
        present_key_value = (k[:, :1], v[:, :1])  # Store only first head

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value
```

#### Grouped-Query Attention (GQA)

GQA balances memory efficiency with model quality:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads // 4)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat key/value heads to match query heads"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Handle past key values
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v)

        # Repeat key/value heads to match query heads
        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value
```

### 3. Efficient Model Architecture Design

```python
class EfficientTransformerConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_scaling=None,
        sliding_window=None,  # For sliding window attention
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window

class EfficientTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)  # or MultiQueryAttention
        self.mlp = EfficientMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        # Pre-norm architecture for better training stability
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_key_value

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class EfficientMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # Swish activation

    def forward(self, x):
        # SwiGLU activation
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

## Multi-GPU Training Strategies

### 1. Modular Design
- **Separate concerns**: Keep data processing, model architecture, training logic, and evaluation separate
- **Configurable components**: Use configuration files for hyperparameters, model architecture, and training settings
- **Reusable modules**: Design components that can be easily swapped or extended

```python
# Example: Modular configuration approach
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    max_position_embeddings: int = 512
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-4
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
```

### 2. Scalable Architecture
- **Parameter-efficient designs**: Follow TinyStories approach for small but effective models
- **Layer-wise scaling**: Design models that can scale from 1M to 100M+ parameters
- **Attention mechanisms**: Implement efficient attention patterns (local, sparse, or sliding window)

## Efficient Model Implementation

### 1. Core Model Components

```python
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn import functional as F

class EfficientTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = EfficientAttention(config)
        self.mlp = EfficientMLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attention_mask=None, past_key_values=None, use_cache=False):
        # Pre-norm architecture for better training stability
        attn_output, present_key_value = self.attention(
            self.ln1(x), 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        
        outputs = (x,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
```

### 2. Efficient Attention Implementation

```python
class EfficientAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use single linear layer for efficiency
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attention_mask=None, past_key_values=None, use_cache=False):
        B, T, C = x.size()
        
        # Compute Q, K, V in one go
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key values for caching
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        present_key_value = (k, v) if use_cache else None
        
        # Use SDPA for efficiency (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Fallback implementation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_weights += attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, present_key_value
```

## Multi-GPU Training Strategies

### 1. Choosing the Right Parallelism Strategy

Based on your setup and model size:

| Setup | Model Fits Single GPU | Strategy |
|-------|----------------------|----------|
| Single Node/Multi-GPU | Yes | DistributedDataParallel (DDP) |
| Single Node/Multi-GPU | No | Pipeline Parallel + ZeRO |
| Multi-Node/Multi-GPU | Any | ZeRO-3 or 3D Parallelism |

### 2. Implementation with Accelerate

```python
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from torch.utils.data import DataLoader

def setup_training():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision='bf16',  # Use bf16 for better stability
        log_with="wandb",
        project_dir="./logs"
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    return accelerator

def train_with_accelerate(model, train_dataloader, optimizer, accelerator):
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            if batch_idx % 100 == 0:
                accelerator.log({"loss": loss.item()}, step=batch_idx)
```

### 3. DeepSpeed Integration

```python
# deepspeed_config.json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-4,
            "warmup_num_steps": 1000
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

## Optimization Techniques

### 1. Mixed Precision Training

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-4,
    
    # Mixed precision settings
    bf16=True,  # Preferred over fp16 for stability
    dataloader_pin_memory=True,
    
    # Optimization settings
    optim="adamw_torch_fused",  # Faster AdamW implementation
    gradient_checkpointing=True,  # Save memory at cost of speed
    
    # Compilation (PyTorch 2.0+)
    torch_compile=True,
    torch_compile_backend="inductor",
    
    # Memory management
    torch_empty_cache_steps=50,
    
    # Data loading optimization
    dataloader_num_workers=4,
    remove_unused_columns=False,
)
```

### 2. Gradient Accumulation and Checkpointing

```python
class OptimizedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Use automatic mixed precision
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        # Scale loss for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        # Backward pass with gradient scaling
        self.accelerator.backward(loss)
        
        return loss.detach()
```

## Memory Management

### 1. Efficient Memory Usage Patterns

```python
def optimize_memory_usage():
    # Clear cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Use gradient checkpointing for large models
    model.gradient_checkpointing_enable()
    
    # Optimize data types
    model = model.to(dtype=torch.bfloat16)  # Use bf16 for better range
    
    # Use memory-efficient optimizers
    from bitsandbytes.optim import AdamW8bit
    optimizer = AdamW8bit(model.parameters(), lr=5e-4)
```

### 2. Dynamic Batch Sizing

```python
class DynamicBatchSampler:
    def __init__(self, dataset, max_tokens=4096, max_batch_size=32):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
    
    def __iter__(self):
        batch = []
        current_tokens = 0
        
        for idx, item in enumerate(self.dataset):
            item_tokens = len(item['input_ids'])
            
            if (current_tokens + item_tokens > self.max_tokens or 
                len(batch) >= self.max_batch_size) and batch:
                yield batch
                batch = []
                current_tokens = 0
            
            batch.append(idx)
            current_tokens += item_tokens
        
        if batch:
            yield batch
```

## Caching and KV Cache Strategies

### 1. Implementing Efficient KV Cache

```python
from transformers import Cache, DynamicCache

class OptimizedGenerationMixin:
    def generate_with_cache(self, input_ids, max_new_tokens=50, use_cache=True):
        if use_cache:
            past_key_values = DynamicCache()
        else:
            past_key_values = None
            
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Only process the last token if using cache
            if use_cache and past_key_values.get_seq_length() > 0:
                input_for_forward = generated_ids[:, -1:]
            else:
                input_for_forward = generated_ids
                
            outputs = self(
                input_ids=input_for_forward,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            
            # Sample next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.multinomial(
                F.softmax(next_token_logits, dim=-1), 
                num_samples=1
            )
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if use_cache:
                past_key_values = outputs.past_key_values
                
        return generated_ids
```

### 2. Memory-Efficient Cache Management

```python
class MemoryEfficientCache(Cache):
    def __init__(self, max_cache_length=1024):
        super().__init__()
        self.max_cache_length = max_cache_length
        self.key_cache = []
        self.value_cache = []
        
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate with existing cache
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
            
            # Trim cache if too long
            if self.key_cache[layer_idx].size(-2) > self.max_cache_length:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, -self.max_cache_length:, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, -self.max_cache_length:, :]
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

## Batch Processing and Data Loading

### 1. Efficient Data Pipeline

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

class EfficientTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, cache_tokenization=True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

        if cache_tokenization:
            self._tokenize_all()

    def _tokenize_all(self):
        """Pre-tokenize all texts for faster training"""
        self.tokenized_texts = []
        for text in self.texts:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
            self.tokenized_texts.append(tokens)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.cache_tokenization:
            return self.tokenized_texts[idx]
        else:
            return self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )

def create_efficient_dataloader(dataset, batch_size=32, num_workers=4):
    """Create optimized DataLoader with proper collation"""

    def collate_fn(batch):
        # Efficient batching with padding
        input_ids = [item['input_ids'].squeeze(0) for item in batch]
        attention_masks = [item['attention_mask'].squeeze(0) for item in batch]

        # Pad to max length in batch (not global max)
        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_masks):
            pad_length = max_len - len(ids)
            padded_input_ids.append(
                torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
            )
            padded_attention_masks.append(
                torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
            )

        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_input_ids)  # For causal LM
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
```

### 2. Smart Batching Strategies

```python
class SmartBatchSampler:
    """Groups samples by length for efficient batching"""

    def __init__(self, dataset, batch_size, max_tokens=None, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.drop_last = drop_last

        # Sort indices by sequence length
        self.length_sorted_indices = self._sort_by_length()

    def _sort_by_length(self):
        lengths = []
        for i, item in enumerate(self.dataset):
            if hasattr(item, 'input_ids'):
                lengths.append((len(item.input_ids), i))
            else:
                # Fallback: estimate length
                lengths.append((len(str(item)), i))

        # Sort by length
        lengths.sort(key=lambda x: x[0])
        return [idx for _, idx in lengths]

    def __iter__(self):
        batch = []
        current_max_len = 0

        for idx in self.length_sorted_indices:
            item_len = len(self.dataset[idx].get('input_ids', []))

            # Check if adding this item would exceed limits
            new_max_len = max(current_max_len, item_len)
            total_tokens = new_max_len * (len(batch) + 1)

            if (len(batch) >= self.batch_size or
                (self.max_tokens and total_tokens > self.max_tokens)) and batch:
                yield batch
                batch = []
                current_max_len = 0

            batch.append(idx)
            current_max_len = max(current_max_len, item_len)

        if batch and not self.drop_last:
            yield batch
```

## Code Organization and Modularity

### 1. Project Structure

```
project/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── configuration.py
│   │   ├── modeling.py
│   │   └── tokenization.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── data_collator.py
│   │   └── callbacks.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── optimization.py
│   └── data/
│       ├── __init__.py
│       ├── dataset.py
│       └── preprocessing.py
├── configs/
│   ├── model_configs/
│   ├── training_configs/
│   └── data_configs/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── tests/
└── requirements.txt
```

### 2. Configuration Management

```python
# configs/base_config.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
import json

@dataclass
class BaseConfig:
    """Base configuration class with serialization support"""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        config_dict = self.to_dict()
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)

    @classmethod
    def from_file(cls, path: str):
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                config_dict = json.load(f)
        return cls(**config_dict)

@dataclass
class ModelConfig(BaseConfig):
    # Architecture
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 512

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Activation
    activation_function: str = "gelu"

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

@dataclass
class TrainingConfig(BaseConfig):
    # Basic training settings
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1

    # Mixed precision and optimization
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    torch_compile: bool = False

    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3

    # Evaluation
    evaluation_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
```

### 3. Modular Training Pipeline

```python
# src/training/trainer.py
from transformers import Trainer, TrainingArguments
from typing import Dict, Optional, Any
import torch
import wandb

class ModularTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        custom_loss_fn=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        self.custom_loss_fn = custom_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with support for different loss functions"""
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(outputs, labels)
        else:
            # Default loss computation
            if labels is not None:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, labels)
                else:
                    loss = outputs.loss
            else:
                loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with custom metrics"""
        # Add custom metrics
        if hasattr(self.model, 'get_memory_usage'):
            logs['memory_usage'] = self.model.get_memory_usage()

        if torch.cuda.is_available():
            logs['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
            logs['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3

        super().log(logs)

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(logs, step=self.state.global_step)

## Performance Monitoring and Debugging

### 1. Performance Profiling

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
from contextlib import contextmanager

class PerformanceMonitor:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_times = []
        self.memory_usage = []

    @contextmanager
    def profile_step(self, step_name="training_step"):
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            with record_function(step_name):
                yield
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            step_time = end_time - start_time
            memory_delta = end_memory - start_memory

            self.step_times.append(step_time)
            self.memory_usage.append(memory_delta)

            if len(self.step_times) % self.log_interval == 0:
                self.log_stats()

    def log_stats(self):
        if self.step_times:
            avg_time = sum(self.step_times[-self.log_interval:]) / min(len(self.step_times), self.log_interval)
            avg_memory = sum(self.memory_usage[-self.log_interval:]) / min(len(self.memory_usage), self.log_interval)

            print(f"Avg step time: {avg_time:.4f}s, Avg memory delta: {avg_memory/1024**2:.2f}MB")

def profile_model_training(model, dataloader, num_steps=10):
    """Profile model training for performance analysis"""

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        model.train()
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            with record_function("forward_pass"):
                outputs = model(**batch)
                loss = outputs.loss

            with record_function("backward_pass"):
                loss.backward()

            with record_function("optimizer_step"):
                # optimizer.step() would go here
                pass

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Save detailed trace
    prof.export_chrome_trace("trace.json")

    return prof
```

### 2. Memory Debugging

```python
class MemoryTracker:
    def __init__(self):
        self.snapshots = []

    def take_snapshot(self, name=""):
        if torch.cuda.is_available():
            snapshot = {
                'name': name,
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved()
            }
            self.snapshots.append(snapshot)
            return snapshot
        return None

    def print_memory_summary(self):
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        print("\n=== Memory Usage Summary ===")
        for i, snapshot in enumerate(self.snapshots):
            print(f"{i}: {snapshot['name']}")
            print(f"  Allocated: {snapshot['allocated']/1024**3:.2f} GB")
            print(f"  Reserved: {snapshot['reserved']/1024**3:.2f} GB")
            print(f"  Max Allocated: {snapshot['max_allocated']/1024**3:.2f} GB")
            print(f"  Max Reserved: {snapshot['max_reserved']/1024**3:.2f} GB")
            print()

    def clear_snapshots(self):
        self.snapshots.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

# Usage example
memory_tracker = MemoryTracker()

# During training
memory_tracker.take_snapshot("before_model_load")
model = load_model()
memory_tracker.take_snapshot("after_model_load")

# ... training code ...

memory_tracker.take_snapshot("after_training")
memory_tracker.print_memory_summary()
```

### 3. Gradient Monitoring

```python
class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.gradient_norms = []
        self.parameter_norms = []

    def compute_gradient_norm(self):
        total_norm = 0.0
        param_count = 0

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        return total_norm

    def compute_parameter_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = total_norm ** (1. / 2)
        self.parameter_norms.append(total_norm)
        return total_norm

    def check_gradient_health(self, threshold=10.0):
        """Check for gradient explosion or vanishing"""
        if self.gradient_norms:
            recent_norm = self.gradient_norms[-1]
            if recent_norm > threshold:
                print(f"Warning: Large gradient norm detected: {recent_norm:.4f}")
            elif recent_norm < 1e-7:
                print(f"Warning: Very small gradient norm detected: {recent_norm:.4e}")
```

## Production Deployment

### 1. Model Optimization for Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.jit import script, trace
import onnx
import onnxruntime as ort

class ModelOptimizer:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def optimize_for_inference(self):
        """Apply various optimizations for inference"""
        # Set to evaluation mode
        self.model.eval()

        # Enable optimizations
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True

        # Compile with torch.compile (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Use SDPA for attention
        if hasattr(self.model.config, 'attn_implementation'):
            self.model.config.attn_implementation = "sdpa"

        return self.model

    def export_to_onnx(self, output_path, sample_input_ids=None):
        """Export model to ONNX format"""
        if sample_input_ids is None:
            sample_input_ids = torch.randint(0, 1000, (1, 10))

        # Export to ONNX
        torch.onnx.export(
            self.model,
            sample_input_ids,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )

        return output_path

    def quantize_model(self, quantization_type="dynamic"):
        """Apply quantization for smaller model size"""
        if quantization_type == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        elif quantization_type == "static":
            # Static quantization requires calibration data
            # This is a simplified example
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            # ... calibration step would go here ...
            quantized_model = torch.quantization.convert(self.model, inplace=False)

        return quantized_model
```

### 2. Serving Infrastructure

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import asyncio
from typing import List, Optional
import uvicorn

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    generation_time: float

class ModelServer:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

        # Optimize for inference
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        generation_time = time.time() - start_time

        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            generation_time=generation_time
        )

# FastAPI app
app = FastAPI(title="Transformer Model Server")
model_server = None

@app.on_event("startup")
async def startup_event():
    global model_server
    model_server = ModelServer("path/to/your/model")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        response = await model_server.generate(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_server is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Best Practices Summary

### 1. Inference Optimization Priority
- **Static KV Cache + torch.compile**: Achieve up to 4x speedup for production inference
- **FlashAttention-2**: Use for memory-efficient attention computation
- **Quantization**: Apply 8-bit or 4-bit quantization for memory-constrained environments
- **Speculative decoding**: Use assistant models for faster token generation
- **Batch optimization**: Group similar-length sequences for efficient processing

### 2. Architecture Design for Inference
- **Positional embeddings**: Use RoPE or ALiBi for better long-sequence handling
- **Attention mechanisms**: Implement GQA or MQA to reduce KV cache memory
- **Model architecture**: Design with inference efficiency in mind from the start
- **Cache strategies**: Choose appropriate cache type based on use case requirements

### 3. Memory Management
- **Dynamic memory allocation**: Adjust batch sizes based on available memory
- **Cache offloading**: Use CPU offloading for memory-constrained scenarios
- **Quantized caches**: Apply cache quantization for long-context generation
- **Memory monitoring**: Track and optimize memory usage throughout inference

### 4. Production Deployment
- **Model optimization**: Combine quantization, compilation, and efficient attention
- **Serving infrastructure**: Implement streaming, batching, and async processing
- **Monitoring**: Track latency, throughput, memory usage, and quality metrics
- **Scaling**: Design for horizontal scaling with load balancing

### 5. Development Workflow
- **Start with inference**: Optimize inference first, then scale to training
- **Profile continuously**: Use profiling tools to identify bottlenecks
- **Modular design**: Keep inference components separate and configurable
- **Quality assurance**: Maintain model quality while optimizing performance

## Conclusion

This guide provides a comprehensive framework for developing efficient transformer models with a primary focus on inference optimization. The key principles are:

1. **Inference-First Design**: Prioritize inference efficiency in architecture and implementation decisions
2. **Memory Efficiency**: Use advanced caching strategies and quantization techniques to minimize memory usage
3. **Performance Optimization**: Leverage modern techniques like static caching, FlashAttention, and torch.compile
4. **Scalable Architecture**: Design models that can efficiently handle long sequences and multi-turn conversations
5. **Production-Ready**: Build with deployment, monitoring, and scaling requirements in mind

Modern transformer inference requires sophisticated optimization techniques to achieve production-ready performance. By combining static KV caching, efficient attention mechanisms, quantization, and advanced generation strategies, you can build transformer models that deliver both high quality and excellent performance in real-world applications.

The landscape of transformer optimization continues to evolve rapidly, with new techniques like speculative decoding, advanced quantization methods, and architectural innovations constantly emerging. Stay current with the latest developments in the Hugging Face ecosystem and the broader research community to maintain optimal performance.

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
```
```
