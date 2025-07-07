# Detailed Guide: Inference Methods Comparison

This comprehensive guide explains how to use, configure, and extend the inference comparison scripts.

## 📁 File Structure and Roles

### Core Scripts

| File | Role | Description |
|------|------|-------------|
| `inference_comparison.py` | Main experiment script | Contains all inference methods, configuration handling, and result analysis |
| `run_experiments.py` | Interactive launcher | Menu-driven interface for easy experiment execution |
| `test_setup.py` | Setup verification | Checks dependencies, CUDA, and model accessibility |
| `run.sh` | Shell launcher | Simple command-line interface for quick access |

### Configuration Files

| File | Purpose | Use Case |
|------|---------|----------|
| `config_example.json` | Standard configuration | Default setup with your specified prompts (elephant/cake, dog/sausage) |
| `config_quick.json` | Fast testing | Minimal settings for quick validation (2 completions, 50 tokens, quantization enabled) |
| `config_comprehensive.json` | Thorough testing | Extensive testing with 5 prompts, 5 completions, speculative decoding enabled |

### Documentation

| File | Content |
|------|---------|
| `README.md` | Quick start guide and basic usage |
| `DETAILED_GUIDE.md` | This comprehensive guide |
| `requirements.txt` | Python dependencies |

## 🔧 Configuration System

### Core Configuration Parameters

#### Model Settings
```json
{
  "main_model": "Qwen/Qwen3-4B",           // Primary model for generation
  "assistant_model": "Qwen/Qwen3-0.6B"    // Assistant model for speculative decoding
}
```

#### Generation Settings
```json
{
  "max_new_tokens": 150,    // Maximum tokens to generate
  "temperature": 0.7,       // Sampling temperature (0.0 = deterministic, 1.0 = random)
  "top_p": 0.9,            // Nucleus sampling threshold
  "top_k": 50              // Top-k sampling limit
}
```

#### Experiment Settings
```json
{
  "num_completions": 3,     // Number of generations per prompt (int or array)
  "batch_size": 4,          // GPU capacity - how many prompts to process simultaneously
  "methods_to_test": null,  // List of methods to test (null = all methods)
  "prompts": [...]          // List of prompts to test
}
```

#### Performance Settings
```json
{
  "use_quantization": false,        // Enable 4-bit quantization for memory efficiency
  "use_flash_attention": true,      // Enable FlashAttention-2 for speed
  "use_speculative_decoding": false, // Enable speculative decoding for all methods
  "cache_implementation": "dynamic"  // Cache strategy (see below)
}
```

### Understanding Prompts vs Completions vs Methods vs Batch Size

#### Relationship Between Parameters

1. **`prompts`**: List of different prompts to test
   - Example: `["story about elephant", "story about dog"]` = 2 prompts

2. **`num_completions`**: Number of generations per prompt
   - **As integer**: Same number for all prompts (e.g., `3` = 3 completions per prompt)
   - **As array**: Specific per prompt (e.g., `[3, 2, 5]` = 3 for first, 2 for second, 5 for third)
   - If array, length must equal number of prompts

3. **`batch_size`**: GPU capacity - how many prompts to process simultaneously
   - **Independent of `num_completions`**
   - Determines batching strategy across all methods
   - Example: `batch_size=4` means process 4 prompts at once

4. **`methods_to_test`**: Which inference methods to run
   - `null` or not specified: Run all 4 methods (standard, beam_search, batched, nucleus_sampling)
   - `["standard", "batched"]`: Run only these 2 methods
   - Total results = methods × total completions across all prompts

#### Method Behavior

- **Standard**: Batched generation with `batch_size=1` (one at a time)
- **Batched**: Uses configured `batch_size` to process multiple prompts simultaneously
- **Beam Search**: Processes prompts one by one, respects `batch_size` for sequences per prompt
- **Nucleus Sampling**: Same as batched but with modified sampling parameters

#### Example Scenarios

**Scenario 1: Equal Completions Per Prompt**
```json
{
  "prompts": [
    "generate short bed time story containing word elephant and cake",
    "generate short bed time story containing word dog and sausage"
  ],
  "num_completions": 3,  // 3 completions for each prompt
  "batch_size": 4        // Process 4 prompts at once
}
```
Result: 3 elephant stories + 3 dog stories = 6 total completions
Batching: Batch 1: [elephant, elephant, elephant, dog], Batch 2: [dog, dog]

**Scenario 2: Different Completion Counts (Your Example)**
```json
{
  "prompts": ["prompt1", "prompt2", "prompt3"],
  "num_completions": [3, 2, 3],  // Array: 3 for prompt1, 2 for prompt2, 3 for prompt3
  "batch_size": 4
}
```
Result: 3 + 2 + 3 = 8 total completions
Batching:
- Batch 1: [prompt1, prompt1, prompt1, prompt2] (4 items)
- Batch 2: [prompt2, prompt3, prompt3, prompt3] (4 items)

**Scenario 3: Single Prompt, Multiple Completions**
```json
{
  "num_completions": 8,
  "batch_size": 4,
  "prompts": ["single prompt"]
}
```
Result: 8 completions of the same prompt
Batching: Batch 1: [prompt, prompt, prompt, prompt], Batch 2: [prompt, prompt, prompt, prompt]

### Cache Implementation Options

| Value | Description | Memory Usage | Speed | torch.compile Support |
|-------|-------------|--------------|-------|----------------------|
| `"dynamic"` | Default cache that grows as needed | Medium | Medium | No |
| `"static"` | Pre-allocated cache for production | Low | High | Yes |
| `"offloaded"` | CPU offloading for memory constraints | Very Low | Low | No |
| `"quantized"` | Quantized cache for long contexts | Low | Medium | No |

**Recommendations:**
- **Development/Testing**: `"dynamic"`
- **Production/Speed**: `"static"`
- **Memory Constrained**: `"offloaded"` or `"quantized"`

## 🚀 Inference Methods Explained

### 1. Standard Generation
- **Description**: Basic autoregressive generation
- **Use Case**: Baseline comparison
- **Parameters**: Uses all generation settings (temperature, top_p, etc.)

### 2. Beam Search
- **Description**: Explores multiple paths simultaneously
- **Use Case**: Higher quality outputs, deterministic results
- **Parameters**: `num_completions` determines number of beams and return sequences
- **Note**: Overrides sampling parameters (sets `do_sample=False`)

### 3. Batched Generation
- **Description**: Processes multiple identical prompts simultaneously
- **Use Case**: Throughput optimization for same prompt
- **Parameters**: `batch_size` controls batch size, `num_completions` total outputs
- **Efficiency**: Better GPU utilization

### 4. Nucleus Sampling
- **Description**: Optimized top-p sampling
- **Use Case**: Balanced quality and diversity
- **Parameters**: Fixed top_p=0.9, temperature=0.8 (overrides config)

### Speculative Decoding (Cross-Method Feature)

**What it is**: Uses a smaller "assistant" model to predict tokens, which are then verified by the main model.

**How to enable**: Set `"use_speculative_decoding": true` in configuration

**Compatibility**: Works with ALL methods:
- `standard` → `standard_speculative`
- `beam_search` → `beam_search_speculative`
- `batched` → `batched_speculative`
- `nucleus_sampling` → `nucleus_sampling_speculative`

**Benefits**:
- Faster generation (especially for longer sequences)
- Same quality as main model
- Automatic fallback if assistant model unavailable

## 📊 Creating Custom Experiments

### Example 1: Compare Speculative Decoding Impact
```json
{
  "main_model": "Qwen/Qwen3-4B",
  "assistant_model": "Qwen/Qwen3-0.6B",
  "num_completions": 5,
  "prompts": ["Write a technical explanation of machine learning"],
  "use_speculative_decoding": true,
  "max_new_tokens": 200
}
```

### Example 2: Memory-Constrained Setup
```json
{
  "use_quantization": true,
  "cache_implementation": "offloaded",
  "use_flash_attention": true,
  "max_new_tokens": 100,
  "batch_size": 2
}
```

### Example 3: Production Speed Test
```json
{
  "cache_implementation": "static",
  "use_flash_attention": true,
  "use_speculative_decoding": true,
  "num_completions": 10,
  "batch_size": 8
}
```

## 🔬 Extending the Scripts

### Adding New Inference Methods

1. **Create method function** in `InferenceMethodsComparator` class:
```python
def method_your_new_method(self, prompt: str, completion_idx: int) -> InferenceResult:
    # Your implementation
    pass
```

2. **Add to experiment runner**:
```python
elif method == "your_new_method":
    # Handle your method
    pass
```

3. **Update methods list**:
```python
methods = ["standard", "beam_search", "batched", "nucleus_sampling", "your_new_method"]
```

### Adding New Configuration Options

1. **Update `InferenceConfig` dataclass**:
```python
@dataclass
class InferenceConfig:
    # ... existing fields ...
    your_new_option: bool = False
```

2. **Use in generation methods**:
```python
if self.config.your_new_option:
    generation_kwargs["your_parameter"] = value
```

3. **Add to interactive config**:
```python
config.your_new_option = input("Enable your feature? [y/N]: ").strip().lower() in ['y', 'yes']
```

### Adding New Metrics

1. **Extend `InferenceResult` dataclass**:
```python
@dataclass
class InferenceResult:
    # ... existing fields ...
    your_new_metric: float = 0.0
```

2. **Calculate in methods**:
```python
your_metric_value = calculate_your_metric(outputs)
return InferenceResult(
    # ... existing fields ...
    your_new_metric=your_metric_value
)
```

3. **Update analysis**:
```python
def analyze_results(self, results):
    # Add your metric to analysis
    your_metrics = [r.your_new_metric for r in method_results]
    analysis[method]["avg_your_metric"] = sum(your_metrics) / len(your_metrics)
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   - Enable quantization: `"use_quantization": true`
   - Use offloaded cache: `"cache_implementation": "offloaded"`
   - Reduce batch size: `"batch_size": 1`
   - Reduce max tokens: `"max_new_tokens": 50`

2. **Model Download Issues**
   - Ensure internet connection
   - Check Hugging Face access
   - Verify model names are correct

3. **FlashAttention Errors**
   - Install: `pip install flash-attn`
   - Disable if issues: `"use_flash_attention": false`

4. **Slow Performance**
   - Enable static cache: `"cache_implementation": "static"`
   - Enable FlashAttention: `"use_flash_attention": true`
   - Try speculative decoding: `"use_speculative_decoding": true`

### Performance Optimization Tips

1. **For Speed**: static cache + FlashAttention + speculative decoding
2. **For Memory**: quantization + offloaded cache + smaller batch sizes
3. **For Quality**: beam search + higher max_new_tokens
4. **For Throughput**: batched generation + larger batch sizes

## 📊 Memory Measurement

The script measures **peak GPU memory usage** during generation using `torch.cuda.max_memory_allocated()`. This provides more accurate memory usage measurements compared to simple before/after comparisons.

**Key points:**
- Memory is reset before each method using `torch.cuda.reset_peak_memory_stats()`
- Measurements reflect the maximum memory used during the generation process
- Values are reported in GB for easy comparison
- Negative values indicate measurement errors (now fixed)

This guide should help you understand and effectively use the inference comparison system!
