# Inference Methods Comparison

This directory contains scripts to compare different transformer inference methods for efficiency and performance.

## Features

The comparison script tests the following inference methods:

1. **Standard Generation**: Basic autoregressive generation
2. **Beam Search**: Multiple sequences with `num_return_sequences`
3. **Batched Generation**: Repeating the same prompt in a batch
4. **Nucleus Sampling**: Top-p sampling with optimized parameters

**Speculative Decoding**: Can be enabled as a flag (`use_speculative_decoding`) that works with ALL methods above, using Qwen3-0.6B as assistant model with Qwen3-4B as main model.

## Performance Metrics

For each method, the script measures:
- Generation time (seconds)
- Memory usage (GB)
- Tokens per second
- Number of tokens generated
- Total completions

## Usage

### Quick Start

pip install -r requirements.txt  

python inference_comparison.py --quick

# Test specific methods only
python inference_comparison.py --methods standard batched

# Interactive configuration
python inference_comparison.py --interactive

# Use configuration file
python inference_comparison.py --config config_example.json

# Simple test (1 method, 1 completion)
python inference_comparison.py --config config_test.json

# Save results to specific file
python inference_comparison.py --output my_results.json
```

### Configuration Options

The script supports various configuration options:

- **Models**: Main model (Qwen3-4B) and assistant model (Qwen3-0.6B) for speculative decoding
- **Generation**: Max tokens, temperature, top-p, top-k parameters
- **Experiments**: Number of completions per prompt, batch size, custom prompts
- **Performance**: Quantization, FlashAttention-2, speculative decoding flag, cache implementation

### Example Configuration

See `config_example.json` for a complete configuration example.

### Sample Prompts

Default prompts are designed to test story generation:
- "generate short (up to 150 words) bed time story containing word elephant and cake"
- "generate short (up to 150 words) bed time story containing word dog and sausage"

You can customize prompts through the interactive mode or configuration file.

## 📖 Detailed Documentation

For comprehensive information about configuration options, extending the scripts, and troubleshooting, see **[DETAILED_GUIDE.md](DETAILED_GUIDE.md)**.

## Requirements

- PyTorch with CUDA support
- Transformers library
- Access to Qwen models (Qwen3-4B and Qwen3-0.6B)
- Sufficient GPU memory (recommended: 16GB+ for full models, 8GB+ with quantization)

## Output

The script generates:
1. **Console output**: Real-time progress and summary statistics
2. **JSON results file**: Detailed results for further analysis
3. **Performance rankings**: Speed, memory efficiency, and timing comparisons

## Notes

- Always uses non-thinking mode (`enable_thinking=False`) as requested
- Supports various cache implementations (dynamic, static, offloaded, quantized)
- Includes memory management and cleanup
- Handles GPU memory constraints with quantization options

## Quick Start

### Option 1: Simple Launcher (Recommended)
```bash
cd experiments/Inference_Methods_Comparison
./run.sh launcher
# or
python run_experiments.py
```

### Option 2: Direct Commands
```bash
# Test setup first
./run.sh test

# Quick experiment
./run.sh quick

# Interactive configuration
./run.sh interactive
```

### Option 3: Configuration Files
```bash
# Use example configuration
python inference_comparison.py --config config_example.json

# Quick test configuration
python inference_comparison.py --config config_quick.json

# Comprehensive test
python inference_comparison.py --config config_comprehensive.json
```

## Configuration Files

Three example configurations are provided:

1. **config_test.json**: Minimal test with 1 method, 1 completion, 30 tokens (fastest)
2. **config_quick.json**: Fast test with 2 methods, 2 completions, 50 tokens, speculative decoding enabled
3. **config_example.json**: Standard test with 3 completions, 150 tokens, default settings
4. **config_comprehensive.json**: Thorough test with 5 completions, 150 tokens, 5 prompts, speculative decoding enabled

## Example Output

```
🏆 Performance Rankings:
1. Fastest (tokens/sec): standard_speculative (45.2 tok/s)
2. Most memory efficient: batched (2.1 GB)
3. Fastest total time: beam_search_speculative (1.8 s)

📝 Sample Outputs:
STANDARD:
Prompt: generate short (up to 150 words) bed time story containing word elephant and cake
Output: Once upon a time, there was a little elephant named Ellie who loved to bake...
Time: 3.2s, Tokens/s: 28.1

STANDARD_SPECULATIVE:
Prompt: generate short (up to 150 words) bed time story containing word elephant and cake
Output: In a magical forest, an elephant discovered a mysterious cake that granted wishes...
Time: 2.1s, Tokens/s: 45.2
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Enable quantization in config or use smaller batch sizes
2. **Model Download**: First run will download models (several GB)
3. **FlashAttention**: Install with `pip install flash-attn` for better performance
4. **Dependencies**: Run `python test_setup.py` to check all requirements

### Performance Tips

- Use quantization for memory-constrained systems
- Enable FlashAttention-2 for better performance
- Use static cache with torch.compile for production
- Batch similar-length prompts together
