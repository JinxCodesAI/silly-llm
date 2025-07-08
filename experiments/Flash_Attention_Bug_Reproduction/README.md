# Flash Attention Bug Reproduction Script

This is a minimal, self-contained script to reproduce potential issues with `flash_attention_2` implementation when using the Qwen2.5-0.5B model.

## Purpose

The script helps identify and reproduce bugs in flash attention by:
1. Running inference with standard attention implementation
2. Running the same inference with `flash_attention_2`
3. Comparing results and identifying any errors or performance issues

## Files

- `flash_attention_bug_reproduction.py` - Main reproduction script (v1)
- `flash_bug_reproduction_v2.py` - Enhanced script based on Qwen3 test patterns (v2)
- `requirements.txt` - Minimal dependencies
- `README.md` - This documentation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install torch>=2.0.0 transformers>=4.40.0 flash-attn>=2.5.0 accelerate>=0.20.0
```

## Usage

### Version 1 (Original):
```bash
python flash_attention_bug_reproduction.py
```

### Version 2 (Based on Qwen3 Tests):
```bash
python flash_bug_reproduction_v2.py
```

**Recommended**: Use v2 as it follows the exact same patterns as the official Qwen3 test suite.

## Differences between v1 and v2

### Version 1 (flash_attention_bug_reproduction.py):
- Uses `AutoModelForCausalLM` with any Qwen model
- Uses `Qwen/Qwen2.5-0.5B` by default
- Simple comparison between standard and flash attention
- Uses chat templates and sampling generation

### Version 2 (flash_bug_reproduction_v2.py):
- Uses `Qwen3ForCausalLM` specifically (matches official tests)
- Uses `Qwen/Qwen3-0.6B-Base` (same as official test suite)
- Tests multiple attention implementations: eager, sdpa, flash_attention_2
- Uses greedy generation (temperature=0) like official tests
- Includes logits testing and numerical comparison with `torch.testing.assert_close`
- Follows exact patterns from `transformers/tests/models/qwen3/test_modeling_qwen3.py`

## What the scripts do

1. **Tests without Flash Attention**: Runs inference using standard attention
2. **Tests with Flash Attention**: Runs the same inference with `attn_implementation="flash_attention_2"`
3. **Compares results**: Shows performance differences and identifies any errors
4. **Reports bugs**: If flash attention fails while standard attention works, it provides detailed error information for bug reports

## Expected Output

### If everything works correctly:
```
✅ Both implementations work correctly
🚀 Flash Attention provides performance improvement
```

### If there's a flash attention bug:
```
❌ Flash Attention implementation has issues!
🐛 This indicates a potential bug in flash_attention_2

Error details for bug report:
Model: Qwen/Qwen2.5-0.5B
Error: [specific error message]
Full traceback: [detailed traceback]
```

## Configuration

You can modify the script to test different scenarios:

1. **Change model**: Edit the `model_name` variable in `main()`
2. **Modify prompt**: Change the hardcoded prompt in `test_inference()`
3. **Adjust generation parameters**: Modify `generation_kwargs` in `test_inference()`

## For Bug Reports

If the script identifies a flash attention bug, include this information in your GitHub issue:

1. **Environment details** (automatically printed by the script):
   - Model name
   - PyTorch version
   - Transformers version
   - Flash Attention version
   - CUDA device info

2. **Error details**:
   - Full error message
   - Complete traceback
   - Confirmation that standard attention works

3. **Reproduction steps**:
   - Copy of this script
   - Command used to run it
   - Any modifications made

## Troubleshooting

### Flash Attention not installed
```
ImportError: No module named 'flash_attn'
```
Solution: `pip install flash-attn`

### CUDA issues
```
RuntimeError: CUDA out of memory
```
Solution: The script uses a small model (0.5B parameters) to minimize memory usage. If you still get OOM errors, try:
- Reducing `max_new_tokens`
- Using CPU instead: modify `device_map="cpu"`

### Model not found
```
OSError: Qwen/Qwen2.5-0.5B does not appear to be a model identifier
```
Solution: The model will be downloaded automatically on first run. Ensure you have internet connection.

## Technical Details

- **Model**: Qwen2.5-0.5B (small model for minimal resource usage)
- **Precision**: bfloat16 (standard for modern models)
- **Device**: Auto-detected (CUDA if available, otherwise CPU)
- **Generation**: 100 tokens with temperature=0.7, top_p=0.9
- **Memory management**: Automatic CUDA cache clearing between tests
