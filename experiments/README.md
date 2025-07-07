# Experiments

This directory contains various experiments and research projects related to transformer models, inference optimization, and performance analysis.

## Available Experiments

### 🚀 Inference Methods Comparison

**Location**: `Inference_Methods_Comparison/`

A comprehensive comparison of different transformer inference methods including:
- Standard autoregressive generation
- Speculative decoding (Qwen3-0.6B + Qwen3-4B)
- Beam search with multiple sequences
- Batched generation strategies
- Nucleus sampling optimization

**Key Features**:
- Interactive configuration system
- Performance metrics (speed, memory, tokens/sec)
- Multiple execution modes (quick, interactive, config-based)
- Support for quantization and FlashAttention-2
- Configurable prompts and completion counts

**Quick Start**:
```bash
cd Inference_Methods_Comparison
./run.sh launcher
```

See `Inference_Methods_Comparison/README.md` for detailed documentation.

## Experiment Structure

Each experiment should follow this structure:
```
experiment_name/
├── README.md              # Detailed documentation
├── requirements.txt       # Dependencies
├── main_script.py        # Primary experiment script
├── config_*.json         # Configuration files
├── test_setup.py         # Setup verification
├── run.sh               # Shell launcher (optional)
└── results/             # Output directory (gitignored)
```

## Adding New Experiments

1. Create a new subdirectory with a descriptive name
2. Include comprehensive README.md with:
   - Purpose and goals
   - Setup instructions
   - Usage examples
   - Expected outputs
3. Provide example configurations
4. Include setup verification script
5. Update this main README.md

## General Requirements

Most experiments require:
- Python 3.8+
- PyTorch 2.0+
- Transformers library
- CUDA-capable GPU (recommended)

Specific requirements are listed in each experiment's `requirements.txt`.

## Contributing

When adding experiments:
- Follow the established structure
- Include comprehensive documentation
- Provide working examples
- Test on different hardware configurations
- Consider memory and compute requirements

## Future Experiments

Planned experiments include:
- Training efficiency comparisons
- Memory optimization techniques
- Multi-GPU scaling analysis
- Model architecture comparisons
- Quantization impact studies
