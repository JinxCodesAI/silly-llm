# Installation Guide

This guide helps you install the right dependencies for your use case.

## Quick Start

Choose the installation option that matches your intended use:

### 1. Testing Only (No External Dependencies)
```bash
pip install -r requirements-minimal.txt
python -m training.synthetic_data_generation.main --mock-provider --num-stories 5
```

### 2. API-Based Generation (OpenAI, Together AI, etc.)
```bash
pip install -r requirements-api.txt
export AI_API_KEY=your_api_key
python -m training.synthetic_data_generation.main --openai-provider --model-name gpt-3.5-turbo --num-stories 10
```

### 3. Local Model Generation (HuggingFace)
```bash
pip install -r requirements-transformers.txt
python -m training.synthetic_data_generation.main --model-name Qwen/Qwen2.5-3B-Instruct --num-stories 10
```

### 4. All Features
```bash
pip install -r requirements.txt
# Now you can use any provider
```

## Requirements Files Explained

| File | Use Case | Dependencies | Size |
|------|----------|--------------|------|
| `requirements-minimal.txt` | Testing with MockProvider | pydantic only | ~10MB |
| `requirements-api.txt` | API-based generation | pydantic + httpx | ~20MB |
| `requirements-transformers.txt` | Local models | pydantic + torch + transformers | ~2-5GB |
| `requirements.txt` | All features | All of the above | ~2-5GB |
| `requirements-dev.txt` | Development | All + testing tools | ~2-5GB |

## Platform-Specific Notes

### GPU Support (for TransformersProvider)
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install the rest
pip install -r requirements-transformers.txt
```

### Apple Silicon (M1/M2)
```bash
# PyTorch with Metal Performance Shaders
pip install torch torchvision torchaudio

# Then install the rest
pip install -r requirements-transformers.txt
```

### CPU Only
```bash
# CPU-only PyTorch (smaller download)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install the rest
pip install -r requirements-transformers.txt
```

## Environment Variables

### For API Providers
```bash
# Required for OpenAI-compatible providers
export AI_API_KEY=your_api_key_here

# Optional: for custom API endpoints
export API_BASE_URL=https://api.your-service.com/v1
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use smaller models or reduce batch size
2. **Slow downloads**: PyTorch and transformers are large packages
3. **Import errors**: Make sure you installed the right requirements file

### Verification

Test your installation:
```bash
# Test minimal installation
python -c "from training.common.data_models import GenerationConfig; print('✅ Core dependencies OK')"

# Test API provider
python -c "from training.common.llm_providers import OpenAICompatibleProvider; print('✅ API provider OK')"

# Test transformers provider
python -c "from training.common.llm_providers import TransformersProvider; print('✅ Transformers provider OK')"
```

## Development Setup

For contributors:
```bash
# Install all dependencies including dev tools
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy training/
```
