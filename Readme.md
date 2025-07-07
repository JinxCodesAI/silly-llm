# Silly LLM

Silly LLM is a modular, extensible framework that builds on the work outlined in **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?** paper.

## Project Vision

This project implements a highly modular and flexible pipeline for creating small language models through synthetic data generation, pretraining, and instruction tuning. The architecture emphasizes:

- **Modularity**: Clear separation of concerns with pluggable components
- **Flexibility**: Easy swapping between different models, APIs, and generation strategies
- **Maintainability**: Clean interfaces, comprehensive configuration management, and extensible design patterns
- **Scalability**: Support for distributed processing and efficient resource utilization

## Pipeline Overview

The framework consists of five main phases, each implemented as a collection of modular, interchangeable components:

1. **Phase 1: Synthetic Dataset Creation** - Modular data generation with pluggable LLM backends
2. **Phase 2: Small Language Model Pretraining** - Flexible training pipeline with configurable architectures
3. **Phase 3: Instruction-Tuning Dataset Creation** - Extensible instruction generation framework
4. **Phase 4: Model Instruction Tuning** - Configurable fine-tuning with multiple strategies
5. **Phase 5: Evaluation and Comparison** - Pluggable evaluation metrics and comparison frameworks

## Architecture Principles

### 1. Plugin-Based Architecture
All major components implement well-defined interfaces, allowing easy extension and modification without touching core code.

### 2. Configuration-Driven Design
Centralized configuration management using structured config files (YAML/JSON) with environment-specific overrides.

### 3. Provider Pattern
Abstract base classes for different types of providers (LLM providers, data providers, evaluation providers) enable seamless switching between implementations.

### 4. Factory Pattern
Component factories handle the instantiation and configuration of different implementations based on configuration.

### 5. Dependency Injection
Clear dependency management through dependency injection containers, making testing and component swapping straightforward.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Generate vocabulary
python -m silly_llm.data.vocabulary --config configs/vocabulary.yaml

# Generate synthetic dataset
python -m silly_llm.data.generation --config configs/generation.yaml

# Train model
python -m silly_llm.training.pretrain --config configs/training.yaml
```

## Documentation Structure

- [`docs/Plan.md`](docs/Plan.md) - Enhanced modular implementation plan
- [`docs/spec/dataset_creation_pipeline.md`](docs/spec/dataset_creation_pipeline.md) - Detailed Phase 1 architecture specification
- [`docs/TinyStories.md`](docs/TinyStories.md) - Original research paper content
- [`docs/architecture/`](docs/architecture/) - Detailed architecture documentation (to be created)
