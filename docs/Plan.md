# Plan: Recreating the TinyStories Methodology

This document outlines the step-by-step plan to replicate the ideas presented in the "TinyStories" paper. The project is divided into five main phases:

1.  **Phase 1: Synthetic Dataset Creation (`TinyStories`)**
2.  **Phase 2: Small Language Model Pretraining**
3.  **Phase 3: Instruction-Tuning Dataset Creation (`TinyStories-Instruct`)**
4.  **Phase 4: Model Instruction Tuning**
5.  **Phase 5: Evaluation and Comparison (`GPT-Eval`)**

---

## Phase 1: Modular Synthetic Dataset Creation

**Objective:** Generate a large dataset of simple, coherent short stories using a flexible, extensible architecture that supports multiple LLM backends and generation strategies.

### Core Components Architecture

#### 1. LLM Provider Interface
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities and limitations."""
        pass
```

#### 2. LLM Provider Implementations

**API-Based Provider (OpenAI-Compatible)**
```python
class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
```

**Transformers-Based Provider**
```python
class TransformersProvider(LLMProvider):
    """Provider for local transformers models."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        # Implementation using transformers generate method
        # Supports chat templates, thinking modes, etc.
        pass
```

#### 3. Vocabulary Management System
```python
class VocabularyManager:
    """Manages vocabulary generation and validation."""

    def __init__(self, provider: LLMProvider, config: VocabularyConfig):
        self.provider = provider
        self.config = config
        self.validator = VocabularyValidator(config.validation_rules)

    async def generate_vocabulary(self) -> Vocabulary:
        """Generate vocabulary using configured provider."""
        pass

    def validate_vocabulary(self, vocab: Vocabulary) -> ValidationResult:
        """Validate vocabulary against age-appropriateness rules."""
        pass
```

#### 4. Story Generation Pipeline
```python
class StoryGenerationPipeline:
    """Orchestrates the story generation process."""

    def __init__(self,
                 llm_provider: LLMProvider,
                 vocabulary: Vocabulary,
                 prompt_generator: PromptGenerator,
                 storage_provider: StorageProvider):
        self.llm_provider = llm_provider
        self.vocabulary = vocabulary
        self.prompt_generator = prompt_generator
        self.storage_provider = storage_provider

    async def generate_stories(self, count: int) -> List[Story]:
        """Generate specified number of stories."""
        pass
```

### Configuration Management

#### Central Configuration Schema
```yaml
# configs/generation.yaml
llm_provider:
  type: "transformers"  # or "openai_compatible"
  config:
    model_name: "Qwen/Qwen3-4B"
    device: "auto"
    generation_params:
      max_new_tokens: 512
      temperature: 0.8
      top_p: 0.9

vocabulary:
  target_word_count: 1500
  categories:
    nouns: 750
    verbs: 375
    adjectives: 375
  validation:
    age_range: [3, 4]
    complexity_threshold: 0.7

generation:
  batch_size: 32
  total_stories: 10000
  features:
    - dialogue
    - plot_twist
    - moral_value
    - bad_ending

storage:
  type: "local"  # or "s3", "gcs"
  config:
    output_dir: "./generated_data"
    format: "jsonl"
```

### Modular Implementation Steps

1.  **Core Framework Setup:**
    *   Implement abstract base classes for all providers
    *   Create dependency injection container
    *   Set up configuration management system

2.  **LLM Provider Implementation:**
    *   Implement OpenAI-compatible provider
    *   Implement transformers-based provider (supporting Qwen and other models)
    *   Create provider factory for dynamic instantiation

3.  **Vocabulary Generation Module:**
    *   Implement vocabulary manager with pluggable validation
    *   Create age-appropriateness validators
    *   Support incremental vocabulary building

4.  **Story Generation Pipeline:**
    *   Implement modular prompt generation system
    *   Create story generation orchestrator
    *   Add support for different generation strategies (batch, streaming, branching)

---

## Phase 2: Small Language Model Pretraining

**Objective:** Train a small, decoder-only transformer model from scratch on the `TinyStories` dataset.

**Steps:**

1.  **Architecture and Tokenization:**
    *   **Action:** Define the model architecture and train a tokenizer.
    *   **Details:**
        *   **Model:** Use a GPT-Neo architecture. The configuration (e.g., 8 layers, 8 heads, 512 embedding dimension) should be easily adjustable to experiment with different model sizes (e.g., 1M, 3M, 8M, 28M parameters).
        *   **Tokenizer:** Train a Byte-Pair Encoding (BPE) tokenizer on the `tinystories_dataset.txt`. Limit the vocabulary size (e.g., to the top 10,000 tokens) as described in the paper. Save it for later use.

2.  **Preprocessing Script:**
    *   **Action:** Create a script to prepare the data for training.
    *   **Details:** This script will tokenize the entire `tinystories_dataset.txt` and chunk it into fixed-size sequences (e.g., 512 tokens) for the model.

3.  **Training Script (`pretrain.py`):**
    *   **Action:** Develop the main script for pretraining the language model.
    *   **Details:**
        *   Use PyTorch and the Hugging Face `transformers` and `accelerate` libraries.
        *   The script will handle:
            *   Loading the preprocessed dataset.
            *   Initializing the GPT-Neo model with the specified configuration.
            *   Implementing a standard training loop with a causal language modeling objective (cross-entropy loss).
            *   Configuring hyperparameters (batch size, learning rate, optimizer).
            *   Logging training and validation loss.
            *   Periodically saving model checkpoints to a `./checkpoints` directory.

---

## Phase 3: Instruction-Tuning Dataset Creation (`TinyStories-Instruct`)

**Objective:** Create a dataset for fine-tuning the model to follow specific instructions.

**Steps:**

1.  **Instruction Generation:**
    *   **Action:** For each story in the original dataset, generate a corresponding set of instructions.
    *   **Details:** The instructions are derived from the story itself and its generation parameters. The four types of instructions are:
        1.  **Words:** The list of 3 words used to prompt the story.
        2.  **Sentence:** A sentence randomly extracted from the story.
        3.  **Features:** The list of features (e.g., `dialogue`, `twist`) used.
        4.  **Summary:** A 1-2 line summary of the story.

2.  **Instruction Assembly Script (`create_instruction_dataset.py`):**
    *   **Action:** Develop a script to create the final instruction dataset.
    *   **Details:** The script will:
        *   Use an LLM (like GPT-3.5) to generate a summary for each story.
        *   For each story, gather the associated words, features, a random sentence, and the new summary.
        *   Create various instruction sets by randomly combining these four elements.
        *   Format the final data as a JSONL file (`instruction_dataset.jsonl`), where each line is a JSON object: `{"instruction": "...", "story": "..."}`.

---

## Phase 4: Model Instruction Tuning

**Objective:** Fine-tune the pretrained `TinyStories` model to become an instruction-following model.

**Steps:**

1.  **Fine-tuning Script (`instruction_tune.py`):**
    *   **Action:** Create a script for supervised fine-tuning (SFT).
    *   **Details:**
        *   Load a pretrained model checkpoint from Phase 2.
        *   Load the `instruction_dataset.jsonl`.
        *   For each data point, format the input as a single string (e.g., `### Instruction:
{instruction}

### Response:
{story}`).
        *   Implement a training loop that calculates loss only on the "Response" (story) part of the sequence to teach the model to respond to instructions.
        *   Train for a small number of epochs.
        *   Save the final instruction-tuned model.

---

## Phase 5: Evaluation and Comparison (`GPT-Eval`)

**Objective:** Evaluate the model's story generation capabilities using GPT-4 as a judge and compare performance across different model sizes and against baselines.

**Steps:**

1.  **Create Evaluation Prompts:**
    *   **Action:** Manually write a set of ~50 diverse, high-quality story beginnings.
    *   **Details:** These prompts will serve as the basis for evaluating the model's completion abilities. Save them in `evaluation_prompts.json`.

2.  **Generate Completions (`generate_completions.py`):**
    *   **Action:** Write a script to generate stories from the evaluation prompts.
    *   **Details:**
        *   Load the trained models (both the base pretrained and the instruction-tuned versions).
        *   For each prompt in `evaluation_prompts.json`, generate ~10 different completions using nucleus sampling (temperature=1.0) to ensure diversity.
        *   Save the completions for evaluation.

3.  **GPT-4 Evaluation Script (`evaluate_with_gpt4.py`):**
    *   **Action:** Create a script to have GPT-4 grade the generated stories.
    *   **Details:**
        *   The script will iterate through the generated completions.
        *   For each one, it will send a request to the GPT-4 API with a carefully crafted prompt. The prompt will ask GPT-4 to act as a teacher and grade the story on:
            *   `Grammar`
            *   `Creativity`
            *   `Consistency` (with the prompt)
            *   `Plot` (for instruction-tuned models)
        *   The script will parse the numerical scores from the GPT-4 response and save them.

4.  **Analysis and Reporting:**
    *   **Action:** Aggregate the scores and compare the models.
    *   **Details:**
        *   Calculate the average scores for each model across all prompts.
        *   Generate tables and charts to visualize the results, comparing:
            *   Models of different sizes and architectures.
            *   The impact of instruction tuning.
            *   Performance against a baseline model like `GPT-2-XL`.
        *   Summarize the findings in a `results.md` file.


