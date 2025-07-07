# Plan: Recreating the TinyStories Methodology

This document outlines the step-by-step plan to replicate the ideas presented in the "TinyStories" paper. The project is divided into five main phases:

1.  **Phase 1: Synthetic Dataset Creation (`TinyStories`)**
2.  **Phase 2: Small Language Model Pretraining**
3.  **Phase 3: Instruction-Tuning Dataset Creation (`TinyStories-Instruct`)**
4.  **Phase 4: Model Instruction Tuning**
5.  **Phase 5: Evaluation and Comparison (`GPT-Eval`)**

---

## Phase 1: Synthetic Dataset Creation (`TinyStories`)

**Objective:** Generate a large dataset of simple, coherent short stories using a vocabulary understandable by a 3-4 year old.

**Steps:**

1.  **Vocabulary Compilation:**
    *   **Action:** Create a vocabulary file (`vocabulary.json`) containing a list of simple words.
    *   **Details:** The list should be categorized into `nouns`, `verbs`, and `adjectives`. Aim for approximately 1,500 words, mimicking the vocabulary of a young child. This can be sourced from educational resources for early childhood language development.

2.  **Story Generation Script (`batch_generate.py`):**
    *   **Action:** Develop a Python script to automate story generation using a powerful foundation model like GPT-4.
    *   **Details:** The script will:
        *   Load the `vocabulary.json`.
        *   Define a list of "story features" (e.g., `dialogue`, `plot twist`, `bad ending`, `moral value`) to enhance diversity.
        *   Loop to generate a target number of stories. In each iteration:
            *   Randomly select one noun, one verb, and one adjective from the vocabulary.
            *   Randomly select a subset of story features.
            *   Construct a detailed prompt for the LLM, instructing it to write a 3-5 paragraph story using only simple words, incorporating the selected words and features.
            *   Call the LLM's API to generate the story.
            *   Append the generated story to a text file (`tinystories_dataset.txt`).

3.  **Dataset Finalization:**
    *   **Action:** Run the `batch_generate.py` script to produce a substantial dataset.
    *   **Details:** The final output will be a single large text file, with each story separated by a unique delimiter. This format will be used for tokenization and training in the next phase.

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
