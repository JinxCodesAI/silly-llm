# Phase 1: Synthetic Dataset Creation Pipeline Specification

## 1. Objective
The primary objective of Phase 1 is to generate a large-scale synthetic dataset of short, simple, and coherent stories, adhering to the principles outlined in the TinyStories paper. This dataset will serve as the pre-training corpus for small language models.

## 2. Model and Framework
*   **Language Model for Generation:** Qwen (specifically, a suitable variant like Qwen-1.8B or Qwen-0.5B, depending on availability and performance, to be loaded from Hugging Face).
*   **Framework:** Hugging Face `transformers` library for model loading, inference, and tokenization. The `accelerate` library will be used for efficient multi-GPU and distributed generation.

## 3. Vocabulary Compilation

*   **Objective:** Programmatically generate a `vocabulary.json` file containing approximately 1,500 simple words (nouns, verbs, adjectives) suitable for a 3-4 year old, using an LLM.
*   **Script Path:** `/teamspace/studios/this_studio/generate_vocabulary.py`

### 3.1. `generate_vocabulary.py` Script Specification

#### 3.1.1. Configuration and Argument Parsing
The script will accept command-line arguments:
*   `--model_name_or_path`: Hugging Face model identifier for the LLM used for vocabulary generation (e.g., `Qwen/Qwen-1_8B`).
*   `--output_path`: Path to save the generated `vocabulary.json` (e.g., `/teamspace/studios/this_studio/vocabulary.json`).
*   `--target_word_count`: Desired total number of words (e.g., 1500).
*   `--batch_size`: Number of words to request from the LLM in each iteration.
*   `--max_retries`: Maximum retries for LLM calls.
*   `--temperature`: Sampling temperature for LLM generation.

#### 3.1.2. Iterative Generation Loop
The script will operate in an iterative loop until `target_word_count` is reached:

1.  **Initialize:** Load existing `vocabulary.json` if it exists, otherwise start with empty lists for `nouns`, `verbs`, `adjectives`.
2.  **Determine Remaining Words:** Calculate how many more words are needed for each category to reach the target (e.g., if target is 1500 total, and current is 267, then 1233 words are remaining. Distribute this proportionally or based on a fixed ratio, e.g., 50% nouns, 25% verbs, 25% adjectives).
3.  **Prompt Construction:**
    *   For each category (`nouns`, `verbs`, `adjectives`), construct a prompt for the LLM.
    *   The prompt will ask the LLM to generate a list of `batch_size` simple words suitable for a 3-4 year old, specifically for that category.
    *   **Example Prompt:**
        ```
        "Generate a comma-separated list of 20 simple English nouns that a 3-4 year old child would likely understand. Examples: cat, dog, house."
        ```
        (Adjust `20` to `batch_size`)
4.  **LLM Query and Parsing:**
    *   Send the prompt to the LLM (using `transformers` for Qwen).
    *   Parse the LLM's response to extract the list of words. Robust parsing will be needed to handle various LLM output formats (e.g., comma-separated, bullet points, numbered lists).
5.  **Validation and Filtering:**
    *   **Duplicate Check:** Before adding, check if each generated word already exists in the current vocabulary (case-insensitive). Only add unique words.
    *   **Simplicity Check (Optimal):** Optionally, a secondary LLM call or a pre-defined simple word list could be used to cross-reference and filter words that are too complex. For initial implementation, rely on the LLM's ability to follow the "simple words" instruction.
    *   **Format Check:** Ensure words are single words and not phrases.
6.  **Update Vocabulary:** Add the validated, unique words to the respective category lists.
7.  **Save Progress:** Periodically save the updated vocabulary to `vocabulary.json` to prevent data loss.
8.  **Loop Termination:** Continue until the total word count across all categories reaches `target_word_count`.

#### 3.1.3. Age-Appropriateness Considerations
*   The primary mechanism for age-appropriateness will be the explicit instruction in the LLM prompt to generate words suitable for a "3-4 year old child."
*   Manual review of the generated `vocabulary.json` will be crucial to ensure quality and adherence to the age-appropriateness criteria. The script will facilitate this by providing a starting point.

### 3.2. `vocabulary.json` Structure and Usage
*   **File Path:** `/teamspace/studios/this_studio/vocabulary.json`
*   **Structure:** The JSON file will contain three top-level keys: `nouns`, `verbs`, and `adjectives`, each mapping to a list of simple English words.
    ```json
    {
      "nouns": ["cat", "dog", "house", ...],
      "verbs": ["run", "play", "eat", ...],
      "adjectives": ["happy", "big", "small", ...]
    }
    ```
*   **Usage:** The `generate_prompts.py` script will load this file at startup and randomly select words from these lists for prompt construction, as described in section 4.2.

## 4. Prompt Generation Script (`generate_prompts.py`)

This script will generate a fixed set of prompts that will be used by the story generation pipeline to ensure reproducibility.

*   **Script Path:** `/teamspace/studios/this_studio/generate_prompts.py`

### 4.1. Configuration and Argument Parsing
The script will accept command-line arguments for flexible execution:
*   `--output_path`: Path to save the generated prompts JSONL file (e.g., `/teamspace/studios/this_studio/generated_prompts.jsonl`).
*   `--num_prompts`: Total number of prompts to generate.
*   `--vocab_path`: Path to `vocabulary.json`.
*   `--features_path`: Path to a JSON file defining story features (e.g., `docs/story_features.json`).
*   `--seed`: Random seed for reproducibility.

### 4.2. Prompt Generation Logic

1.  **Load Resources:**
    *   Load `vocabulary.json`.
    *   Load `docs/story_features.json` (a new file to be created, containing a list of features like "dialogue", "plot twist", "bad ending", "moral value").
2.  **Iterative Prompt Construction:** For each prompt to be generated (up to `--num_prompts`):
    *   **Word Selection:** Randomly select one noun, one verb, and one adjective from `vocabulary.json`.
    *   **Feature Selection:** Randomly select a subset (0 to N) of story features from `docs/reviewlet's work`.
    *   **Prompt Template:** Construct a detailed prompt for the Qwen model. The prompt will instruct the model to write a 3-5 paragraph story using only simple words (from the vocabulary), incorporating the selected words and features.
        *   **Example Prompt Structure:**
            ```
            "Write a short story (3-5 paragraphs) which only uses very simple words that a 3-4 year old child would likely understand.
            The story should use the verb "{verb}", the noun "{noun}" and the adjective "{adjective}".
            {features_clause}
            Remember to only use simple words!
            "
            ```
            (Where `{features_clause}` is constructed as follows:
            If `selected_features` is empty, `{features_clause}` is an empty string.
            If `selected_features` is not empty, `{features_clause}` is formed by joining the natural language feature phrases (from `docs/story_features.json`) with ", " and prepending "Additionally, ". For example: "Additionally, make sure the story contains a dialogue, include a plot twist.")
    *   **Store Prompt:** Save the constructed prompt along with its metadata (selected words, features) into a list.

### 4.3. Output Format and Storage
*   **File Format:** Each generated prompt and its associated metadata will be stored as a single JSON object per line in a JSONL (JSON Lines) file.
*   **File Path:** `--output_path` (e.g., `/teamspace/studios/this_studio/generated_prompts.jsonl`)
*   **JSONL Entry Structure:** Each line in the JSONL file will be a JSON object like:
    ```json
    {
      "prompt_id": "unique_id_abc",
      "prompt_text": "Write a short story...",
      "selected_words": {"noun": "...", "verb": "...", "adjective": "..."},
      "selected_features": ["dialogue", "plot twist"]
    }
    ```

## 5. Story Generation Pipeline (`batch_generate.py`)

This script will orchestrate the entire data generation process, focusing on efficiency and scalability.

*   **Script Path:** `/teamspace/studios/this_studio/batch_generate.py`

### 5.1. Configuration and Argument Parsing
The script will accept command-line arguments for flexible execution:
*   `--model_name_or_path`: Hugging Face model identifier for Qwen (e.g., `Qwen/Qwen-1_8B`).
*   `--prompts_path`: Path to the JSONL file containing pre-generated prompts (e.g., `/teamspace/studios/this_studio/generated_prompts.jsonl`).
*   `--output_dir`: Directory to save generated data (e.g., `/teamspace/studios/this_studio/generated_stories/`).
*   `--batch_size`: Number of prompts to process in parallel during generation.
*   `--max_new_tokens`: Maximum length of generated stories.
*   `--temperature`: Sampling temperature for generation (e.g., 0.7-1.0 for diversity).
*   `--top_k`, `--top_p`: Top-k and Top-p sampling parameters.
*   `--repetition_penalty`: Penalty for repeating tokens.
*   `--num_gpus`: Number of GPUs to utilize (defaults to all available, or specified).

### 5.2. Data Generation with Qwen and Hugging Face Transformers (Revised)

1.  **Load Prompts:** Load the prompts from the `prompts_path` JSONL file.
2.  **Model and Tokenizer Loading:**
    *   Load the specified Qwen model and its corresponding tokenizer using `AutoModelForCausalLM` and `AutoTokenizer` from `transformers`.
    *   Ensure the model is loaded onto available GPUs efficiently using `accelerate` or `torch.nn.DataParallel` for multi-GPU setups.
3.  **Logit-Based Branching Generation (Depth-First Search):**
    This approach will involve a custom, step-by-step generation loop that inspects token probabilities (logits) at each step to identify branching points and generate families of similar stories.

    *   **Core Logic:** For each initial prompt from the loaded `prompts_path`:
        a.  **Initialization:** Create a stack to manage generation paths. Each entry on the stack will represent a potential story variant to explore and will contain:
            *   `current_input_ids`: The token IDs generated so far for this path.
            *   `past_key_values`: The KV cache state at the beginning of this path segment.
            *   `generated_text_prefix`: The decoded text corresponding to `current_input_ids`.
            *   `branch_depth`: Current depth of branching (to limit exploration).
        b.  **Iterative Token Generation:** While the stack is not empty and `max_variants_per_prompt` has not been reached:
            *   Pop a path from the stack.
            *   Generate the next token(s) one by one, using the model's `forward` pass to get logits and update `past_key_values`.
            *   **Branching Decision:** At each token generation step:
                *   Calculate softmax probabilities from the model's logits for the next token.
                *   Identify the token with the highest probability (the "primary" path).
                *   Identify "alternative" tokens: all tokens whose probability is above a `low_threshold` (e.g., 0.01, as seen in `scripts/main.py`) and whose probability is *not* above a `high_threshold` (e.g., 0.9) for the primary path. This ensures branching on plausible but less certain alternatives.
                *   If `branch_depth` is below `max_branching_depth` and `max_variants_per_prompt` has not been reached, store these alternative tokens and their corresponding generation states (including `current_input_ids` and `past_key_values`) as potential "branches" onto the stack for later exploration. Prioritize adding branches that represent earlier divergence points to the stack to maintain a depth-first exploration.
            *   **Story Completion:** Continue generating along the current path until `max_new_tokens` is reached or an end-of-sequence token is generated. This forms one story variant.
            *   Add the completed story variant to the list of generated stories for the current prompt.

    *   **Parameters for Branching:**
        *   `--high_prob_threshold`: (e.g., 0.9) Tokens with probability above this are considered highly likely and typically not branched from.
        *   `--low_prob_threshold`: (e.g., 0.01) Tokens with probability above this (but below `high_prob_threshold`) are considered valid alternatives for branching.
        *   `--max_variants_per_prompt`: Maximum number of story variants to generate for each initial prompt. This bounds the exploration.
        *   `--max_branching_depth`: (Optional) Limits how many branching points deep the search can go, preventing excessively long or complex branching trees.

#### Performance Considerations:

*   **KV Caching (Key-Value Caching):**
    *   **Crucial for Efficiency:** KV caching is paramount. When exploring a branch, the initial part of the story (before the branching point) is identical to the primary path. By saving and restoring the `past_key_values` (as demonstrated in `scripts/main.py`'s `outputs.past_key_values`) at each branching point, we avoid recomputing the attention keys and values for the common prefix of the sequence. This significantly reduces redundant computation.
    *   **Implementation:** The custom generation loop will explicitly manage `past_key_values`. When a branch is taken, the `past_key_values` from the branching point are used as the starting state for the new generation. This requires careful handling of the `past_key_values` tensor across different generation paths.

*   **Parallelism:**
    *   **Batching Initial Prompts:** The most effective form of parallelism will be processing multiple *initial* prompts in parallel. Each prompt in a batch can independently generate its family of stories using the logit-based branching logic.
    *   **Multi-GPU/Distributed Execution:** `accelerate` will be used to distribute these batches of initial prompts across multiple GPUs or even multiple nodes. Each GPU would run its own instance of the branching generation logic for a subset of the prompts, maximizing hardware utilization.
    *   **Within-Prompt Parallelism:** Parallelizing the exploration of branches *within a single prompt's family* on a single GPU is generally inefficient due to the sequential nature of token generation and the need to restore specific KV cache states. The depth-first search approach inherently serializes this exploration for a given prompt.

*   **Efficiency Optimizations:**
    *   **Threshold Tuning:** The `high_prob_threshold` and `low_prob_threshold` are critical hyperparameters. They need to be carefully tuned to balance diversity (more branches) and coherence (fewer, more plausible branches). Empirical testing will be necessary.
    *   **Variant Bounding:** `max_variants_per_prompt` and `max_branching_depth` are essential to prevent combinatorial explosion and control the computational cost. The depth-first search naturally manages the number of active generations.
    *   **Early Stopping for Branches:** Consider implementing criteria to prune branches that quickly become incoherent or irrelevant to the desired story characteristics, saving computation.
    *   **Memory Management:** Generating multiple variants and managing their respective KV caches will be memory-intensive. Careful batch sizing and potentially offloading strategies (if `accelerate` supports it for custom loops) will be important.

4.  **Error Handling:** Implement robust error handling for API calls and generation failures (e.g., retry mechanisms, logging failed prompts).

### 5.3. Output Format and Storage (for stories)

*   **File Format:** Each generated story and its associated metadata will be stored as a single JSON object per line in a JSONL (JSON Lines) file. This format is highly efficient for large datasets and preserves metadata.
*   **Folder Structure:**
    *   `output_dir/` (e.g., `/teamspace/studios/this_studio/generated_stories/`)
        *   `generation_run_YYYYMMDD_HHMMSS/` (timestamped directory for each run)
            *   `stories_part_000.jsonl`
            *   `stories_part_001.jsonl`
            *   ...
            *   `generation_log.txt` (detailed log of the generation process, including parameters used, errors, and progress)
*   **JSONL Entry Structure:** Each line in the JSONL file will be a JSON object like:
    ```json
    {
      "story_id": "unique_id_123",
      "prompt": "...",
      "generated_text": "Once upon a time...",
      "selected_words": {"noun": "...", "verb": "...", "adjective": "..."},
      "selected_features": ["dialogue", "plot twist"],
      "generation_params": {"temperature": 0.8, "max_new_tokens": 200, ...},
      "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
    }
    ```
*   **Performance Considerations:**
    *   **Checkpointing:** Periodically save generated stories to disk to prevent data loss in case of interruptions.
    *   **Parallelism:** Leverage `accelerate` for distributed training/inference across multiple GPUs or even multiple nodes if needed. The `batch_size` parameter will directly control GPU memory usage and throughput.
    *   **Memory Management:** Implement strategies to manage GPU memory, such as gradient accumulation if fine-tuning, or careful batch sizing during inference.
    *   **Logging:** Comprehensive logging of progress, generation speed (stories/second, tokens/second), and any errors.

## 6. Dataset Finalization

*   **Consolidation (Optional):** If stories are generated in multiple smaller JSONL files (e.g., due to checkpointing or distributed runs), a separate utility script can concatenate them into a single large JSONL file or convert them into a Hugging Face `Dataset` object for easier downstream processing.
*   **Pre-training Format:** The final output for pre-training will be a large JSONL file containing all generated stories and their metadata. This format is easily loadable by Hugging Face `datasets` library for tokenization and model training.

## 7. Next Steps (Pre-computation)

Before proceeding to Phase 2 (Small Language Model Pretraining), the generated dataset will need to be tokenized and chunked into fixed-size sequences. This will be a separate pre-processing step, potentially using the Hugging Face `datasets` library's mapping functionalities for efficiency.
