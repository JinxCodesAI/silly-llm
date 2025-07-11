# Custom Validation System

This module provides a flexible custom validation system for synthetic story generation. It allows you to use LLM providers to validate generated stories according to custom criteria.

## Overview

The validation system consists of:

- **BaseValidator**: Abstract base class for all validators
- **QualityValidator**: Example implementation for English language validation
- **ValidationResult**: Data model for validation outcomes
- **Configuration Support**: Integration with the main configuration system

## Configuration

Add validation configuration to your JSON config file:

```json
{
  "validation_settings": {
    "validate_stories": true,
    "min_words": 50,
    "max_words": 300,
    "custom_validation": {
      "model_name": "Qwen/Qwen3-4B",
      "provider": "TransformersProvider",
      "validator_class": "training.synthetic_data_generation.validation.QualityValidator",
      "generation": {
        "batch_size": 8,
        "max_new_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": true,
        "repetition_penalty": 1.0
      }
    }
  }
}
```

### Configuration Parameters

- **model_name**: Model to use for validation (can be different from generation model)
- **provider**: Provider type (`TransformersProvider`, `OpenAICompatible`, `MockProvider`)
- **validator_class**: Full Python path to validator class
- **generation**: Generation parameters for validation (inherits from main config if not specified)
  - **batch_size**: Number of stories to validate in parallel (default: 1 for backward compatibility)

## Supported Providers

### TransformersProvider
Uses local HuggingFace models for validation:
```json
{
  "model_name": "Qwen/Qwen3-4B",
  "provider": "TransformersProvider"
}
```

### OpenAICompatible
Uses OpenAI-compatible APIs:
```json
{
  "model_name": "gpt-3.5-turbo",
  "provider": "OpenAICompatible"
}
```

### MockProvider
For testing without real models:
```json
{
  "model_name": "mock-validator",
  "provider": "MockProvider"
}
```

## Built-in Validators

### QualityValidator

Validates that stories contain only English words using the prompt:
```
Does provided story contain only English words, think step by step and finish generation with either ANSWER:NO or ANSWER:YES

STORY TO ANALYZE: {story}
```

The validator looks for `ANSWER:YES` or `ANSWER:NO` patterns in the response and falls back to keyword analysis if the pattern is not found.

## Creating Custom Validators

The validation system uses a clean separation of concerns:

- **BaseValidator**: Handles LLM provider interaction and common logic
- **Custom Validators**: Focus on prompt formatting and response parsing

### Simple Example

```python
from training.synthetic_data_generation.validation.base import BaseValidator, CustomValidationResult

class SimpleValidator(BaseValidator):
    def get_validation_prompt(self, story_content: str) -> str:
        """Generate the validation prompt."""
        return f"Is this appropriate for children? Answer YES or NO.\n\nStory: {story_content}"

    def parse_validation_response(self, response: str) -> CustomValidationResult:
        """Parse the LLM response."""
        if "YES" in response.upper():
            return CustomValidationResult(is_valid=True, score=1.0, reasoning="Approved")
        else:
            return CustomValidationResult(is_valid=False, score=0.0, reasoning="Rejected")
```

### Advanced Example with Custom Scoring

```python
class ScoredValidator(BaseValidator):
    def get_validation_prompt(self, story_content: str) -> str:
        return (
            "Rate this story 1-10 for age appropriateness. "
            "Format: SCORE: X\n\n"
            f"Story: {story_content}"
        )

    def parse_validation_response(self, response: str) -> CustomValidationResult:
        # Extract score from "SCORE: X" pattern
        try:
            score_text = response.upper().split("SCORE:")[1].strip()
            score = int(score_text.split()[0])
            normalized_score = (score - 1) / 9  # Convert 1-10 to 0-1
            is_valid = score >= 6

            return CustomValidationResult(
                is_valid=is_valid,
                score=normalized_score,
                details={"raw_score": score},
                reasoning=f"Scored {score}/10"
            )
        except:
            return CustomValidationResult(
                is_valid=False,
                score=0.0,
                reasoning="Could not parse score"
            )
```

### Key Benefits

1. **BaseValidator handles**: Provider interaction, error handling, common details
2. **Custom validators handle**: Prompt formatting, response parsing logic
3. **Clean separation**: Easy to test and maintain
4. **Consistent interface**: All validators work the same way in the pipeline

## Usage Examples

### Basic Usage with Config File

```bash
# Use validation with transformers provider
python -m training.synthetic_data_generation.main \
    --config config/validation_example_config.json

# Use validation with OpenAI API
export AI_API_KEY=your_key
python -m training.synthetic_data_generation.main \
    --config config/openai_validation_config.json \
    --openai-provider

# Test with mock provider
python -m training.synthetic_data_generation.main \
    --config config/mock_validation_config.json \
    --mock-provider
```

### Programmatic Usage

```python
from training.synthetic_data_generation.validation import QualityValidator
from training.common.llm_providers import TransformersProvider

# Create provider
provider = TransformersProvider(model_name="Qwen/Qwen3-4B")

# Create validator with batch support
config = {
    "generation": {
        "batch_size": 8,
        "max_new_tokens": 128,
        "temperature": 0.1
    }
}
validator = QualityValidator(provider, config)

# Validate a single story
story = "Once upon a time, there was a happy cat."
result = await validator.validate(story)

print(f"Valid: {result.is_valid}")
print(f"Score: {result.score}")
print(f"Reasoning: {result.reasoning}")

# Validate multiple stories in batch
stories = [
    "Once upon a time, there was a happy cat.",
    "The brave little mouse saved the day.",
    "A magical forest full of friendly animals."
]
results = await validator.validate_batch(stories)

for i, result in enumerate(results):
    print(f"Story {i+1}: Valid={result.is_valid}, Score={result.score}")
```

## Validation Flow

### Individual Validation (Legacy)
1. **Story Generation**: Stories are generated using the main LLM provider
2. **Traditional Validation**: Basic checks (word count, required words) for each story
3. **Custom Validation**: If configured, each story is validated individually
4. **Result Processing**: Only stories that pass all validations are included in output

### Batch Validation (New)
1. **Story Generation**: Stories are generated using the main LLM provider
2. **Traditional Validation**: Basic checks (word count, required words) for all stories
3. **Batch Custom Validation**: Valid stories are sent for validation in batches
4. **Result Processing**: Only stories that pass all validations are included in output

The system automatically chooses batch validation when:
- Custom validator is configured
- Validator supports `validate_batch()` method
- `generation.batch_size > 1` in validation settings

## Performance Considerations

- Validation adds processing time (each story requires an additional LLM call)
- Use smaller/faster models for validation when possible
- **Batch validation** significantly improves throughput by validating multiple stories simultaneously
- Configure `generation.batch_size` in validation settings to control parallelization
- Larger batch sizes improve efficiency but use more memory
- Mock provider is useful for testing without validation overhead

### Batch Validation Performance

The validation system now supports batch processing for improved performance:

- **Individual validation** (batch_size=1): Each story validated separately (legacy behavior)
- **Batch validation** (batch_size>1): Multiple stories validated in parallel
- **Automatic fallback**: If batch validation fails, falls back to individual validation
- **Memory efficiency**: Batch size should be tuned based on available GPU memory

Recommended batch sizes:
- **Local models**: 8-16 stories per batch
- **API providers**: 4-8 stories per batch (to avoid rate limits)
- **Large models**: 2-4 stories per batch (memory constraints)

## Error Handling

- Validation errors are logged but don't stop generation
- Stories with validation errors are excluded from output
- Provider initialization errors disable custom validation
- Fallback to traditional validation if custom validation fails
- **Batch validation fallback**: If batch validation fails, automatically falls back to individual validation
- **Graceful degradation**: Missing batch_size parameter defaults to individual validation (batch_size=1)

## Backward Compatibility

The batch validation implementation maintains full backward compatibility:

- **Existing validators**: Continue to work without modification
- **Configuration files**: Work with or without `batch_size` parameter
- **Individual validation**: Still available and used as fallback
- **Default behavior**: If `batch_size` is not specified, defaults to 1 (individual validation)
- **Legacy support**: Validators without `validate_batch()` method automatically use individual validation

## Testing

Run the validation tests:
```bash
python -m pytest training/synthetic_data_generation/tests/test_validation.py -v
```

Run the demo:
```bash
python -m training.synthetic_data_generation.demo_validation
```
