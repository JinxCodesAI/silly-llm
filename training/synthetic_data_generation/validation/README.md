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

# Create validator
config = {"generation": {"max_new_tokens": 128, "temperature": 0.1}}
validator = QualityValidator(provider, config)

# Validate a story
story = "Once upon a time, there was a happy cat."
result = await validator.validate(story)

print(f"Valid: {result.is_valid}")
print(f"Score: {result.score}")
print(f"Reasoning: {result.reasoning}")
```

## Validation Flow

1. **Story Generation**: Stories are generated using the main LLM provider
2. **Traditional Validation**: Basic checks (word count, required words)
3. **Custom Validation**: If configured, stories are validated using the custom validator
4. **Result Processing**: Only stories that pass all validations are included in output

## Performance Considerations

- Validation adds processing time (each story requires an additional LLM call)
- Use smaller/faster models for validation when possible
- Consider batch validation for better throughput
- Mock provider is useful for testing without validation overhead

## Error Handling

- Validation errors are logged but don't stop generation
- Stories with validation errors are excluded from output
- Provider initialization errors disable custom validation
- Fallback to traditional validation if custom validation fails

## Testing

Run the validation tests:
```bash
python -m pytest training/synthetic_data_generation/tests/test_validation.py -v
```

Run the demo:
```bash
python -m training.synthetic_data_generation.demo_validation
```
