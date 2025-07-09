# K-Shot Prompting Functionality Improvement Plan

## Executive Summary

This document outlines a comprehensive improvement plan for the k-shot prompting functionality in the synthetic data generation pipeline. The current implementation has several limitations that make it complex to use and maintain. This plan addresses three key goals: enabling JSON-based k-shot samples, simplifying usage by removing graceful degradation, and ensuring provider-agnostic functionality.

## Current State Analysis

### Current Implementation Overview

The k-shot prompting system currently consists of:

1. **PromptGenerator** (`training/synthetic_data_generation/src/prompt_generator.py`)
   - Manages k-shot example selection and prompt generation
   - Uses `parse_conversation_examples()` to load examples from text files
   - Implements graceful degradation when examples are missing

2. **TemplateManager** (`training/synthetic_data_generation/src/template_manager.py`)
   - Formats prompts with k-shot examples
   - Provides `format_for_chat_template()` for model consumption
   - Handles template creation and formatting

3. **Data Models** (`training/common/data_models.py`)
   - `KShotExample`: Individual message with role and content
   - `ConversationExample`: Complete conversation with multiple messages
   - `StoryPrompt`: Contains k-shot examples and metadata

4. **Utilities** (`training/common/utils.py`)
   - `parse_conversation_examples()`: Parses text-based conversation files
   - Uses regex parsing for PROMPT:/RESPONSE: format

### Current Data Sources

1. **Text-based format** (`training/synthetic_data_generation/config/example_conversation.txt`)
   - Uses PROMPT:/RESPONSE: markers
   - Requires regex parsing
   - Difficult to maintain and extend

2. **JSON samples** (`docs/k_shot_prompting_samples.json`)
   - Well-structured format with metadata
   - Contains multiple example configurations
   - Currently not used by the pipeline

### Current Usage Pattern

```bash
python -m training.synthetic_data_generation.main \
    --openai-provider \
    --model-name gpt-4.1-nano \
    --num-stories 10 \
    --conversation-examples-path "path/to/examples.txt"
```

### Identified Problems

1. **Complex Data Format**: Text-based format with regex parsing is error-prone
2. **Graceful Degradation**: System continues with warnings when k-shot data is missing
3. **Limited Flexibility**: Cannot easily switch between different k-shot configurations
4. **Provider Coupling**: Some k-shot logic is tied to specific providers
5. **Maintenance Burden**: Multiple data formats and parsing methods
6. **Poor Error Handling**: Failures are silently ignored or produce warnings

## Improvement Goals

### Goal 1: Enable JSON-based K-Shot Sources

**Objective**: Make it possible to use files like `docs/k_shot_prompting_samples.json` as a source for k-shot generation while keeping current parsing as an alternative.

**Benefits**:
- Structured, maintainable data format
- Rich metadata support
- Multiple configuration options in one file
- Better validation and error detection

### Goal 2: Simplify K-Shot Usage

**Objective**: Stop using graceful degradation principle - things should fail clearly if something is missing, with minimal required parameters.

**Benefits**:
- Clear error messages when configuration is wrong
- Reduced complexity in error handling
- More predictable behavior
- Easier debugging and troubleshooting

### Goal 3: Provider-Agnostic K-Shot Support

**Objective**: Ensure k-shot prompting works consistently with any provider (OpenAI, Transformers, Mock).

**Benefits**:
- Consistent behavior across providers
- Simplified command-line usage
- Better testing capabilities
- Future-proof architecture

## Proposed Solution Architecture

### 1. New K-Shot Configuration System

#### 1.1 Enhanced Data Models

```python
class KShotConfiguration(BaseModel):
    """Configuration for k-shot prompting."""
    name: str
    description: str
    k_shot_count: int
    messages: List[KShotExample]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KShotSource(BaseModel):
    """Container for multiple k-shot configurations."""
    description: str
    format: str
    configurations: List[KShotConfiguration]
    usage_notes: List[str] = Field(default_factory=list)
```

#### 1.2 Unified K-Shot Loader

```python
class KShotLoader:
    """Unified loader for k-shot examples from multiple sources."""
    
    def load_from_json(self, file_path: str) -> List[KShotConfiguration]:
        """Load k-shot configurations from JSON file."""
        
    def load_from_text(self, file_path: str) -> List[KShotConfiguration]:
        """Load k-shot configurations from legacy text format."""
        
    def get_configuration(self, name: Optional[str] = None) -> KShotConfiguration:
        """Get specific configuration by name or default."""
```

### 2. Simplified Command-Line Interface

#### 2.1 New Parameters

```bash
# Use JSON-based k-shot configuration
--k-shot-config-file "docs/k_shot_prompting_samples.json"
--k-shot-config-name "2-shot example with dialogue and moral value"

# Use legacy text format (backward compatibility)
--conversation-examples-path "config/example_conversation.txt"

# Explicit k-shot control
--k-shot-count 2
--require-k-shot  # Fail if k-shot data is missing
```

#### 2.2 Updated Usage Examples

```bash
# Using JSON configuration with specific named config
python -m training.synthetic_data_generation.main \
    --openai-provider \
    --model-name gpt-4.1-nano \
    --num-stories 10 \
    --k-shot-config-file "docs/k_shot_prompting_samples.json" \
    --k-shot-config-name "2-shot example with dialogue and moral value"

# Using JSON configuration with default config
python -m training.synthetic_data_generation.main \
    --openai-provider \
    --model-name gpt-4.1-nano \
    --num-stories 10 \
    --k-shot-config-file "docs/k_shot_prompting_samples.json"

# Legacy text format (backward compatibility)
python -m training.synthetic_data_generation.main \
    --openai-provider \
    --model-name gpt-4.1-nano \
    --num-stories 10 \
    --conversation-examples-path "config/example_conversation.txt" \
    --require-k-shot
```

### 3. Enhanced Error Handling

#### 3.1 Strict Validation Mode

- `--require-k-shot` flag makes k-shot data mandatory
- Clear error messages when configuration is invalid
- Validation of k-shot data structure before generation starts
- Provider-specific validation for k-shot compatibility

#### 3.2 Error Message Examples

```
ERROR: K-shot configuration required but not provided. Use --k-shot-config-file or --conversation-examples-path
ERROR: K-shot configuration 'invalid-name' not found in docs/k_shot_prompting_samples.json
ERROR: K-shot file 'missing.json' does not exist
ERROR: Invalid k-shot format in configuration 'test-config': missing required field 'messages'
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### Task 1.1: Enhance Data Models
- Add `KShotConfiguration` and `KShotSource` models
- Update existing models to support new structure
- Add validation methods

#### Task 1.2: Create KShotLoader Class
- Implement JSON loading functionality
- Maintain backward compatibility with text format
- Add configuration selection logic

#### Task 1.3: Update Utilities
- Extend `parse_conversation_examples()` for backward compatibility
- Add JSON parsing utilities
- Implement validation functions

### Phase 2: Integration (Week 2)

#### Task 2.1: Update PromptGenerator
- Integrate KShotLoader
- Remove graceful degradation logic
- Add strict validation mode

#### Task 2.2: Update TemplateManager
- Ensure compatibility with new k-shot structure
- Maintain existing formatting capabilities
- Add provider-specific optimizations

#### Task 2.3: Update Configuration System
- Add new command-line parameters
- Update configuration file schema
- Implement parameter validation

### Phase 3: Command-Line Interface (Week 3)

#### Task 3.1: Update Main Entry Point
- Add new command-line arguments
- Implement parameter validation
- Update help documentation

#### Task 3.2: Update Error Handling
- Implement strict validation mode
- Add comprehensive error messages
- Remove graceful degradation paths

#### Task 3.3: Provider Integration Testing
- Test with OpenAI provider
- Test with Transformers provider
- Test with Mock provider

### Phase 4: Documentation and Testing (Week 4)

#### Task 4.1: Update Documentation
- Update DATA_GENERATION.md
- Add usage examples
- Document migration path

#### Task 4.2: Create Migration Guide
- Document changes for existing users
- Provide conversion tools if needed
- Update example configurations

#### Task 4.3: Comprehensive Testing
- Unit tests for new components
- Integration tests for all providers
- Backward compatibility tests

## Detailed Technical Specifications

### 1. JSON Configuration Format

The new JSON format will be based on the existing `docs/k_shot_prompting_samples.json` structure:

```json
{
  "description": "K-shot configurations for bedtime story generation",
  "format": "OpenAI chat format with role/content messages",
  "configurations": [
    {
      "name": "2-shot-dialogue-moral",
      "description": "2-shot example with dialogue and moral value",
      "k_shot_count": 2,
      "messages": [
        {
          "role": "user",
          "content": "Generate simple, short (up to 150 words) bed time story..."
        },
        {
          "role": "assistant", 
          "content": "Benny blinked his sleepy eyes..."
        }
      ],
      "metadata": {
        "story_features": ["dialogue", "moral_value"],
        "complexity": "medium",
        "target_age": 3
      }
    }
  ]
}
```

### 2. Backward Compatibility Strategy

- Keep existing `parse_conversation_examples()` function
- Maintain support for `--conversation-examples-path`
- Convert text format to new internal structure
- Deprecation warnings for old format usage

### 3. Provider-Specific Optimizations

#### OpenAI Provider
- Direct chat format compatibility
- Optimized message structure
- Token usage optimization

#### Transformers Provider  
- Chat template integration
- Model-specific formatting
- Memory usage optimization

#### Mock Provider
- Simplified testing format
- Deterministic responses
- Debug information

## Migration Strategy

### For Existing Users

1. **Immediate**: Continue using existing text format with deprecation warnings
2. **Short-term**: Migrate to JSON format using provided conversion tools
3. **Long-term**: Adopt new command-line interface and strict validation

### Conversion Tools

Provide utility script to convert existing text files to JSON format:

```bash
python -m training.synthetic_data_generation.tools.convert_k_shot \
    --input config/example_conversation.txt \
    --output config/k_shot_config.json \
    --name "converted-examples"
```

## Risk Assessment and Mitigation

### Risks

1. **Breaking Changes**: New strict validation may break existing workflows
2. **Performance Impact**: JSON parsing overhead
3. **Complexity**: Additional configuration options may confuse users

### Mitigation Strategies

1. **Gradual Migration**: Maintain backward compatibility during transition
2. **Performance Testing**: Benchmark JSON vs text parsing performance
3. **Documentation**: Comprehensive examples and migration guides
4. **Testing**: Extensive testing across all providers and configurations

## Success Metrics

1. **Functionality**: All three goals achieved and tested
2. **Usability**: Simplified command-line interface with clear error messages
3. **Compatibility**: Works consistently across all providers
4. **Performance**: No significant performance degradation
5. **Adoption**: Existing users can migrate without major issues

## Conclusion

This improvement plan addresses the key limitations of the current k-shot prompting system while maintaining backward compatibility. The proposed changes will result in a more maintainable, flexible, and user-friendly system that supports the project's growth and evolution.

The phased implementation approach ensures minimal disruption to existing users while providing a clear path forward for enhanced functionality.
