# K-Shot Prompting Functionality Improvement Plan

## Executive Summary

This document outlines a comprehensive improvement plan for the k-shot prompting functionality in the synthetic data generation pipeline. The current implementation has **critical architectural flaws** that make k-shot prompting inconsistent across providers and broken for OpenAI-compatible APIs. This plan addresses three key goals: fixing the broken provider interface, enabling JSON-based k-shot samples, and ensuring consistent behavior across all providers.

## Current State Analysis

### Critical Architectural Problems

The k-shot prompting system has fundamental design flaws:

1. **Broken Provider Interface**: All providers use `generate_batch(prompts: List[str])` but k-shot requires conversation structure
2. **OpenAI Provider Completely Broken**: Receives pre-formatted strings and wraps them in single user messages, losing all k-shot context
3. **Inconsistent Provider Behavior**: Only TransformersProvider with tokenizer gets proper k-shot formatting
4. **Ugly Hacks in BatchProcessor**: Provider-specific logic with `hasattr(self.llm_provider, 'tokenizer')` checks
5. **Unused Code**: `generate_with_messages()` method exists but is never called

### Current Implementation Flow (BROKEN)

1. **PromptGenerator** creates `StoryPrompt` with `k_shot_examples` field
2. **BatchProcessor** has provider-specific hacks:
   - **TransformersProvider**: Uses `tokenizer.apply_chat_template()` to format messages
   - **OpenAI Provider**: Gets pre-formatted string, wraps in `{"role": "user", "content": prompt}` - **LOSES ALL K-SHOT CONTEXT**
   - **MockProvider**: Gets formatted string but ignores k-shot context
3. **All providers** receive `List[str]` instead of conversation structure

### Root Cause Analysis

The fundamental issue is the **wrong provider interface**:

```python
# CURRENT (BROKEN)
async def generate_batch(self, prompts: List[str], config: GenerationConfig) -> List[str]

# NEEDED
async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]
```

Where `LLMRequest` contains conversation structure that each provider can handle appropriately.

### Current Data Sources

1. **Text-based format** (`training/synthetic_data_generation/config/example_conversation.txt`)
   - Uses PROMPT:/RESPONSE: markers
   - Requires regex parsing
   - Difficult to maintain and extend

2. **JSON samples** (`docs/k_shot_prompting_samples.json`)
   - Well-structured format with metadata
   - Contains multiple example configurations
   - Currently not used by the pipeline

### Evidence of Broken Implementation

**OpenAI Provider** (`training/common/llm_providers.py` lines 340-382):
```python
async def _generate_single(self, client, prompt: str, config) -> str:
    payload = {
        "model": self.model_name,
        "messages": [{"role": "user", "content": prompt}]  # ← LOSES K-SHOT CONTEXT!
    }
```

**BatchProcessor Hack** (`training/synthetic_data_generation/src/batch_processor.py` lines 65-88):
```python
if prompt.k_shot_examples and hasattr(self.llm_provider, 'tokenizer') and self.llm_provider.tokenizer:
    # Only works for TransformersProvider!
```

## Improvement Goals

### Goal 1: Fix Broken Provider Interface (CRITICAL)

**Objective**: Replace `generate_batch(prompts: List[str])` with `generate_batch(requests: List[LLMRequest])` to properly support conversation structure.

**Benefits**:
- K-shot prompting actually works for all providers
- Clean separation of concerns
- No more provider-specific hacks in BatchProcessor
- Future-proof for additional features (tools, system messages, etc.)

### Goal 2: Enable JSON-based K-Shot Sources

**Objective**: Make it possible to use files like `docs/k_shot_prompting_samples.json` as a source for k-shot generation while keeping current parsing as an alternative.

**Benefits**:
- Structured, maintainable data format
- Rich metadata support
- Multiple configuration options in one file
- Better validation and error detection

### Goal 3: Simplify K-Shot Usage

**Objective**: Stop using graceful degradation principle - things should fail clearly if something is missing, with minimal required parameters.

**Benefits**:
- Clear error messages when configuration is wrong
- Reduced complexity in error handling
- More predictable behavior
- Easier debugging and troubleshooting

## Proposed Solution Architecture

### 1. New LLMRequest Interface (CRITICAL FIX)

#### 1.1 New Data Models

```python
class LLMMessage(BaseModel):
    """Individual message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str

class LLMRequest(BaseModel):
    """Request structure for LLM providers."""
    messages: List[LLMMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_story_prompt(cls, prompt: StoryPrompt) -> "LLMRequest":
        """Create LLMRequest from StoryPrompt."""
        messages = []

        # Add k-shot examples
        for example in prompt.k_shot_examples:
            messages.append(LLMMessage(role=example.role, content=example.content))

        # Add the actual prompt
        messages.append(LLMMessage(role="user", content=prompt.full_prompt))

        return cls(
            messages=messages,
            metadata={
                "prompt_id": prompt.prompt_id,
                "selected_words": prompt.selected_words,
                "k_shot_count": len(prompt.k_shot_examples)
            }
        )

    def to_simple_prompt(self) -> str:
        """Fallback: convert to simple string for providers that need it."""
        if len(self.messages) == 1 and self.messages[0].role == "user":
            return self.messages[0].content

        # Simple concatenation fallback
        parts = []
        for msg in self.messages:
            if msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        return "\n\n".join(parts)
```

#### 1.2 Updated Provider Interface

```python
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        """Generate responses for a batch of requests."""
        pass
```

### 2. Provider-Specific Implementations

#### 2.1 TransformersProvider

```python
class TransformersProvider(LLMProvider):
    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        # Convert LLMRequest to chat template format
        texts_batch = []
        for request in requests:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            if self.tokenizer.chat_template:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                # Fallback to simple concatenation
                text = request.to_simple_prompt()

            texts_batch.append(text)

        # Use existing tokenization and generation logic
        return await self._generate_from_texts(texts_batch, config)
```

#### 2.2 OpenAICompatibleProvider

```python
class OpenAICompatibleProvider(LLMProvider):
    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        responses = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for request in requests:
                # Convert LLMRequest directly to OpenAI messages format
                messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

                payload = {
                    "model": self.model_name,
                    "messages": messages,  # ← PROPER K-SHOT SUPPORT!
                    "max_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                }

                response = await self._make_api_call(client, payload)
                responses.append(response)

        return responses
```

#### 2.3 MockProvider

```python
class MockLLMProvider(LLMProvider):
    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        responses = []

        for request in requests:
            # Can analyze k-shot examples for better mock responses
            k_shot_context = [msg for msg in request.messages if msg.role == "assistant"]
            current_prompt = request.messages[-1].content

            # Generate mock response considering k-shot context
            response = self._generate_mock_with_context(current_prompt, k_shot_context)
            responses.append(response)

        return responses
```

### 3. Simplified BatchProcessor

```python
class BatchProcessor:
    async def process_batch(self, prompts: List[StoryPrompt]) -> GenerationResult:
        # Convert StoryPrompts to LLMRequests
        requests = [LLMRequest.from_story_prompt(prompt) for prompt in prompts]

        # No more provider-specific hacks!
        generated_texts = await self.llm_provider.generate_batch(requests, self.generation_config)

        # Process results as before
        return self._process_results(prompts, generated_texts)
```

### 4. JSON Configuration System

#### 4.1 Enhanced Data Models

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

#### 4.2 Unified K-Shot Loader

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

### 5. Updated Command-Line Interface

#### 5.1 New Parameters

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

## Implementation Plan

### Phase 1: Fix Provider Interface (Week 1) - CRITICAL

#### Task 1.1: Create New Data Models
- Add `LLMMessage` and `LLMRequest` models to `training/common/data_models.py`
- Add `from_story_prompt()` and `to_simple_prompt()` methods
- Update imports across the codebase

#### Task 1.2: Update Provider Interface
- Change `LLMProvider.generate_batch()` signature to use `List[LLMRequest]`
- Update abstract base class in `training/common/llm_providers.py`
- Ensure backward compatibility during transition

#### Task 1.3: Fix TransformersProvider
- Update `generate_batch()` to handle `LLMRequest` objects
- Use `apply_chat_template()` properly with message structure
- Remove the unused `generate_with_messages()` method

### Phase 2: Fix OpenAI and Mock Providers (Week 1-2)

#### Task 2.1: Fix OpenAICompatibleProvider (CRITICAL BUG FIX)
- Update `generate_batch()` to use conversation messages directly
- Fix `_generate_single()` to send proper message arrays to API
- Remove string-wrapping hack that loses k-shot context

#### Task 2.2: Update MockProvider
- Enhance mock generation to consider k-shot context
- Improve mock responses based on conversation history
- Add debugging capabilities for k-shot testing

#### Task 2.3: Simplify BatchProcessor
- Remove all provider-specific hacks and `hasattr()` checks
- Use clean `LLMRequest.from_story_prompt()` conversion
- Remove `_format_messages_simple()` fallback method

### Phase 3: JSON Configuration System (Week 2-3)

#### Task 3.1: Create KShotLoader
- Implement JSON loading functionality
- Maintain backward compatibility with text format
- Add configuration selection logic

#### Task 3.2: Update Command-Line Interface
- Add `--k-shot-config-file` and `--k-shot-config-name` parameters
- Add `--require-k-shot` for strict validation
- Update help documentation

#### Task 3.3: Integration Testing
- Test k-shot functionality with all three providers
- Verify consistent behavior across providers
- Test both JSON and text configuration sources

### Phase 4: Documentation and Cleanup (Week 3-4)

#### Task 4.1: Update Documentation
- Update DATA_GENERATION.md with corrected architecture
- Add examples showing k-shot working with all providers
- Document the LLMRequest interface

#### Task 4.2: Comprehensive Testing
- Unit tests for LLMRequest and provider implementations
- Integration tests verifying k-shot works consistently
- Regression tests to prevent future breakage

#### Task 4.3: Remove Dead Code
- Remove unused `generate_with_messages()` method
- Clean up provider-specific hacks in BatchProcessor
- Remove `format_for_chat_template()` if no longer needed

## Detailed Technical Specifications

### 1. LLMRequest Interface Specification

```python
class LLMMessage(BaseModel):
    """Individual message in a conversation."""
    role: str = Field(description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")

class LLMRequest(BaseModel):
    """Request structure for LLM providers."""
    messages: List[LLMMessage] = Field(description="Conversation messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")

    @classmethod
    def from_story_prompt(cls, prompt: StoryPrompt) -> "LLMRequest":
        """Create LLMRequest from StoryPrompt with k-shot examples."""
        messages = []

        # Add k-shot examples in order
        for example in prompt.k_shot_examples:
            messages.append(LLMMessage(role=example.role, content=example.content))

        # Add the current prompt as user message
        messages.append(LLMMessage(role="user", content=prompt.full_prompt))

        return cls(
            messages=messages,
            metadata={
                "prompt_id": prompt.prompt_id,
                "selected_words": prompt.selected_words,
                "additional_condition": prompt.additional_condition,
                "k_shot_count": len(prompt.k_shot_examples)
            }
        )

    def to_simple_prompt(self) -> str:
        """Fallback conversion to simple string for legacy support."""
        if len(self.messages) == 1 and self.messages[0].role == "user":
            return self.messages[0].content

        # Format as simple conversation
        parts = []
        for msg in self.messages:
            if msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            elif msg.role == "system":
                parts.append(f"System: {msg.content}")

        return "\n\n".join(parts)
```

### 2. Provider Implementation Requirements

#### All Providers Must:
- Accept `List[LLMRequest]` instead of `List[str]`
- Handle conversation structure appropriately for their backend
- Maintain consistent behavior for k-shot prompting
- Support empty k-shot (single user message) gracefully

#### TransformersProvider:
- Use `tokenizer.apply_chat_template()` when available
- Fall back to simple concatenation for models without chat templates
- Preserve existing batching and memory management

#### OpenAICompatibleProvider:
- Send `messages` array directly to API (fixes current bug)
- Support multi-turn conversations properly
- Handle rate limiting and error recovery

#### MockProvider:
- Analyze k-shot context for better mock responses
- Provide deterministic behavior for testing
- Support debugging and inspection of k-shot data

## Critical Bug Fixes Required

### 1. OpenAI Provider K-Shot Bug (CRITICAL)

**Current Code** (`training/common/llm_providers.py:340-382`):
```python
async def _generate_single(self, client, prompt: str, config) -> str:
    payload = {
        "model": self.model_name,
        "messages": [{"role": "user", "content": prompt}]  # ← LOSES ALL K-SHOT CONTEXT!
    }
```

**Fixed Code**:
```python
async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
    responses = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for request in requests:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            payload = {
                "model": self.model_name,
                "messages": messages,  # ← PROPER K-SHOT SUPPORT!
                "max_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            response = await self._make_api_call(client, payload)
            responses.append(response)
    return responses
```

### 2. BatchProcessor Hack Removal (CRITICAL)

**Current Code** (`training/synthetic_data_generation/src/batch_processor.py:65-88`):
```python
if prompt.k_shot_examples and hasattr(self.llm_provider, 'tokenizer') and self.llm_provider.tokenizer:
    # Provider-specific hack!
```

**Fixed Code**:
```python
async def process_batch(self, prompts: List[StoryPrompt]) -> GenerationResult:
    # Clean conversion - no provider-specific hacks!
    requests = [LLMRequest.from_story_prompt(prompt) for prompt in prompts]
    generated_texts = await self.llm_provider.generate_batch(requests, self.generation_config)
    return self._process_results(prompts, generated_texts)
```

## Migration Strategy

### Phase 1: Interface Migration (Breaking Change)
1. **Update all provider signatures** to use `List[LLMRequest]`
2. **Update BatchProcessor** to use new interface
3. **Test thoroughly** with all providers

### Phase 2: JSON Configuration (Additive)
1. **Add JSON support** while keeping text format
2. **Add new CLI parameters** for JSON configuration
3. **Maintain backward compatibility**

### Backward Compatibility
- Keep existing `--conversation-examples-path` parameter
- Support existing text format indefinitely
- Add deprecation warnings for old usage patterns

## Risk Assessment

### High Risk: Interface Changes
- **Risk**: Breaking existing code that directly uses providers
- **Mitigation**: Comprehensive testing and clear migration documentation

### Medium Risk: Provider Behavior Changes
- **Risk**: Different k-shot formatting across providers
- **Mitigation**: Standardized testing suite for all providers

### Low Risk: JSON Configuration
- **Risk**: Additional complexity
- **Mitigation**: Optional feature with good defaults

## Success Metrics

1. **K-shot works consistently** across all providers (OpenAI, Transformers, Mock)
2. **No provider-specific hacks** in BatchProcessor
3. **Clean, maintainable code** with proper separation of concerns
4. **Backward compatibility** maintained for existing users
5. **JSON configuration** provides enhanced flexibility

## Conclusion

The current k-shot implementation is **fundamentally broken** due to architectural flaws in the provider interface. The OpenAI provider completely loses k-shot context, and the system relies on ugly provider-specific hacks.

This improvement plan fixes these critical issues by:
1. **Introducing proper LLMRequest interface** that preserves conversation structure
2. **Fixing the OpenAI provider** to actually use k-shot examples
3. **Removing provider-specific hacks** from BatchProcessor
4. **Adding JSON configuration** for better maintainability

The fixes are essential for the system to work as intended and provide a foundation for future enhancements.
