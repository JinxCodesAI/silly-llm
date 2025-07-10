# Modular Prompt Template Architecture

## Executive Summary

This document outlines a comprehensive redesign of the current prompt generation system to create a flexible, extensible, and configurable template architecture. The current system has **critical architectural limitations** that make it impossible to experiment with different prompt strategies, template variations, or feature combinations without code changes. This plan addresses three key goals: creating a plugin-based prompt generator system, enabling simple JSON-based configuration, and providing extensibility through dedicated handlers while keeping simple scenarios code-free.

## Current State Analysis

### Critical Architectural Problems

The current prompt template system has fundamental design flaws that severely limit its flexibility and extensibility:

1. **Hardcoded Template**: The `TemplateManager` contains a single hardcoded template string that cannot be modified without code changes
2. **Monolithic Design**: No separation between template structure, feature logic, and prompt formatting
3. **Static Feature System**: Story features are simple strings with no metadata, relationships, or conditional logic
4. **No Template Variation**: Impossible to A/B test different prompt formats or strategies
5. **Tight Coupling**: Template logic is scattered across multiple components making changes risky
6. **Limited Configurability**: No way to configure templates, features, or prompt strategies through configuration files

### Current Implementation Flow (INFLEXIBLE)

1. **StoryGenerator** initializes with hardcoded paths:
   ```python
   # Initialize template manager
   self.template_manager = TemplateManager(
       story_features_path=self.story_features_path  # Only configurable part
   )
   ```

2. **TemplateManager** loads with hardcoded template:
   ```python
   self.base_template = (
       "Generate simple, short (up to 150 words) bed time story written entirely in English, easy to understand and follow by 3 years old who knows only English\n"
       "containing 3 English words {word1} {word2} {word3}\n\n"
       "{additional_condition}\n\n"
       "keep story coherent and gramatically correct, write full content of the story and nothing else (no commentary, title, etc)"
   )
   ```

3. **Story Features** are loaded as simple strings:
   ```python
   # From docs/story_features.json
   [
     "make sure the story contains a dialogue",
     "make sure story convey a clear moral value",
     "make sure story has sad elements but ends well",
     "make sure story do not have sad elements and ends well",
     "make sure story has scary elements but ends well"
   ]
   ```

4. **Prompt Creation** uses basic string formatting:
   ```python
   full_prompt = self.base_template.format(
       word1=selected_words.get("word1", ""),
       word2=selected_words.get("word2", ""),
       word3=selected_words.get("word3", ""),
       additional_condition=additional_condition
   )
   ```

### Root Cause Analysis

The fundamental issue is the **lack of abstraction and modularity**:

```python
# CURRENT (INFLEXIBLE)
class TemplateManager:
    def __init__(self, story_features_path: Optional[str] = None):
        self.base_template = "hardcoded template string..."  # ← HARDCODED!
        self.story_features = load_story_features(story_features_path)  # ← SIMPLE STRINGS!

# NEEDED (MODULAR)
class TemplateManager:
    def __init__(self, template_config: TemplateConfiguration):
        self.template_registry = TemplateRegistry(template_config)
        self.feature_manager = FeatureManager(template_config.features)
        self.strategy_selector = TemplateStrategySelector(template_config.strategies)
```

### Evidence of Inflexible Implementation

**Hardcoded Template** (`training/synthetic_data_generation/src/template_manager.py` lines 20-25):
```python
self.base_template = (
    "Generate simple, short (up to 150 words) bed time story written entirely in English, easy to understand and follow by 3 years old who knows only English\n"
    "containing 3 English words {word1} {word2} {word3}\n\n"
    "{additional_condition}\n\n"
    "keep story coherent and gramatically correct, write full content of the story and nothing else (no commentary, title, etc)"
)
```

**Simple Feature Selection** (`training/synthetic_data_generation/src/template_manager.py` lines 48-53):
```python
if additional_condition is None:
    if self.story_features:
        additional_condition = random.choice(self.story_features)  # ← RANDOM STRING SELECTION!
    else:
        additional_condition = ""
```

**No Template Variation** - The system can only use one template format, making it impossible to:
- Test different prompt structures
- Adapt templates for different age groups
- Use specialized templates for different story types
- Configure templates per deployment environment

## Improvement Goals

### Goal 1: Create Plugin-Based Prompt Generator System (CRITICAL)

**Objective**: Replace hardcoded template with a plugin-based system where each prompt generator is a self-contained handler with its own configuration schema and logic.

**Benefits**:
- Multiple prompt generators for different use cases
- A/B testing capabilities for prompt optimization
- Easy addition of new generators without modifying core code
- Clean separation between configuration and logic
- Environment-specific generator configurations

### Goal 2: Implement Simple JSON Configuration

**Objective**: Replace complex embedded logic with simple JSON configuration that references external handlers for complex scenarios.

**Benefits**:
- Clean, readable configuration files
- No business logic embedded in configuration
- Easy validation and schema enforcement
- Version control friendly
- Simple scenarios remain code-free

### Goal 3: Enable Extensibility Through Handlers

**Objective**: Provide a plugin architecture where complex logic is implemented in dedicated handler classes with strict interfaces.

**Benefits**:
- Complex scenarios handled by dedicated code
- Simple scenarios use built-in handlers
- Clear separation of concerns
- Easy testing and maintenance
- Extensible without core system changes

## Proposed Solution Architecture

### 1. Plugin-Based Prompt Generator System

#### 1.1 Core Configuration Format

```json
{
  "prompt_generation_system": {
    "name": "Modular Prompt Generation System",
    "version": "2.0",
    "description": "Plugin-based prompt generation with configurable handlers",

    "prompt_generators": {
      "basic_bedtime_story": {
        "name": "Basic Bedtime Story Generator",
        "description": "Simple bedtime stories for young children",
        "handler_class": "training.synthetic_data_generation.handlers.BasicBedtimeStoryHandler",
        "config": {
          "max_words": 150,
          "target_age": 3,
          "language": "English",
          "word_count": 3,
          "template": "Generate simple, short (up to {max_words} words) bed time story written entirely in {language}, easy to understand and follow by {target_age} years old who knows only {language}\ncontaining {word_count} {language} words {word1} {word2} {word3}\n\n{feature_instructions}\n\nkeep story coherent and gramatically correct, write full content of the story and nothing else (no commentary, title, etc)",
          "dynamic_placeholders": ["{word1}", "{word2}", "{word3}", "{feature_instructions}"],
          "artifacts": [
            {
              "name": "word_list",
              "type": "file",
              "value": "training/synthetic_data_generation/config/vocabulary.json"
            },
            {
              "name": "story_features",
              "type": "file",
              "value": "docs/story_features.json"
            }
          ]
        }
      },

      "advanced_bedtime_story": {
        "name": "Advanced Bedtime Story Generator",
        "description": "More complex stories for older children",
        "handler_class": "training.synthetic_data_generation.handlers.AdvancedBedtimeStoryHandler",
        "config": {
          "max_words": 200,
          "target_age": 6,
          "language": "English",
          "word_count": 3,
          "template": "Create an engaging bedtime story (up to {max_words} words) in {language} suitable for {target_age} year old children.\nThe story must include these {word_count} words: {word1}, {word2}, and {word3}\n\nStory requirements:\n{feature_instructions}\n\nGuidelines:\n- Use age-appropriate vocabulary and concepts\n- Include descriptive language to engage imagination\n- Ensure the story has a clear beginning, middle, and end\n- Write only the story content without titles or commentary",
          "dynamic_placeholders": ["{word1}", "{word2}", "{word3}", "{feature_instructions}"],
          "artifacts": [
            {
              "name": "word_list",
              "type": "file",
              "value": "training/synthetic_data_generation/config/vocabulary.json"
            },
            {
              "name": "story_features",
              "type": "file",
              "value": "docs/story_features.json"
            }
          ]
        }
      },

      "educational_story": {
        "name": "Educational Story Generator",
        "description": "Stories with educational content",
        "handler_class": "training.synthetic_data_generation.handlers.EducationalStoryHandler",
        "config": {
          "max_words": 180,
          "target_age": 5,
          "language": "English",
          "word_count": 3,
          "artifacts": [
            {
              "name": "word_list",
              "type": "file",
              "value": "training/synthetic_data_generation/config/vocabulary.json"
            },
            {
              "name": "educational_topics",
              "type": "file",
              "value": "training/synthetic_data_generation/config/educational_topics.json"
            }
          ]
        }
      }
    },

    "k_shot_system": {
      "enabled": true,
      "default_strategy": "conversation_examples",
      "strategies": {
        "conversation_examples": {
          "handler_class": "training.synthetic_data_generation.k_shot.ConversationExamplesHandler",
          "config": {
            "source_file": "training/synthetic_data_generation/config/example_conversation.txt",
            "format": "text",
            "default_count": 2
          }
        },
        "json_examples": {
          "handler_class": "training.synthetic_data_generation.k_shot.JsonExamplesHandler",
          "config": {
            "source_file": "docs/k_shot_prompting_samples.json",
            "format": "json",
            "default_count": 3
          }
        }
      }
    }
  }
}
```

#### 1.2 Handler Interface Specification

The handler interface is the core abstraction that enables the plugin-based architecture. Each prompt generator is implemented as a handler class that follows a strict interface, allowing the system to dynamically load and execute different prompt generation strategies without modifying the core system.

**Purpose**: The handler interface provides a standardized way to implement prompt generation logic while maintaining complete flexibility in how that logic is implemented. It separates the concerns of configuration management (handled by the registry) from prompt generation logic (handled by individual handlers).

**Key Responsibilities**:
- Load and validate configuration data and artifacts
- Generate dynamic placeholders based on context and artifacts
- Create formatted prompts from templates and placeholders
- Validate configuration at both startup and runtime
- Provide extensible validation for custom requirements

**Relationship to System**: Handlers are instantiated by the PromptGeneratorRegistry and called by the ModularTemplateManager. They act as the bridge between static configuration and dynamic prompt generation, with the k-shot system operating independently on top of the generated prompts.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from training.common.data_models import StoryPrompt

class PromptGeneratorHandler(ABC):
    """Abstract base class for prompt generator handlers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize handler with configuration."""
        self.config = config
        self.artifacts = self._load_artifacts(config.get("artifacts", []))
        self.dynamic_placeholders = config.get("dynamic_placeholders", [])

        # Validate configuration at initialization
        if not self.validate_config():
            raise ValueError(f"Invalid configuration for {self.__class__.__name__}")

    @abstractmethod
    def generate_prompt(self, context: Dict[str, Any]) -> StoryPrompt:
        """Generate a story prompt based on configuration and context.

        Args:
            context: Complete generation context containing:
                    - 'config': Handler configuration (merged with any overrides)
                    - 'artifacts': Loaded artifact data (word_list, story_features, etc.)
                    - Any additional context data (prompt_id, target_age, etc.)

        Returns:
            StoryPrompt object
        """
        pass

    @abstractmethod
    def generate_dynamic_placeholders(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate dynamic placeholder values based on context.

        Args:
            context: Complete generation context with config and artifacts

        Returns:
            Dictionary mapping placeholder names to their values
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the handler configuration at initialization."""
        pass

    def validate_runtime(self, context: Dict[str, Any]) -> bool:
        """Validate runtime context and generated placeholders."""
        # Base validation - check required context structure
        if 'config' not in context or 'artifacts' not in context:
            return False

        # Validate that dynamic placeholders can be generated
        try:
            placeholders = self.generate_dynamic_placeholders(context)
            # Check that all dynamic placeholders are covered
            config = context['config']
            for placeholder in self.dynamic_placeholders:
                placeholder_key = placeholder.strip('{}')
                if placeholder_key not in placeholders and placeholder_key not in config:
                    return False
            return True
        except Exception:
            return False

    def _load_artifacts(self, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load artifacts specified in configuration."""
        loaded_artifacts = {}
        for artifact in artifacts:
            name = artifact["name"]
            artifact_type = artifact["type"]
            value = artifact["value"]

            if artifact_type == "file":
                loaded_artifacts[name] = self._load_file_artifact(value)
            elif artifact_type == "array":
                # Parse array if it's a JSON string, otherwise use as-is
                if isinstance(value, str):
                    try:
                        loaded_artifacts[name] = json.loads(value)
                    except json.JSONDecodeError:
                        loaded_artifacts[name] = [value]  # Single string as array
                else:
                    loaded_artifacts[name] = value
            elif artifact_type == "string":
                loaded_artifacts[name] = str(value)
            elif artifact_type == "handler":
                loaded_artifacts[name] = self._load_handler_artifact(value)
            else:
                raise ValueError(f"Unknown artifact type: {artifact_type}")

        return loaded_artifacts

    def _load_file_artifact(self, file_path: str) -> Any:
        """Load artifact from file."""
        import json
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact file not found: {file_path}")

        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            # Load as text file
            with open(file_path, 'r') as f:
                return f.read().strip()

    def _load_handler_artifact(self, handler_class_path: str) -> Any:
        """Load and instantiate handler class."""
        module_path, class_name = handler_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        handler_class = getattr(module, class_name)
        return handler_class()

class BasicBedtimeStoryHandler(PromptGeneratorHandler):
    """Handler for basic bedtime stories - simple, code-free implementation."""

    def generate_prompt(self, context: Dict[str, Any]) -> StoryPrompt:
        """Generate basic bedtime story prompt."""
        # Validate runtime context
        if not self.validate_runtime(context):
            raise ValueError("Invalid runtime context for BasicBedtimeStoryHandler")

        config = context["config"]

        # Generate dynamic placeholders
        dynamic_values = self.generate_dynamic_placeholders(context)

        # Combine config and dynamic values for template formatting
        template_values = {**config, **dynamic_values}

        # Format template
        template = config["template"]
        full_prompt = template.format(**template_values)

        return StoryPrompt(
            prompt_id=context.get("prompt_id", f"prompt_{random.randint(100000, 999999)}"),
            template=template,
            selected_words=dynamic_values.get("selected_words", {}),
            additional_condition=dynamic_values.get("feature_instructions", ""),
            full_prompt=full_prompt,
            metadata={
                "generator": "basic_bedtime_story",
                "target_age": config.get("target_age"),
                "max_words": config.get("max_words")
            }
        )

    def generate_dynamic_placeholders(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate dynamic placeholder values."""
        artifacts = context["artifacts"]
        placeholders = {}

        # Extract words from context if available (this is just one possible use case)
        if "selected_words" in context:
            selected_words = context["selected_words"]
            placeholders.update({
                "word1": selected_words.get("word1", ""),
                "word2": selected_words.get("word2", ""),
                "word3": selected_words.get("word3", "")
            })

        # Simple random selection from story features
        story_features = artifacts.get("story_features", [])
        if story_features:
            placeholders["feature_instructions"] = random.choice(story_features)
        else:
            placeholders["feature_instructions"] = ""

        return placeholders

    def validate_config(self) -> bool:
        """Validate basic configuration."""
        required_fields = ["template"]
        has_required = all(field in self.config for field in required_fields)

        # Validate that template placeholders match dynamic_placeholders + config
        if has_required and "template" in self.config:
            template = self.config["template"]
            # Extract placeholders from template
            import re
            template_placeholders = set(re.findall(r'\{(\w+)\}', template))

            # Check that all template placeholders are covered
            config_keys = set(self.config.keys())
            dynamic_keys = set(p.strip('{}') for p in self.dynamic_placeholders)
            available_keys = config_keys | dynamic_keys

            uncovered = template_placeholders - available_keys
            if uncovered:
                print(f"Warning: Template placeholders not covered: {uncovered}")
                return False

        return has_required

class AdvancedBedtimeStoryHandler(PromptGeneratorHandler):
    """Handler for advanced bedtime stories with complex age-based logic."""

    def generate_prompt(self, context: Dict[str, Any]) -> StoryPrompt:
        """Generate advanced bedtime story prompt with complex logic."""
        # Validate runtime context
        if not self.validate_runtime(context):
            raise ValueError("Invalid runtime context for AdvancedBedtimeStoryHandler")

        config = context["config"]
        target_age = context.get("target_age", config.get("target_age", 3))

        # Generate dynamic placeholders with age-based logic
        dynamic_values = self.generate_dynamic_placeholders(context)

        # Apply age-based template modifications
        template = self._get_age_appropriate_template(target_age, config)

        # Combine config and dynamic values for template formatting
        template_values = {**config, **dynamic_values, "target_age": target_age}

        # Format template
        full_prompt = template.format(**template_values)

        return StoryPrompt(
            prompt_id=context.get("prompt_id", f"prompt_{random.randint(100000, 999999)}"),
            template=template,
            selected_words=dynamic_values.get("selected_words", {}),
            additional_condition=dynamic_values.get("feature_instructions", ""),
            full_prompt=full_prompt,
            metadata={
                "generator": "advanced_bedtime_story",
                "target_age": target_age,
                "max_words": config.get("max_words"),
                "feature_complexity": "advanced"
            }
        )

    def generate_dynamic_placeholders(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate dynamic placeholders with age-based feature selection."""
        config = context["config"]
        artifacts = context["artifacts"]
        target_age = context.get("target_age", config.get("target_age", 3))

        placeholders = {}

        # Extract words from context if available
        if "selected_words" in context:
            selected_words = context["selected_words"]
            placeholders.update({
                "word1": selected_words.get("word1", ""),
                "word2": selected_words.get("word2", ""),
                "word3": selected_words.get("word3", "")
            })

        # Age-based feature selection from story features
        story_features = artifacts.get("story_features", [])
        if story_features:
            # Filter features based on age appropriateness
            if target_age < 5:
                # Simple features for younger children
                simple_features = [f for f in story_features if "scary" not in f.lower()]
                placeholders["feature_instructions"] = random.choice(simple_features) if simple_features else ""
            else:
                # All features available for older children
                placeholders["feature_instructions"] = random.choice(story_features)
        else:
            placeholders["feature_instructions"] = ""

        return placeholders

    def _get_age_appropriate_template(self, age: int, config: Dict[str, Any]) -> str:
        """Select template based on age - this is handler-specific logic."""
        base_template = config["template"]

        if age < 5:
            # Simpler language for younger children
            return base_template.replace("engaging", "simple").replace("descriptive language", "easy words")
        elif age > 8:
            # More complex for older children
            return base_template + "\n- Include some challenging vocabulary with context clues"

        return base_template

    def validate_config(self) -> bool:
        """Validate advanced configuration."""
        required_fields = ["template", "max_words", "target_age", "language", "word_count"]
        has_required = all(field in self.config for field in required_fields)

        # Validate template placeholders like in BasicBedtimeStoryHandler
        if has_required and "template" in self.config:
            template = self.config["template"]
            import re
            template_placeholders = set(re.findall(r'\{(\w+)\}', template))

            config_keys = set(self.config.keys())
            dynamic_keys = set(p.strip('{}') for p in self.dynamic_placeholders)
            available_keys = config_keys | dynamic_keys

            uncovered = template_placeholders - available_keys
            if uncovered:
                print(f"Warning: Template placeholders not covered: {uncovered}")
                return False

        return has_required
```

### 2. New Architecture Components

#### 2.1 Prompt Generator Registry

The Prompt Generator Registry is the central component responsible for loading, managing, and providing access to all available prompt generators. It acts as a factory and registry pattern implementation that handles the dynamic loading of handler classes and their configuration.

**Purpose**: The registry abstracts the complexity of handler instantiation and configuration management from the rest of the system. It provides a clean interface for accessing prompt generators while handling the technical details of dynamic class loading, configuration validation, and error handling.

**Key Responsibilities**:
- Load prompt generator configurations from JSON files
- Dynamically instantiate handler classes using importlib
- Validate all generator configurations at startup
- Provide access to generators by ID
- Handle errors gracefully during handler loading

**Relationship to System**: The registry is used by the ModularTemplateManager to access prompt generators. It sits between the configuration files and the actual handler implementations, providing a stable interface that doesn't change even when new handlers are added.

```python
class PromptGeneratorDefinition(BaseModel):
    """Definition of a prompt generator."""
    name: str
    description: str
    handler_class: str
    config: Dict[str, Any]

class PromptGeneratorRegistry:
    """Registry for managing prompt generators."""

    def __init__(self, config_path: str):
        self.config = self._load_configuration(config_path)
        self.generators = {}
        self._initialize_generators()

    def _load_configuration(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _initialize_generators(self):
        """Initialize all prompt generators from configuration."""
        for generator_id, generator_config in self.config["prompt_generation_system"]["prompt_generators"].items():
            handler_class = self._load_handler_class(generator_config["handler_class"])
            self.generators[generator_id] = handler_class(generator_config["config"])

    def _load_handler_class(self, class_path: str) -> type:
        """Dynamically load handler class."""
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def get_generator(self, generator_id: str) -> PromptGeneratorHandler:
        """Get prompt generator by ID."""
        if generator_id not in self.generators:
            raise ValueError(f"Prompt generator '{generator_id}' not found")
        return self.generators[generator_id]

    def list_generators(self) -> List[str]:
        """List available generator IDs."""
        return list(self.generators.keys())

    def validate_all_generators(self) -> Dict[str, bool]:
        """Validate all generator configurations."""
        results = {}
        for generator_id, generator in self.generators.items():
            try:
                results[generator_id] = generator.validate_config()
            except Exception as e:
                results[generator_id] = False
        return results
```

#### 2.2 K-Shot System Integration

The K-Shot System operates independently from prompt generation, providing a clean separation of concerns. It manages the loading, selection, and formatting of example conversations that are added to prompts after they are generated by the prompt handlers.

**Purpose**: The K-Shot system provides a pluggable architecture for managing different types of example data (text files, JSON files, etc.) and different selection strategies (random, weighted, context-aware). It operates as a separate layer that can be enabled or disabled without affecting prompt generation.

**Key Responsibilities**:
- Load k-shot examples from various sources (text, JSON, etc.)
- Implement different selection strategies for examples
- Format examples according to the expected conversation format
- Provide context-aware example selection when needed

**Relationship to System**: The K-Shot system is used by the ModularTemplateManager after prompt generation is complete. It receives the generated prompt and context, then adds appropriate k-shot examples. This separation allows prompt generators to focus solely on prompt creation while k-shot logic remains independent and reusable.

```python
class KShotHandler(ABC):
    """Abstract base class for K-shot example handlers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_k_shot_examples(self, context: Dict[str, Any]) -> List[KShotExample]:
        """Get K-shot examples based on context."""
        pass

class ConversationExamplesHandler(KShotHandler):
    """Handler for text-based conversation examples."""

    def get_k_shot_examples(self, context: Dict[str, Any]) -> List[KShotExample]:
        """Load examples from text file."""
        source_file = self.config["source_file"]
        count = context.get("k_shot_count", self.config["default_count"])

        # Load and parse conversation examples
        examples = parse_conversation_examples(source_file)

        # Select random examples
        if len(examples) >= count:
            selected = random.sample(examples, count)
        else:
            selected = examples

        # Flatten to individual messages
        k_shot_examples = []
        for conversation in selected:
            k_shot_examples.extend(conversation.messages)

        return k_shot_examples

class JsonExamplesHandler(KShotHandler):
    """Handler for JSON-based K-shot examples."""

    def get_k_shot_examples(self, context: Dict[str, Any]) -> List[KShotExample]:
        """Load examples from JSON file."""
        source_file = self.config["source_file"]
        count = context.get("k_shot_count", self.config["default_count"])

        with open(source_file, 'r') as f:
            data = json.load(f)

        # Select appropriate configuration
        config_name = context.get("k_shot_config", "default")
        examples_config = self._find_examples_config(data, config_name)

        # Extract messages
        messages = examples_config.get("messages", [])
        return [KShotExample(role=msg["role"], content=msg["content"]) for msg in messages[:count]]

class KShotSystem:
    """Manages K-shot example generation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.handlers = {}
        self._initialize_handlers()

    def _initialize_handlers(self):
        """Initialize K-shot handlers."""
        for strategy_name, strategy_config in self.config["strategies"].items():
            handler_class = self._load_handler_class(strategy_config["handler_class"])
            self.handlers[strategy_name] = handler_class(strategy_config["config"])

    def get_k_shot_examples(self, strategy: str, context: Dict[str, Any]) -> List[KShotExample]:
        """Get K-shot examples using specified strategy."""
        if strategy not in self.handlers:
            strategy = self.config["default_strategy"]

        handler = self.handlers[strategy]
        return handler.get_k_shot_examples(context)
```

#### 2.3 Modular Template Manager

The Modular Template Manager is the main orchestrator that brings together the prompt generator registry and k-shot system to provide a unified interface for prompt creation. It replaces the current TemplateManager with a more flexible, configurable system.

**Purpose**: The manager provides a simple, stable interface for prompt generation while internally coordinating between different subsystems. It handles the flow from generator selection through prompt creation to k-shot example addition, ensuring that all components work together seamlessly.

**Key Responsibilities**:
- Coordinate between prompt generators and k-shot system
- Provide a simple interface for prompt creation
- Handle context preparation and validation
- Manage the overall prompt generation workflow
- Provide system-wide validation and information methods

**Relationship to System**: The manager is used by the StoryGenerator and PromptGenerator classes, replacing the current TemplateManager. It provides the same high-level interface but with much more flexibility and configurability underneath.

```python
class ModularTemplateManager:
    """New modular template manager using plugin architecture."""

    def __init__(self, config_path: str):
        """Initialize with configuration file path."""
        self.config_path = config_path
        self.generator_registry = PromptGeneratorRegistry(config_path)

        # Initialize K-shot system if enabled
        config = self._load_config()
        k_shot_config = config["prompt_generation_system"].get("k_shot_system", {})
        if k_shot_config.get("enabled", False):
            self.k_shot_system = KShotSystem(k_shot_config)
        else:
            self.k_shot_system = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def create_prompt(self,
                     generator_id: str,
                     context: Optional[Dict[str, Any]] = None,
                     config_overrides: Optional[Dict[str, Any]] = None,
                     prompt_id: Optional[str] = None) -> StoryPrompt:
        """Create a story prompt using specified generator.

        Args:
            generator_id: ID of the prompt generator to use
            context: Generation context (may include selected_words, target_age, etc.)
            config_overrides: Override generator configuration
            prompt_id: Optional prompt identifier

        Returns:
            StoryPrompt object
        """
        if prompt_id is None:
            prompt_id = f"prompt_{random.randint(100000, 999999)}"

        # Get generator
        generator = self.generator_registry.get_generator(generator_id)

        # Prepare complete context
        complete_context = context or {}
        complete_context["prompt_id"] = prompt_id

        # Merge config with overrides
        base_config = generator.config.copy()
        if config_overrides:
            base_config.update(config_overrides)
        complete_context["config"] = base_config

        # Add artifacts to context
        complete_context["artifacts"] = generator.artifacts

        return generator.generate_prompt(complete_context)

    def create_k_shot_prompt(self,
                           generator_id: str,
                           k_shot_strategy: str = "conversation_examples",
                           context: Optional[Dict[str, Any]] = None,
                           config_overrides: Optional[Dict[str, Any]] = None,
                           prompt_id: Optional[str] = None) -> StoryPrompt:
        """Create a k-shot prompt with examples."""
        # Create base prompt
        prompt = self.create_prompt(generator_id, context, config_overrides, prompt_id)

        # Add K-shot examples if system is available
        if self.k_shot_system:
            k_shot_context = context or {}
            k_shot_examples = self.k_shot_system.get_k_shot_examples(k_shot_strategy, k_shot_context)
            prompt.k_shot_examples = k_shot_examples
            prompt.metadata["k_shot_count"] = len(k_shot_examples)
            prompt.metadata["k_shot_strategy"] = k_shot_strategy

        return prompt

    def get_available_generators(self) -> List[str]:
        """Get list of available generator IDs."""
        return self.generator_registry.list_generators()

    def get_generator_info(self, generator_id: str) -> Dict[str, Any]:
        """Get information about a specific generator."""
        generator = self.generator_registry.get_generator(generator_id)
        return {
            "config": generator.config,
            "artifacts": list(generator.artifacts.keys()),
            "valid": generator.validate_config()
        }

    def validate_system(self) -> Dict[str, Any]:
        """Validate the entire system configuration."""
        return {
            "generators": self.generator_registry.validate_all_generators(),
            "k_shot_enabled": self.k_shot_system is not None,
            "config_path": self.config_path
        }
```

#### 2.4 New Modular Template Manager

```python
class ModularTemplateManager:
    """New modular template manager with full configurability."""

    def __init__(self,
                 template_config_path: str,
                 feature_config_path: str,
                 strategy_config: Optional[Dict[str, Any]] = None):
        """Initialize modular template manager.

        Args:
            template_config_path: Path to template configuration YAML
            feature_config_path: Path to feature configuration YAML
            strategy_config: Template strategy configuration
        """
        self.template_registry = TemplateRegistry(template_config_path)
        self.feature_manager = FeatureManager(feature_config_path)
        self.strategy_selector = TemplateStrategySelector(strategy_config or {})

    def create_prompt(self,
                     selected_words: Dict[str, str],
                     context: Optional[Dict[str, Any]] = None,
                     template_name: Optional[str] = None,
                     features: Optional[List[str]] = None,
                     prompt_id: Optional[str] = None) -> StoryPrompt:
        """Create a story prompt using modular system.

        Args:
            selected_words: Dictionary with word1, word2, word3
            context: Generation context (age, complexity, etc.)
            template_name: Specific template to use (optional)
            features: Specific features to include (optional)
            prompt_id: Optional prompt identifier

        Returns:
            StoryPrompt object
        """
        if prompt_id is None:
            prompt_id = f"prompt_{random.randint(100000, 999999)}"

        # Prepare context
        context = context or {}
        context.update({
            "selected_words": selected_words,
            "prompt_id": prompt_id
        })

        # Select features if not provided
        if features is None:
            if template_name:
                template = self.template_registry.get_template(template_name)
                features = self.feature_manager.select_features(template, context)
            else:
                # Select features first, then compatible template
                features = self.feature_manager.select_features(None, context)

        # Select template if not provided
        if template_name is None:
            template_name = self.strategy_selector.select_template(
                self.template_registry, features, context
            )

        template = self.template_registry.get_template(template_name)

        # Validate feature compatibility
        if not self.feature_manager.validate_feature_compatibility(features):
            raise ValueError(f"Incompatible features selected: {features}")

        # Generate feature instructions
        feature_instructions = self.feature_manager.generate_feature_instructions(features, context)

        # Prepare template parameters
        template_params = {
            **template.parameters,
            **self.template_registry.config.global_parameters,
            **selected_words,
            "feature_instructions": feature_instructions
        }

        # Format the template
        full_prompt = template.template.format(**template_params)

        return StoryPrompt(
            prompt_id=prompt_id,
            template=template.template,
            selected_words=selected_words,
            additional_condition=feature_instructions,
            full_prompt=full_prompt,
            metadata={
                "template_name": template_name,
                "template_version": template.metadata.get("template_version", "2.0"),
                "features": features,
                "feature_count": len(features),
                "context": context,
                "template_complexity": template.metadata.get("complexity_level", 0.5)
            }
        )

    def create_k_shot_prompt(self,
                           selected_words: Dict[str, str],
                           k_shot_examples: List[KShotExample],
                           context: Optional[Dict[str, Any]] = None,
                           template_name: Optional[str] = None,
                           features: Optional[List[str]] = None,
                           prompt_id: Optional[str] = None) -> StoryPrompt:
        """Create a k-shot prompt with examples using modular system."""
        # Create base prompt
        prompt = self.create_prompt(
            selected_words=selected_words,
            context=context,
            template_name=template_name,
            features=features,
            prompt_id=prompt_id
        )

        # Add k-shot examples
        prompt.k_shot_examples = k_shot_examples
        prompt.metadata["k_shot_count"] = len(k_shot_examples)
        prompt.metadata["k_shot_strategy"] = self.template_registry.get_template(
            prompt.metadata["template_name"]
        ).k_shot_strategy

        return prompt

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return self.template_registry.list_templates()

    def get_available_features(self) -> List[str]:
        """Get list of available feature names."""
        return list(self.feature_manager.features.keys())

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get detailed information about a template."""
        template = self.template_registry.get_template(template_name)
        return {
            "name": template.name,
            "description": template.description,
            "compatible_features": template.compatible_features,
            "incompatible_features": template.incompatible_features,
            "metadata": template.metadata,
            "parameters": template.parameters
        }

    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get detailed information about a feature."""
        feature = self.feature_manager.features[feature_name]
        return {
            "name": feature.name,
            "description": feature.description,
            "category": feature.category,
            "compatibility": feature.compatibility,
            "constraints": feature.constraints,
            "metadata": feature.metadata
        }
```

### 3. Configuration System Integration

#### 3.1 Enhanced Configuration Format

```json
{
  "story_generation_config": {
    "model_name": "Qwen/Qwen3-0.6B",
    "device": "auto",
    "generation": {
      "batch_size": 8,
      "max_new_tokens": 512,
      "temperature": 0.8,
      "top_p": 0.9,
      "do_sample": true,
      "repetition_penalty": 1.1,
      "use_cache": true
    },
    "data_paths": {
      "vocabulary_path": "training/synthetic_data_generation/config/vocabulary.json"
    },
    "generation_settings": {
      "num_stories": 1000,
      "k_shot_count": 2,
      "use_k_shot": true,
      "ensure_diversity": true,
      "prompt_generator": "basic_bedtime_story",
      "generator_overrides": {
        "config": {
          "max_words": 200,
          "target_age": 5
        }
      }
    },
    "output_settings": {
      "output_path": "generated_stories.jsonl",
      "save_intermediate": true,
      "intermediate_save_interval": 100
    },
    "validation_settings": {
      "validate_stories": true,
      "min_words": 50,
      "max_words": 300
    },
    "logging": {
      "log_level": "INFO"
    },
    "prompt_system": {
      "config_file": "config/prompt_generators.json",
      "default_generator": "basic_bedtime_story"
    }
  }
}
```

#### 3.2 Updated Configuration Models

```python
class PromptSystemConfig(BaseModel):
    """Configuration for the prompt generation system."""
    config_file: str = Field(description="Path to prompt generators configuration")
    default_generator: str = Field(description="Default prompt generator to use")

class ModularStoryGenerationConfig(BaseModel):
    """Enhanced configuration with prompt system support."""

    # Existing fields
    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    device: str = Field(default="auto")
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    data_paths: DataPaths = Field(description="Data file paths")
    generation_settings: GenerationSettings = Field(default_factory=GenerationSettings)
    output_settings: OutputSettings = Field(description="Output settings")
    validation_settings: ValidationSettings = Field(default_factory=ValidationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # New prompt system configuration
    prompt_system: PromptSystemConfig = Field(description="Prompt generation system configuration")

class GenerationSettings(BaseModel):
    """Generation behavior settings with prompt generator selection."""
    num_stories: int = Field(default=1000, description="Number of stories to generate")
    k_shot_count: int = Field(default=2, description="Number of k-shot examples")
    use_k_shot: bool = Field(default=True, description="Whether to use k-shot examples")
    ensure_diversity: bool = Field(default=True, description="Ensure word diversity across prompts")
    prompt_generator: str = Field(default="basic_bedtime_story", description="Prompt generator to use")
    generator_overrides: Dict[str, Any] = Field(default_factory=dict, description="Override generator configuration")
```

### 4. Migration Strategy and Implementation Plan

#### Phase 1: Core Infrastructure (Week 1-2) - FOUNDATION

##### Task 1.1: Create Handler Interface and Base Classes
- **File**: `training/common/prompt_handlers.py`
- **Content**: Abstract PromptGeneratorHandler and base implementations
- **Dependencies**: Pydantic, existing data models
- **Testing**: Unit tests for handler interface and basic implementations

```python
# New file: training/common/prompt_handlers.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import importlib
import json
import random

# Handler interface and basic implementations as defined above
```

##### Task 1.2: Implement Prompt Generator Registry
- **File**: `training/synthetic_data_generation/src/prompt_generator_registry.py`
- **Content**: PromptGeneratorRegistry class with JSON loading and dynamic handler instantiation
- **Dependencies**: Handler interface, importlib
- **Testing**: Test registry loading, handler instantiation, validation

##### Task 1.3: Create Basic Handler Implementations
- **File**: `training/synthetic_data_generation/handlers/basic_handlers.py`
- **Content**: BasicBedtimeStoryHandler and other simple handlers
- **Dependencies**: Handler interface, existing utilities
- **Testing**: Test basic prompt generation with simple configurations

#### Phase 2: K-Shot System Integration (Week 2-3) - K-SHOT HANDLERS

##### Task 2.1: Create K-Shot Handler Framework
- **File**: `training/synthetic_data_generation/k_shot/k_shot_handlers.py`
- **Content**: Abstract KShotHandler and concrete implementations
- **Dependencies**: Existing k-shot utilities, handler interface
- **Testing**: Test k-shot example loading and selection

##### Task 2.2: Implement K-Shot System
- **File**: `training/synthetic_data_generation/k_shot/k_shot_system.py`
- **Content**: KShotSystem class for managing k-shot strategies
- **Dependencies**: K-shot handlers
- **Testing**: Test k-shot strategy selection and example generation

##### Task 2.3: Create Modular Template Manager
- **File**: `training/synthetic_data_generation/src/modular_template_manager.py`
- **Content**: ModularTemplateManager class (replacement for TemplateManager)
- **Dependencies**: Registry, k-shot system
- **Testing**: End-to-end prompt generation testing

#### Phase 3: Configuration Integration (Week 3-4) - CONFIGURATION

##### Task 3.1: Create Default Configurations
- **Files**:
  - `config/prompt_generators.json`
  - `config/story_generation_config.json`
- **Content**: Complete default configurations with handler references
- **Testing**: Configuration loading and validation

##### Task 3.2: Update Configuration System
- **File**: `training/synthetic_data_generation/src/config.py`
- **Content**: Enhanced configuration models and loading
- **Dependencies**: New prompt system models
- **Testing**: Configuration parsing and validation

##### Task 3.3: Create Advanced Handler Examples
- **File**: `training/synthetic_data_generation/handlers/advanced_handlers.py`
- **Content**: Complex handlers with custom logic (AdvancedBedtimeStoryHandler, EducationalStoryHandler)
- **Dependencies**: Handler interface, custom feature selectors
- **Testing**: Test complex prompt generation scenarios

#### Phase 4: Integration and Migration (Week 4-5) - INTEGRATION

##### Task 4.1: Update StoryGenerator
- **File**: `training/synthetic_data_generation/src/story_generator.py`
- **Content**: Integration with ModularTemplateManager
- **Dependencies**: Modular template system
- **Testing**: Full generation pipeline testing

##### Task 4.2: Create Configuration Migration Tool
- **File**: `training/synthetic_data_generation/tools/migrate_config.py`
- **Content**: Tool to convert existing configurations to new format
- **Dependencies**: Old and new configuration systems
- **Testing**: Migration of existing config files

##### Task 4.3: Update Command-Line Interface
- **File**: `training/synthetic_data_generation/main.py`
- **Content**: New CLI parameters for prompt generator selection
- **Dependencies**: Enhanced configuration system
- **Testing**: CLI parameter validation and usage

#### Phase 5: Testing and Documentation (Week 5-6) - VALIDATION

##### Task 5.1: Comprehensive Testing
- **Files**: `training/synthetic_data_generation/tests/test_modular_prompt_system.py`
- **Content**: Complete test suite for modular system
- **Dependencies**: All new components
- **Testing**: Unit, integration, and end-to-end tests

##### Task 5.2: Performance Testing
- **Files**: `training/synthetic_data_generation/tests/test_prompt_performance.py`
- **Content**: Performance comparison with old system
- **Dependencies**: Both old and new systems
- **Testing**: Benchmark generation speed and memory usage

##### Task 5.3: Update Documentation
- **Files**:
  - `docs/PROMPT_SYSTEM.md`
  - `docs/HANDLER_DEVELOPMENT.md`
  - Updated `docs/DATA_GENERATION.md`
- **Content**: Complete documentation for new system
- **Dependencies**: Working modular system
- **Testing**: Documentation examples should work

### 5. Detailed Technical Specifications

#### 5.1 Template Configuration File Format

**File Structure**:
```
config/
├── templates/
│   ├── bedtime_stories.yaml      # Main template definitions
│   ├── educational_stories.yaml  # Educational story templates
│   └── adventure_stories.yaml    # Adventure story templates
├── features/
│   ├── story_features.yaml       # Core story features
│   ├── educational_features.yaml # Educational features
│   └── age_specific_features.yaml # Age-specific features
├── strategies/
│   ├── template_weights.yaml     # Template selection weights
│   └── feature_weights.yaml      # Feature selection weights
└── modular_template_config.yaml  # Main configuration
```

**Template Definition Schema**:
```yaml
# JSON Schema for template validation
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "template_configuration": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "description": {"type": "string"},
        "global_parameters": {"type": "object"},
        "templates": {
          "type": "object",
          "patternProperties": {
            "^[a-zA-Z_][a-zA-Z0-9_]*$": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "template": {"type": "string"},
                "parameters": {"type": "object"},
                "required_placeholders": {
                  "type": "array",
                  "items": {"type": "string"}
                },
                "optional_placeholders": {
                  "type": "array",
                  "items": {"type": "string"}
                },
                "compatible_features": {
                  "type": "array",
                  "items": {"type": "string"}
                },
                "incompatible_features": {
                  "type": "array",
                  "items": {"type": "string"}
                },
                "k_shot_strategy": {"type": "string"},
                "k_shot_count": {"type": "integer", "minimum": 0},
                "metadata": {"type": "object"}
              },
              "required": ["name", "description", "template", "required_placeholders"]
            }
          }
        }
      },
      "required": ["name", "version", "templates"]
    }
  },
  "required": ["template_configuration"]
}
```

#### 5.2 Feature Configuration File Format

**Feature Definition Schema**:
```yaml
# JSON Schema for feature validation
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "feature_configuration": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "description": {"type": "string"},
        "features": {
          "type": "object",
          "patternProperties": {
            "^[a-zA-Z_][a-zA-Z0-9_]*$": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "category": {"type": "string"},
                "instruction_templates": {
                  "type": "object",
                  "properties": {
                    "basic": {"type": "string"},
                    "detailed": {"type": "string"},
                    "advanced": {"type": "string"}
                  },
                  "required": ["basic"]
                },
                "parameters": {"type": "object"},
                "compatibility": {
                  "type": "object",
                  "properties": {
                    "requires": {"type": "array", "items": {"type": "string"}},
                    "conflicts": {"type": "array", "items": {"type": "string"}},
                    "enhances": {"type": "array", "items": {"type": "string"}}
                  }
                },
                "constraints": {
                  "type": "object",
                  "properties": {
                    "min_age": {"type": "integer", "minimum": 0},
                    "max_age": {"type": "integer", "minimum": 0},
                    "complexity_weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "frequency_weight": {"type": "number", "minimum": 0, "maximum": 1}
                  }
                },
                "conditional_logic": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "condition": {"type": "string"},
                      "modifications": {"type": "object"}
                    },
                    "required": ["condition", "modifications"]
                  }
                },
                "variations": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "weight": {"type": "number", "minimum": 0, "maximum": 1},
                      "instruction": {"type": "string"}
                    },
                    "required": ["weight", "instruction"]
                  }
                },
                "k_shot_examples": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "example_id": {"type": "string"},
                      "weight": {"type": "number", "minimum": 0, "maximum": 1},
                      "age_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2
                      }
                    },
                    "required": ["example_id", "weight"]
                  }
                },
                "metadata": {"type": "object"}
              },
              "required": ["name", "description", "category", "instruction_templates"]
            }
          }
        }
      },
      "required": ["name", "version", "features"]
    }
  },
  "required": ["feature_configuration"]
}
```

#### 5.3 API Interface Specifications

**ModularTemplateManager Interface**:
```python
class ModularTemplateManager:
    """Complete API specification for the modular template manager."""

    def __init__(self,
                 template_config_path: str,
                 feature_config_path: str,
                 strategy_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the modular template manager."""

    def create_prompt(self,
                     selected_words: Dict[str, str],
                     context: Optional[Dict[str, Any]] = None,
                     template_name: Optional[str] = None,
                     features: Optional[List[str]] = None,
                     prompt_id: Optional[str] = None) -> StoryPrompt:
        """Create a story prompt using the modular system."""

    def create_k_shot_prompt(self,
                           selected_words: Dict[str, str],
                           k_shot_examples: List[KShotExample],
                           context: Optional[Dict[str, Any]] = None,
                           template_name: Optional[str] = None,
                           features: Optional[List[str]] = None,
                           prompt_id: Optional[str] = None) -> StoryPrompt:
        """Create a k-shot prompt with examples."""

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""

    def get_available_features(self) -> List[str]:
        """Get list of available feature names."""

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get detailed information about a template."""

    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get detailed information about a feature."""

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current configuration and return status."""

    def reload_configuration(self) -> None:
        """Reload configuration from files."""

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics and performance metrics."""
```

### 6. Critical Implementation Details

#### 6.1 Backward Compatibility Requirements

**Legacy Interface Support**:
```python
class LegacyTemplateAdapter:
    """Adapter to maintain backward compatibility with existing TemplateManager interface."""

    def __init__(self, modular_manager: ModularTemplateManager):
        self.modular_manager = modular_manager
        # Load legacy story features for compatibility
        self.story_features = self._load_legacy_features()

    def create_prompt(self,
                     selected_words: Dict[str, str],
                     additional_condition: Optional[str] = None,
                     prompt_id: Optional[str] = None) -> StoryPrompt:
        """Legacy interface - maps to modular system."""
        context = {"target_age": 3}  # Default legacy context

        if additional_condition:
            # Map legacy additional_condition to features
            features = self._map_legacy_condition_to_features(additional_condition)
        else:
            features = None

        return self.modular_manager.create_prompt(
            selected_words=selected_words,
            context=context,
            template_name="basic_bedtime_story",  # Default legacy template
            features=features,
            prompt_id=prompt_id
        )

    def create_k_shot_prompt(self,
                           selected_words: Dict[str, str],
                           k_shot_examples: List[KShotExample],
                           additional_condition: Optional[str] = None,
                           prompt_id: Optional[str] = None) -> StoryPrompt:
        """Legacy k-shot interface."""
        prompt = self.create_prompt(selected_words, additional_condition, prompt_id)
        prompt.k_shot_examples = k_shot_examples
        prompt.metadata["k_shot_count"] = len(k_shot_examples)
        return prompt

    def get_available_features(self) -> List[str]:
        """Legacy feature interface."""
        return self.story_features

    def add_custom_feature(self, feature: str):
        """Legacy feature addition."""
        if feature not in self.story_features:
            self.story_features.append(feature)
```

#### 6.2 Performance Considerations

**Caching Strategy**:
```python
class PerformanceOptimizedTemplateManager(ModularTemplateManager):
    """Performance-optimized version with caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._template_cache = {}
        self._feature_cache = {}
        self._compatibility_cache = {}

    @lru_cache(maxsize=1000)
    def _get_cached_template(self, template_name: str) -> TemplateDefinition:
        """Cache template lookups."""
        return self.template_registry.get_template(template_name)

    @lru_cache(maxsize=500)
    def _get_cached_compatible_templates(self, features_tuple: tuple) -> List[str]:
        """Cache compatibility calculations."""
        return self.template_registry.get_compatible_templates(list(features_tuple))

    def create_prompt(self, *args, **kwargs) -> StoryPrompt:
        """Optimized prompt creation with caching."""
        # Use cached lookups for better performance
        return super().create_prompt(*args, **kwargs)
```

#### 6.3 Error Handling and Validation

**Comprehensive Error Handling**:
```python
class TemplateSystemError(Exception):
    """Base exception for template system errors."""
    pass

class TemplateNotFoundError(TemplateSystemError):
    """Raised when a requested template is not found."""
    pass

class FeatureCompatibilityError(TemplateSystemError):
    """Raised when incompatible features are selected."""
    pass

class ConfigurationError(TemplateSystemError):
    """Raised when configuration is invalid."""
    pass

class TemplateValidationError(TemplateSystemError):
    """Raised when template validation fails."""
    pass

# Error handling in ModularTemplateManager
def create_prompt(self, *args, **kwargs) -> StoryPrompt:
    try:
        # Template creation logic
        pass
    except KeyError as e:
        raise TemplateNotFoundError(f"Template or feature not found: {e}")
    except ValueError as e:
        raise FeatureCompatibilityError(f"Feature compatibility error: {e}")
    except Exception as e:
        raise TemplateSystemError(f"Unexpected error in template creation: {e}")
```

## Migration Strategy

### Phase 1: Preparation and Foundation (Week 1-2)

#### Step 1.1: Environment Setup
1. **Create new configuration directories**:
   ```bash
   mkdir -p config/handlers
   mkdir -p training/synthetic_data_generation/handlers
   mkdir -p training/synthetic_data_generation/k_shot
   ```

2. **No additional dependencies required** - using standard JSON and importlib

3. **Create configuration schemas**:
   - JSON schema for prompt generator configuration
   - Handler interface validation

#### Step 1.2: Create Default Configurations
1. **Convert existing hardcoded template to JSON configuration**:
   ```json
   {
     "prompt_generation_system": {
       "name": "Modular Prompt Generation System",
       "version": "2.0",
       "prompt_generators": {
         "basic_bedtime_story": {
           "name": "Basic Bedtime Story Generator",
           "handler_class": "training.synthetic_data_generation.handlers.BasicBedtimeStoryHandler",
           "config": {
             "max_words": 150,
             "target_age": 3,
             "language": "English",
             "word_count": 3,
             "template": "Generate simple, short (up to {max_words} words) bed time story...",
             "artifacts": [
               {
                 "name": "feature_instructions",
                 "type": "array",
                 "value": ["make sure the story contains a dialogue", "..."]
               }
             ]
           }
         }
       }
     }
   }
   ```

2. **Create main configuration file**:
   ```json
   {
     "story_generation_config": {
       "generation_settings": {
         "prompt_generator": "basic_bedtime_story"
       },
       "prompt_system": {
         "config_file": "config/prompt_generators.json",
         "default_generator": "basic_bedtime_story"
       }
     }
   }
   ```

#### Step 1.3: Implement Core Handler Interface
1. **Create handler interface** (`training/common/prompt_handlers.py`)
2. **Add basic implementations** with simple logic
3. **Create unit tests** for handler interface

### Phase 2: Core Implementation (Week 2-3)

#### Step 2.1: Implement Prompt Generator Registry
```python
# training/synthetic_data_generation/src/prompt_generator_registry.py
class PromptGeneratorRegistry:
    def __init__(self, config_path: str):
        self.config = self._load_configuration(config_path)
        self.generators = {}
        self._initialize_generators()

    def _initialize_generators(self):
        """Initialize all prompt generators from configuration."""
        for generator_id, generator_config in self.config["prompt_generation_system"]["prompt_generators"].items():
            handler_class = self._load_handler_class(generator_config["handler_class"])
            self.generators[generator_id] = handler_class(generator_config["config"])
```

#### Step 2.2: Create Basic Handler Implementations
```python
# training/synthetic_data_generation/handlers/basic_handlers.py
class BasicBedtimeStoryHandler(PromptGeneratorHandler):
    def generate_prompt(self, selected_words: Dict[str, str], context: Dict[str, Any]) -> StoryPrompt:
        """Generate basic bedtime story prompt - simple, code-free implementation."""
        template = self.config["template"]
        feature_instructions = self._select_feature_instructions(context)

        full_prompt = template.format(
            max_words=self.config["max_words"],
            target_age=self.config["target_age"],
            language=self.config["language"],
            word_count=self.config["word_count"],
            word1=selected_words["word1"],
            word2=selected_words["word2"],
            word3=selected_words["word3"],
            feature_instructions=feature_instructions
        )

        return StoryPrompt(...)
```

#### Step 2.3: Implement K-Shot System
```python
# training/synthetic_data_generation/k_shot/k_shot_system.py
class KShotSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.handlers = {}
        self._initialize_handlers()

    def get_k_shot_examples(self, strategy: str, context: Dict[str, Any]) -> List[KShotExample]:
        """Get K-shot examples using specified strategy."""
        if strategy not in self.handlers:
            strategy = self.config["default_strategy"]

        handler = self.handlers[strategy]
        return handler.get_k_shot_examples(context)
```

### Phase 3: Integration (Week 3-4)

#### Step 3.1: Create Modular Template Manager
```python
# training/synthetic_data_generation/src/modular_template_manager.py
class ModularTemplateManager:
    def __init__(self, config_path: str):
        self.generator_registry = PromptGeneratorRegistry(config_path)

        # Initialize K-shot system if enabled
        config = self._load_config()
        k_shot_config = config["prompt_generation_system"].get("k_shot_system", {})
        if k_shot_config.get("enabled", False):
            self.k_shot_system = KShotSystem(k_shot_config)

    def create_prompt(self, generator_id: str, selected_words: Dict[str, str], context: Dict[str, Any]) -> StoryPrompt:
        """Create prompt using specified generator."""
        generator = self.generator_registry.get_generator(generator_id)
        return generator.generate_prompt(selected_words, context)
```

#### Step 3.2: Update StoryGenerator Integration
```python
# training/synthetic_data_generation/src/story_generator.py
def _initialize_components(self):
    # ... existing code ...

    # Initialize modular template manager
    prompt_config_path = self.config.prompt_system.config_file
    self.template_manager = ModularTemplateManager(prompt_config_path)

    # Update prompt generator to use specified generator
    self.prompt_generator = PromptGenerator(
        vocabulary=self.vocabulary,
        template_manager=self.template_manager,
        conversation_examples_path=self.conversation_examples_path,
        k_shot_count=self.k_shot_count,
        generator_id=self.config.generation_settings.prompt_generator
    )
```

#### Step 3.3: Create Advanced Handler Examples
```python
# training/synthetic_data_generation/handlers/advanced_handlers.py
class AdvancedBedtimeStoryHandler(PromptGeneratorHandler):
    def generate_prompt(self, selected_words: Dict[str, str], context: Dict[str, Any]) -> StoryPrompt:
        """Generate advanced bedtime story prompt with complex logic."""
        # Use advanced feature selector handler
        feature_selector = self.artifacts.get("advanced_features")
        feature_instructions = feature_selector.select_features(context, self.config)

        # Apply age-based template modifications
        template = self._get_age_appropriate_template(context.get("target_age", self.config["target_age"]))

        # Complex formatting logic here
        return StoryPrompt(...)
```

### Phase 4: Testing and Validation (Week 4-5)

#### Step 4.1: Comprehensive Testing
1. **Unit Tests**:
   ```python
   # training/synthetic_data_generation/tests/test_modular_prompt_system.py
   def test_prompt_generator_registry():
       registry = PromptGeneratorRegistry("config/prompt_generators.json")
       assert len(registry.generators) > 0
       assert "basic_bedtime_story" in registry.generators

   def test_basic_handler():
       handler = BasicBedtimeStoryHandler(basic_config)
       test_context = {
           "config": basic_config,
           "artifacts": handler.artifacts,
           "selected_words": {"word1": "moon", "word2": "dance", "word3": "happy"}
       }
       prompt = handler.generate_prompt(test_context)
       assert prompt.full_prompt is not None
       assert handler.validate_config()
   ```

2. **Integration Tests**:
   ```python
   def test_end_to_end_prompt_generation():
       template_manager = ModularTemplateManager("config/prompt_generators.json")

       prompt = template_manager.create_prompt(
           generator_id="basic_bedtime_story",
           context={
               "selected_words": {"word1": "moon", "word2": "dance", "word3": "happy"},
               "target_age": 4
           }
       )

       assert prompt.full_prompt is not None
       assert prompt.metadata["generator"] == "basic_bedtime_story"
   ```

3. **Performance Tests**:
   ```python
   def test_performance_comparison():
       # Compare old vs new system performance
       old_manager = TemplateManager("docs/story_features.json")
       new_manager = ModularTemplateManager("config/prompt_generators.json")

       # Benchmark prompt generation speed
       assert new_generation_time <= old_generation_time * 1.05  # Allow 5% overhead
   ```

#### Step 4.2: Configuration Migration
```python
def test_configuration_migration():
    # Test migration of existing configurations
    migrator = ConfigurationMigrator()
    new_config = migrator.migrate_legacy_config("config/default_config.json")

    assert new_config.prompt_system is not None
    assert new_config.generation_settings.prompt_generator == "basic_bedtime_story"
```

### Phase 5: Deployment and Documentation (Week 5-6)

#### Step 5.1: Update Command-Line Interface
```python
# training/synthetic_data_generation/main.py
parser.add_argument(
    "--prompt-generator",
    type=str,
    help="Prompt generator to use (overrides config setting)"
)

parser.add_argument(
    "--prompt-config",
    type=str,
    help="Path to prompt generators configuration file"
)

parser.add_argument(
    "--list-generators",
    action="store_true",
    help="List available prompt generators and exit"
)
```

#### Step 5.2: Create Migration Tools
```python
# training/synthetic_data_generation/tools/migrate_config.py
class ConfigurationMigrator:
    """Tool to migrate existing configurations to new format."""

    def migrate_legacy_config(self, old_config_path: str) -> ModularStoryGenerationConfig:
        """Migrate legacy configuration to modular format."""
        old_config = load_config(old_config_path)

        # Create prompt system configuration
        prompt_system_config = PromptSystemConfig(
            config_file="config/prompt_generators.json",
            default_generator="basic_bedtime_story"
        )

        return ModularStoryGenerationConfig(
            **old_config.dict(),
            prompt_system=prompt_system_config
        )
```

#### Step 5.3: Documentation Updates
1. **Create comprehensive documentation**:
   - `docs/PROMPT_SYSTEM.md` - Complete guide to prompt generator system
   - `docs/HANDLER_DEVELOPMENT.md` - Guide for creating custom handlers
   - `docs/MIGRATION_GUIDE.md` - Migration from legacy system

2. **Update existing documentation**:
   - Update `docs/DATA_GENERATION.md` with new architecture
   - Add examples to `docs/synthetic_data_generation_implementation.md`

3. **Create example configurations**:
   - Multiple prompt generators for different use cases
   - Handler examples for various complexity levels
   - K-shot configurations for different scenarios

## Risk Assessment and Mitigation

### High Risk: Handler Loading and Dynamic Imports

**Risk**: Dynamic loading of handler classes may fail or introduce security vulnerabilities.

**Mitigation**:
- Strict validation of handler class paths
- Whitelist of allowed handler modules
- Comprehensive error handling for import failures
- Security review of dynamic loading mechanism

### Medium Risk: Configuration Complexity

**Risk**: JSON configuration may become complex for advanced scenarios.

**Mitigation**:
- Provide simple built-in handlers for common cases
- Clear separation between simple and complex configurations
- Comprehensive documentation with examples
- Configuration validation and helpful error messages

### Medium Risk: Performance Impact

**Risk**: Additional abstraction layers may impact generation performance.

**Mitigation**:
- Minimal overhead in handler interface
- Benchmark testing against current system
- Lazy loading of artifacts and handlers
- Caching of frequently used configurations

### Low Risk: Handler Development Complexity

**Risk**: Creating custom handlers may be too complex for users.

**Mitigation**:
- Simple base implementations for common patterns
- Clear handler interface with good documentation
- Examples of handlers with increasing complexity
- Helper utilities for common handler tasks

## Success Metrics

1. **Functionality**: Current behavior can be replicated with basic handler configuration
2. **Performance**: New system performs within 5% of current system speed
3. **Flexibility**: Can create 10+ different prompt generators without code changes for simple cases
4. **Extensibility**: Complex scenarios can be handled with custom handlers
5. **Usability**: Basic configuration changes take < 5 minutes
6. **Maintainability**: Adding new simple generators requires only JSON configuration
7. **Reliability**: System handles invalid configurations gracefully with clear error messages

## Conclusion

The current prompt template system is **fundamentally inflexible** due to hardcoded templates, static features, and monolithic design. This severely limits experimentation, customization, and evolution of the prompt generation system.

This plugin-based architecture plan addresses these critical limitations by:

1. **Creating a plugin-based prompt generator system** that separates configuration from logic
2. **Using simple JSON configuration** with no embedded business logic
3. **Enabling extensibility through dedicated handlers** for complex scenarios
4. **Keeping simple scenarios code-free** while allowing complex customization
5. **Providing clean separation of concerns** between prompt generation and k-shot management

The modular system will transform prompt generation from a rigid, hardcoded process into a flexible, configurable, and extensible framework that can adapt to changing requirements while maintaining simplicity for common use cases.
```
```
```
```
