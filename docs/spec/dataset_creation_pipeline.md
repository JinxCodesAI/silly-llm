# Phase 1: Modular Synthetic Dataset Creation Pipeline

## 1. Objective
Design and implement a highly modular, extensible synthetic dataset creation pipeline that generates large-scale datasets of short, simple, and coherent stories. The architecture emphasizes flexibility, maintainability, and future-proofing through clean separation of concerns and pluggable components.

## 2. Architectural Overview

### 2.1 Core Design Principles
- **Provider Pattern**: Abstract interfaces for LLM providers, storage providers, and data processors
- **Dependency Injection**: Loose coupling through dependency injection container
- **Configuration-Driven**: Centralized configuration management with validation
- **Plugin Architecture**: Easy extension through plugin system
- **Async/Await**: Non-blocking operations for improved performance
- **Type Safety**: Comprehensive type hints and runtime validation

### 2.2 Component Hierarchy
```
DatasetCreationPipeline
├── LLMProviderFactory
│   ├── OpenAICompatibleProvider
│   ├── TransformersProvider (Qwen, etc.)
│   └── CustomProvider (extensible)
├── VocabularyManager
│   ├── VocabularyGenerator
│   ├── VocabularyValidator
│   └── VocabularyPersistence
├── PromptGenerationEngine
│   ├── PromptTemplateManager
│   ├── FeatureSelector
│   └── WordSelector
├── StoryGenerationOrchestrator
│   ├── GenerationStrategy (batch, streaming, branching)
│   ├── QualityFilter
│   └── DuplicationDetector
└── StorageManager
    ├── LocalStorageProvider
    ├── CloudStorageProvider
    └── DatabaseProvider
```

### 2.3 Framework Selection Rationale
- **Primary Framework**: Hugging Face `transformers` for model abstraction
- **Async Framework**: `asyncio` with `aiohttp` for API providers
- **Acceleration**: `accelerate` library for distributed processing
- **Configuration**: `pydantic` for type-safe configuration management
- **Dependency Injection**: Custom lightweight DI container

## 3. Modular Vocabulary Management System

### 3.1 Architecture Overview

The vocabulary management system is designed as a collection of loosely coupled components that can be easily extended, tested, and maintained.

#### 3.1.1 Core Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum

class WordCategory(Enum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"

class VocabularyWord(BaseModel):
    word: str
    category: WordCategory
    confidence_score: float
    age_appropriateness: float
    complexity_score: float
    source: str

class VocabularyConfig(BaseModel):
    target_word_count: int = 1500
    category_distribution: Dict[WordCategory, float] = {
        WordCategory.NOUN: 0.5,
        WordCategory.VERB: 0.25,
        WordCategory.ADJECTIVE: 0.25
    }
    age_range: tuple[int, int] = (3, 4)
    batch_size: int = 20
    max_retries: int = 3
    quality_threshold: float = 0.7

class VocabularyProvider(ABC):
    """Abstract base class for vocabulary generation providers."""

    @abstractmethod
    async def generate_words(self, category: WordCategory, count: int, existing_words: set[str]) -> List[VocabularyWord]:
        """Generate vocabulary words for specified category."""
        pass

    @abstractmethod
    def validate_word(self, word: str, category: WordCategory) -> float:
        """Validate word appropriateness and return confidence score."""
        pass
```

#### 3.1.2 LLM-Based Vocabulary Provider

```python
class LLMVocabularyProvider(VocabularyProvider):
    """LLM-based vocabulary generation using configurable providers."""

    def __init__(self, llm_provider: LLMProvider, config: VocabularyConfig):
        self.llm_provider = llm_provider
        self.config = config
        self.prompt_templates = self._load_prompt_templates()

    async def generate_words(self, category: WordCategory, count: int, existing_words: set[str]) -> List[VocabularyWord]:
        """Generate words using LLM with sophisticated prompting."""
        prompt = self._build_generation_prompt(category, count, existing_words)

        for attempt in range(self.config.max_retries):
            try:
                response = await self.llm_provider.generate(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=200
                )
                words = self._parse_response(response, category)
                validated_words = [word for word in words if self._validate_word_quality(word)]
                return validated_words[:count]
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise VocabularyGenerationError(f"Failed to generate words after {self.config.max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _build_generation_prompt(self, category: WordCategory, count: int, existing_words: set[str]) -> str:
        """Build sophisticated prompt for vocabulary generation."""
        base_template = self.prompt_templates[category]

        # Include examples and constraints
        examples = self._get_category_examples(category)
        constraints = self._build_constraints(existing_words)

        return base_template.format(
            count=count,
            examples=examples,
            constraints=constraints,
            age_range="3-4 year old"
        )
```

#### 3.1.3 Vocabulary Validation System

```python
class VocabularyValidator:
    """Comprehensive vocabulary validation system."""

    def __init__(self, config: VocabularyConfig):
        self.config = config
        self.age_appropriateness_checker = AgeAppropriatenessChecker()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.profanity_filter = ProfanityFilter()

    def validate_word(self, word: VocabularyWord) -> ValidationResult:
        """Comprehensive word validation."""
        checks = [
            self._check_age_appropriateness(word),
            self._check_complexity(word),
            self._check_profanity(word),
            self._check_format(word),
            self._check_category_consistency(word)
        ]

        return ValidationResult(
            is_valid=all(check.passed for check in checks),
            checks=checks,
            overall_score=sum(check.score for check in checks) / len(checks)
        )

    def _check_age_appropriateness(self, word: VocabularyWord) -> ValidationCheck:
        """Check if word is appropriate for target age range."""
        score = self.age_appropriateness_checker.score_word(word.word, self.config.age_range)
        return ValidationCheck(
            name="age_appropriateness",
            passed=score >= self.config.quality_threshold,
            score=score,
            details=f"Age appropriateness score: {score}"
        )
```

#### 3.1.4 Vocabulary Manager Orchestrator

```python
class VocabularyManager:
    """Main orchestrator for vocabulary generation and management."""

    def __init__(self,
                 provider: VocabularyProvider,
                 validator: VocabularyValidator,
                 persistence: VocabularyPersistence,
                 config: VocabularyConfig):
        self.provider = provider
        self.validator = validator
        self.persistence = persistence
        self.config = config

    async def generate_vocabulary(self) -> Vocabulary:
        """Generate complete vocabulary with validation and persistence."""
        vocabulary = Vocabulary()

        # Load existing vocabulary if available
        existing_vocab = await self.persistence.load_vocabulary()
        if existing_vocab:
            vocabulary = existing_vocab
            logger.info(f"Loaded existing vocabulary with {len(vocabulary)} words")

        # Generate missing words for each category
        for category in WordCategory:
            target_count = int(self.config.target_word_count * self.config.category_distribution[category])
            current_count = len(vocabulary.get_words_by_category(category))

            if current_count < target_count:
                needed_words = target_count - current_count
                logger.info(f"Generating {needed_words} {category.value} words")

                new_words = await self._generate_category_words(
                    category, needed_words, vocabulary.get_all_words()
                )
                vocabulary.add_words(new_words)

                # Periodic persistence
                await self.persistence.save_vocabulary(vocabulary)

        # Final validation and cleanup
        vocabulary = self._finalize_vocabulary(vocabulary)
        await self.persistence.save_vocabulary(vocabulary)

        return vocabulary

    async def _generate_category_words(self, category: WordCategory, count: int, existing_words: set[str]) -> List[VocabularyWord]:
        """Generate words for specific category with quality control."""
        generated_words = []
        attempts = 0
        max_attempts = count * 3  # Allow multiple attempts to reach target

        while len(generated_words) < count and attempts < max_attempts:
            batch_size = min(self.config.batch_size, count - len(generated_words))

            try:
                candidate_words = await self.provider.generate_words(
                    category, batch_size, existing_words
                )

                # Validate each word
                for word in candidate_words:
                    validation_result = self.validator.validate_word(word)
                    if validation_result.is_valid:
                        generated_words.append(word)
                        existing_words.add(word.word)

                attempts += 1

            except Exception as e:
                logger.warning(f"Failed to generate batch for {category}: {e}")
                attempts += 1
                await asyncio.sleep(1)

        if len(generated_words) < count:
            logger.warning(f"Only generated {len(generated_words)}/{count} words for {category}")

        return generated_words
```

### 3.2 Configuration Management

```yaml
# configs/vocabulary.yaml
vocabulary:
  target_word_count: 1500
  category_distribution:
    noun: 0.5
    verb: 0.25
    adjective: 0.25

  age_range: [3, 4]
  quality_threshold: 0.7
  batch_size: 20
  max_retries: 3

llm_provider:
  type: "transformers"  # or "openai_compatible"
  config:
    model_name: "Qwen/Qwen3-4B"
    device: "auto"
    generation_params:
      temperature: 0.7
      max_tokens: 200

validation:
  age_appropriateness:
    enabled: true
    threshold: 0.7

  complexity:
    enabled: true
    max_syllables: 3
    max_letters: 8

  profanity:
    enabled: true
    strict_mode: true

persistence:
  type: "local"  # or "s3", "database"
  config:
    file_path: "./data/vocabulary.json"
    backup_enabled: true
    versioning: true
```

### 3.3 Usage Examples

#### 3.3.1 Basic Vocabulary Generation
```python
# Example usage of the vocabulary management system
from silly_llm.data.vocabulary import VocabularyManager
from silly_llm.core.config import load_config
from silly_llm.core.container import DIContainer

async def generate_vocabulary():
    # Load configuration
    config = load_config("configs/vocabulary.yaml")

    # Setup dependency injection container
    container = DIContainer()
    container.register_config(config)

    # Get vocabulary manager from container
    vocab_manager = container.get(VocabularyManager)

    # Generate vocabulary
    vocabulary = await vocab_manager.generate_vocabulary()

    print(f"Generated vocabulary with {len(vocabulary)} words")
    print(f"Nouns: {len(vocabulary.get_words_by_category(WordCategory.NOUN))}")
    print(f"Verbs: {len(vocabulary.get_words_by_category(WordCategory.VERB))}")
    print(f"Adjectives: {len(vocabulary.get_words_by_category(WordCategory.ADJECTIVE))}")

# Run the generation
asyncio.run(generate_vocabulary())
```

#### 3.3.2 Custom Vocabulary Provider
```python
class CustomVocabularyProvider(VocabularyProvider):
    """Custom provider that combines multiple sources."""

    def __init__(self, llm_provider: LLMProvider, word_list_provider: WordListProvider):
        self.llm_provider = llm_provider
        self.word_list_provider = word_list_provider

    async def generate_words(self, category: WordCategory, count: int, existing_words: set[str]) -> List[VocabularyWord]:
        # First try to get words from curated word lists
        curated_words = await self.word_list_provider.get_words(category, count // 2)

        # Then generate additional words using LLM
        remaining_count = count - len(curated_words)
        if remaining_count > 0:
            llm_words = await self._generate_with_llm(category, remaining_count, existing_words)
            curated_words.extend(llm_words)

        return curated_words[:count]
```

### 3.4 Data Structures and Persistence

#### 3.4.1 Enhanced Vocabulary Data Model
```python
class Vocabulary(BaseModel):
    """Enhanced vocabulary data model with metadata."""

    words: Dict[WordCategory, List[VocabularyWord]] = defaultdict(list)
    metadata: VocabularyMetadata
    version: str = "1.0"
    created_at: datetime
    updated_at: datetime

    def add_words(self, words: List[VocabularyWord]) -> None:
        """Add words to vocabulary with deduplication."""
        for word in words:
            if not self._word_exists(word.word):
                self.words[word.category].append(word)
        self.updated_at = datetime.now()

    def get_words_by_category(self, category: WordCategory) -> List[VocabularyWord]:
        """Get all words for a specific category."""
        return self.words[category]

    def get_random_words(self, category: WordCategory, count: int) -> List[VocabularyWord]:
        """Get random words from category for story generation."""
        words = self.words[category]
        return random.sample(words, min(count, len(words)))

    def get_all_words(self) -> set[str]:
        """Get set of all word strings for deduplication."""
        all_words = set()
        for category_words in self.words.values():
            all_words.update(word.word for word in category_words)
        return all_words

    def export_simple_format(self) -> Dict[str, List[str]]:
        """Export to simple format for backward compatibility."""
        return {
            category.value: [word.word for word in words]
            for category, words in self.words.items()
        }

class VocabularyMetadata(BaseModel):
    """Metadata about vocabulary generation."""

    target_word_count: int
    actual_word_count: int
    generation_config: VocabularyConfig
    quality_metrics: Dict[str, float]
    generation_time: float
    provider_info: Dict[str, Any]
```

#### 3.4.2 Flexible Persistence Layer
```python
class VocabularyPersistence(ABC):
    """Abstract base class for vocabulary persistence."""

    @abstractmethod
    async def save_vocabulary(self, vocabulary: Vocabulary) -> None:
        """Save vocabulary to storage."""
        pass

    @abstractmethod
    async def load_vocabulary(self) -> Optional[Vocabulary]:
        """Load vocabulary from storage."""
        pass

    @abstractmethod
    async def backup_vocabulary(self, vocabulary: Vocabulary) -> str:
        """Create backup and return backup identifier."""
        pass

class LocalVocabularyPersistence(VocabularyPersistence):
    """Local file system persistence."""

    def __init__(self, file_path: str, backup_enabled: bool = True):
        self.file_path = Path(file_path)
        self.backup_enabled = backup_enabled
        self.backup_dir = self.file_path.parent / "backups"

    async def save_vocabulary(self, vocabulary: Vocabulary) -> None:
        """Save vocabulary with atomic write and optional backup."""
        if self.backup_enabled and self.file_path.exists():
            await self.backup_vocabulary(vocabulary)

        # Atomic write using temporary file
        temp_path = self.file_path.with_suffix('.tmp')
        async with aiofiles.open(temp_path, 'w') as f:
            await f.write(vocabulary.json(indent=2))

        temp_path.replace(self.file_path)
        logger.info(f"Vocabulary saved to {self.file_path}")

class CloudVocabularyPersistence(VocabularyPersistence):
    """Cloud storage persistence (S3, GCS, etc.)."""

    def __init__(self, cloud_provider: CloudStorageProvider, bucket: str, key: str):
        self.cloud_provider = cloud_provider
        self.bucket = bucket
        self.key = key

    async def save_vocabulary(self, vocabulary: Vocabulary) -> None:
        """Save vocabulary to cloud storage."""
        data = vocabulary.json(indent=2)
        await self.cloud_provider.upload_text(self.bucket, self.key, data)
```

## 4. Modular Story Generation Pipeline

### 4.1 Architecture Overview

The story generation pipeline is designed as a highly modular system that separates concerns and enables easy extension and testing.

#### 4.1.1 Core Components

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel
from enum import Enum

class GenerationStrategy(Enum):
    BATCH = "batch"
    STREAMING = "streaming"
    BRANCHING = "branching"
    ADAPTIVE = "adaptive"

class StoryPrompt(BaseModel):
    """Structured representation of a story generation prompt."""

    prompt_id: str
    template: str
    selected_words: Dict[WordCategory, str]
    selected_features: List[str]
    metadata: Dict[str, Any]
    generation_params: Dict[str, Any]

class GeneratedStory(BaseModel):
    """Structured representation of a generated story."""

    story_id: str
    prompt_id: str
    content: str
    metadata: StoryMetadata
    quality_scores: Dict[str, float]
    generation_time: float

class StoryMetadata(BaseModel):
    """Metadata about story generation."""

    word_count: int
    paragraph_count: int
    required_words_used: List[str]
    features_present: List[str]
    generation_strategy: GenerationStrategy
    model_info: Dict[str, Any]
```

#### 4.1.2 Prompt Generation System

```python
class PromptGenerator(ABC):
    """Abstract base class for prompt generators."""

    @abstractmethod
    def generate_prompts(self, count: int, config: PromptConfig) -> List[StoryPrompt]:
        """Generate story prompts based on configuration."""
        pass

class TemplateBasedPromptGenerator(PromptGenerator):
    """Template-based prompt generator with sophisticated word and feature selection."""

    def __init__(self,
                 vocabulary: Vocabulary,
                 feature_manager: FeatureManager,
                 template_manager: TemplateManager):
        self.vocabulary = vocabulary
        self.feature_manager = feature_manager
        self.template_manager = template_manager

    def generate_prompts(self, count: int, config: PromptConfig) -> List[StoryPrompt]:
        """Generate prompts with intelligent word and feature combinations."""
        prompts = []

        for i in range(count):
            # Select words with diversity constraints
            selected_words = self._select_words_with_diversity(prompts)

            # Select features with compatibility checking
            selected_features = self._select_compatible_features(selected_words, config)

            # Choose appropriate template
            template = self.template_manager.select_template(selected_features)

            # Build prompt
            prompt = self._build_prompt(template, selected_words, selected_features)

            prompts.append(StoryPrompt(
                prompt_id=f"prompt_{i:06d}",
                template=template.name,
                selected_words=selected_words,
                selected_features=selected_features,
                metadata=self._generate_metadata(selected_words, selected_features),
                generation_params=config.generation_params
            ))

        return prompts

    def _select_words_with_diversity(self, existing_prompts: List[StoryPrompt]) -> Dict[WordCategory, str]:
        """Select words ensuring diversity across prompts."""
        # Track word usage to ensure diversity
        used_combinations = {
            (p.selected_words[WordCategory.NOUN],
             p.selected_words[WordCategory.VERB],
             p.selected_words[WordCategory.ADJECTIVE])
            for p in existing_prompts
        }

        max_attempts = 100
        for _ in range(max_attempts):
            words = {
                WordCategory.NOUN: random.choice(self.vocabulary.get_words_by_category(WordCategory.NOUN)).word,
                WordCategory.VERB: random.choice(self.vocabulary.get_words_by_category(WordCategory.VERB)).word,
                WordCategory.ADJECTIVE: random.choice(self.vocabulary.get_words_by_category(WordCategory.ADJECTIVE)).word
            }

            combination = (words[WordCategory.NOUN], words[WordCategory.VERB], words[WordCategory.ADJECTIVE])
            if combination not in used_combinations:
                return words

        # Fallback to random selection if diversity constraint can't be met
        return {
            WordCategory.NOUN: random.choice(self.vocabulary.get_words_by_category(WordCategory.NOUN)).word,
            WordCategory.VERB: random.choice(self.vocabulary.get_words_by_category(WordCategory.VERB)).word,
            WordCategory.ADJECTIVE: random.choice(self.vocabulary.get_words_by_category(WordCategory.ADJECTIVE)).word
        }
```

#### 4.1.3 Feature Management System

```python
class StoryFeature(BaseModel):
    """Represents a story feature with metadata."""

    name: str
    description: str
    prompt_text: str
    compatibility: List[str]  # Compatible with other features
    conflicts: List[str]      # Conflicts with other features
    difficulty: float         # 0.0 to 1.0
    frequency_weight: float   # Relative frequency in dataset

class FeatureManager:
    """Manages story features and their relationships."""

    def __init__(self, features_config: Dict[str, Any]):
        self.features = self._load_features(features_config)
        self.compatibility_matrix = self._build_compatibility_matrix()

    def select_features(self, count_range: tuple[int, int], constraints: Optional[Dict[str, Any]] = None) -> List[str]:
        """Select compatible features based on constraints."""
        available_features = list(self.features.keys())

        if constraints:
            available_features = self._apply_constraints(available_features, constraints)

        # Select number of features
        num_features = random.randint(*count_range)

        if num_features == 0:
            return []

        # Start with a random feature
        selected = [random.choice(available_features)]

        # Add compatible features
        for _ in range(num_features - 1):
            compatible = self._get_compatible_features(selected, available_features)
            if compatible:
                selected.append(random.choice(compatible))
            else:
                break

        return selected

    def _get_compatible_features(self, selected: List[str], available: List[str]) -> List[str]:
        """Get features compatible with all selected features."""
        compatible = []

        for feature in available:
            if feature in selected:
                continue

            # Check compatibility with all selected features
            is_compatible = True
            for selected_feature in selected:
                if not self._are_compatible(feature, selected_feature):
                    is_compatible = False
                    break

            if is_compatible:
                compatible.append(feature)

        return compatible

    def _are_compatible(self, feature1: str, feature2: str) -> bool:
        """Check if two features are compatible."""
        f1 = self.features[feature1]
        f2 = self.features[feature2]

        # Check explicit conflicts
        if feature2 in f1.conflicts or feature1 in f2.conflicts:
            return False

        # Check explicit compatibility
        if feature2 in f1.compatibility or feature1 in f2.compatibility:
            return True

        # Default compatibility based on difficulty
        difficulty_diff = abs(f1.difficulty - f2.difficulty)
        return difficulty_diff <= 0.5  # Compatible if difficulty difference is small
```

#### 4.1.4 Template Management System

```python
class PromptTemplate(BaseModel):
    """Represents a prompt template."""

    name: str
    template: str
    required_placeholders: List[str]
    optional_placeholders: List[str]
    suitable_features: List[str]
    complexity_level: float

class TemplateManager:
    """Manages prompt templates and selection logic."""

    def __init__(self, templates_config: Dict[str, Any]):
        self.templates = self._load_templates(templates_config)

    def select_template(self, features: List[str]) -> PromptTemplate:
        """Select most appropriate template for given features."""
        suitable_templates = []

        for template in self.templates:
            suitability_score = self._calculate_suitability(template, features)
            if suitability_score > 0.5:
                suitable_templates.append((template, suitability_score))

        if not suitable_templates:
            # Fallback to default template
            return self._get_default_template()

        # Select template with highest suitability score
        suitable_templates.sort(key=lambda x: x[1], reverse=True)
        return suitable_templates[0][0]

    def _calculate_suitability(self, template: PromptTemplate, features: List[str]) -> float:
        """Calculate how suitable a template is for given features."""
        if not features:
            return 1.0  # All templates suitable for no features

        suitable_features = set(template.suitable_features)
        selected_features = set(features)

        # Calculate overlap
        overlap = len(suitable_features.intersection(selected_features))
        total = len(selected_features)

        return overlap / total if total > 0 else 0.0
```

### 4.2 Configuration Schema

```yaml
# configs/prompt_generation.yaml
prompt_generation:
  num_prompts: 10000
  seed: 42

  word_selection:
    diversity_constraint: true
    max_word_reuse: 5  # Maximum times a word can be reused

  feature_selection:
    count_range: [0, 3]  # Min and max number of features
    difficulty_balance: true

  templates:
    default: "basic_story"
    custom_templates_enabled: true

generation_params:
  max_new_tokens: 512
  temperature: 0.8
  top_p: 0.9
  repetition_penalty: 1.1

features:
  dialogue:
    description: "Story contains character dialogue"
    prompt_text: "The story should contain at least one dialogue between characters"
    compatibility: ["character_interaction", "conflict"]
    conflicts: ["monologue"]
    difficulty: 0.3
    frequency_weight: 0.4

  plot_twist:
    description: "Story has an unexpected turn"
    prompt_text: "Include a plot twist or unexpected turn in the story"
    compatibility: ["suspense", "mystery"]
    conflicts: ["predictable_ending"]
    difficulty: 0.7
    frequency_weight: 0.2

  moral_value:
    description: "Story teaches a moral lesson"
    prompt_text: "The story should teach a simple moral lesson appropriate for young children"
    compatibility: ["character_growth", "conflict_resolution"]
    conflicts: ["amoral"]
    difficulty: 0.5
    frequency_weight: 0.3

templates:
  basic_story:
    template: |
      Write a short story (3-5 paragraphs) which only uses very simple words that a 3-4 year old child would likely understand.
      The story should use the verb "{verb}", the noun "{noun}" and the adjective "{adjective}".
      {features_clause}
      Remember to only use simple words!
    required_placeholders: ["verb", "noun", "adjective"]
    optional_placeholders: ["features_clause"]
    suitable_features: []  # Suitable for all features
    complexity_level: 0.5

  dialogue_focused:
    template: |
      Write a short story (3-5 paragraphs) with simple words for a 3-4 year old.
      The story must include conversations between characters.
      Use the words: "{verb}", "{noun}", and "{adjective}".
      {features_clause}
      Make sure the dialogue sounds natural for young children.
    required_placeholders: ["verb", "noun", "adjective"]
    optional_placeholders: ["features_clause"]
    suitable_features: ["dialogue", "character_interaction"]
    complexity_level: 0.6
```

## 5. Story Generation Orchestration System

### 5.1 Architecture Overview

The story generation orchestration system coordinates all components to efficiently generate high-quality synthetic stories at scale.

#### 5.1.1 Generation Strategy Interface

```python
class GenerationStrategy(ABC):
    """Abstract base class for story generation strategies."""

    @abstractmethod
    async def generate_stories(self,
                             prompts: List[StoryPrompt],
                             llm_provider: LLMProvider,
                             config: GenerationConfig) -> AsyncGenerator[GeneratedStory, None]:
        """Generate stories using specific strategy."""
        pass

    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return information about this strategy."""
        pass

class BatchGenerationStrategy(GenerationStrategy):
    """Batch processing strategy for efficient parallel generation."""

    async def generate_stories(self,
                             prompts: List[StoryPrompt],
                             llm_provider: LLMProvider,
                             config: GenerationConfig) -> AsyncGenerator[GeneratedStory, None]:
        """Generate stories in batches with parallel processing."""

        # Process prompts in batches
        for batch_start in range(0, len(prompts), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            # Generate stories for batch in parallel
            tasks = [
                self._generate_single_story(prompt, llm_provider, config)
                for prompt in batch_prompts
            ]

            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Yield successful results
            for result in batch_results:
                if isinstance(result, GeneratedStory):
                    yield result
                else:
                    logger.error(f"Generation failed: {result}")

    async def _generate_single_story(self,
                                   prompt: StoryPrompt,
                                   llm_provider: LLMProvider,
                                   config: GenerationConfig) -> GeneratedStory:
        """Generate a single story with error handling and quality validation."""
        start_time = time.time()

        try:
            # Generate story content
            content = await llm_provider.generate(
                prompt=prompt.template.format(**prompt.selected_words,
                                            features_clause=self._build_features_clause(prompt.selected_features)),
                **config.generation_params
            )

            # Validate and process content
            processed_content = await self._process_content(content, prompt)

            # Calculate quality scores
            quality_scores = await self._calculate_quality_scores(processed_content, prompt)

            # Create story object
            story = GeneratedStory(
                story_id=f"story_{uuid.uuid4().hex[:8]}",
                prompt_id=prompt.prompt_id,
                content=processed_content,
                metadata=self._create_metadata(processed_content, prompt),
                quality_scores=quality_scores,
                generation_time=time.time() - start_time
            )

            return story

        except Exception as e:
            logger.error(f"Failed to generate story for prompt {prompt.prompt_id}: {e}")
            raise GenerationError(f"Story generation failed: {e}")

class BranchingGenerationStrategy(GenerationStrategy):
    """Advanced branching strategy for generating story variants."""

    def __init__(self, branching_config: BranchingConfig):
        self.config = branching_config

    async def generate_stories(self,
                             prompts: List[StoryPrompt],
                             llm_provider: LLMProvider,
                             config: GenerationConfig) -> AsyncGenerator[GeneratedStory, None]:
        """Generate multiple story variants using logit-based branching."""

        for prompt in prompts:
            # Generate story family using branching
            story_variants = await self._generate_story_family(prompt, llm_provider, config)

            for variant in story_variants:
                yield variant

    async def _generate_story_family(self,
                                   prompt: StoryPrompt,
                                   llm_provider: LLMProvider,
                                   config: GenerationConfig) -> List[GeneratedStory]:
        """Generate multiple variants of a story using branching logic."""

        if not isinstance(llm_provider, TransformersProvider):
            # Fallback to simple generation for non-transformers providers
            story = await self._generate_single_story(prompt, llm_provider, config)
            return [story]

        # Use advanced branching with transformers
        return await self._generate_with_branching(prompt, llm_provider, config)

    async def _generate_with_branching(self,
                                     prompt: StoryPrompt,
                                     llm_provider: TransformersProvider,
                                     config: GenerationConfig) -> List[GeneratedStory]:
        """Implement logit-based branching generation."""

        # Initialize generation state
        input_text = prompt.template.format(**prompt.selected_words,
                                           features_clause=self._build_features_clause(prompt.selected_features))
        input_ids = llm_provider.tokenizer.encode(input_text, return_tensors="pt")

        # Generation stack for depth-first exploration
        generation_stack = [(input_ids, None, 0)]  # (input_ids, past_key_values, depth)
        completed_stories = []

        while generation_stack and len(completed_stories) < self.config.max_variants_per_prompt:
            current_ids, past_kv, depth = generation_stack.pop()

            # Generate next token(s)
            story_variant = await self._continue_generation(
                current_ids, past_kv, llm_provider, config, depth
            )

            if story_variant:
                completed_stories.append(story_variant)

        return completed_stories
```

#### 5.1.2 Story Generation Orchestrator

```python
class StoryGenerationOrchestrator:
    """Main orchestrator for the story generation pipeline."""

    def __init__(self,
                 llm_provider: LLMProvider,
                 prompt_generator: PromptGenerator,
                 generation_strategy: GenerationStrategy,
                 quality_filter: QualityFilter,
                 storage_manager: StorageManager,
                 config: GenerationConfig):
        self.llm_provider = llm_provider
        self.prompt_generator = prompt_generator
        self.generation_strategy = generation_strategy
        self.quality_filter = quality_filter
        self.storage_manager = storage_manager
        self.config = config

    async def generate_dataset(self) -> DatasetGenerationResult:
        """Generate complete synthetic dataset."""
        logger.info(f"Starting dataset generation with {self.config.total_stories} target stories")

        # Generate prompts
        prompts = self.prompt_generator.generate_prompts(
            count=self.config.total_stories,
            config=self.config.prompt_config
        )
        logger.info(f"Generated {len(prompts)} prompts")

        # Initialize progress tracking
        progress_tracker = ProgressTracker(total=len(prompts))
        generated_stories = []

        # Generate stories
        async for story in self.generation_strategy.generate_stories(prompts, self.llm_provider, self.config):
            # Apply quality filtering
            if await self.quality_filter.passes_filter(story):
                generated_stories.append(story)

                # Save story incrementally
                await self.storage_manager.save_story(story)

                # Update progress
                progress_tracker.update(1)

                if len(generated_stories) % 100 == 0:
                    logger.info(f"Generated {len(generated_stories)} high-quality stories")

        # Finalize dataset
        dataset_result = await self._finalize_dataset(generated_stories)

        logger.info(f"Dataset generation complete: {len(generated_stories)} stories generated")
        return dataset_result

    async def _finalize_dataset(self, stories: List[GeneratedStory]) -> DatasetGenerationResult:
        """Finalize dataset with metadata and statistics."""

        # Calculate dataset statistics
        stats = DatasetStatistics(
            total_stories=len(stories),
            total_words=sum(story.metadata.word_count for story in stories),
            average_story_length=sum(story.metadata.word_count for story in stories) / len(stories),
            feature_distribution=self._calculate_feature_distribution(stories),
            quality_metrics=self._calculate_quality_metrics(stories)
        )

        # Create dataset metadata
        metadata = DatasetMetadata(
            generation_config=self.config,
            statistics=stats,
            generation_time=time.time() - self.start_time,
            model_info=await self.llm_provider.get_capabilities()
        )

        # Save final dataset
        await self.storage_manager.save_dataset_metadata(metadata)

        return DatasetGenerationResult(
            stories=stories,
            metadata=metadata,
            statistics=stats
        )
```

#### 5.1.3 Quality Control System

```python
class QualityFilter:
    """Comprehensive quality filtering for generated stories."""

    def __init__(self, config: QualityConfig):
        self.config = config
        self.validators = self._initialize_validators()

    async def passes_filter(self, story: GeneratedStory) -> bool:
        """Check if story passes all quality filters."""

        # Run all validators
        validation_results = []
        for validator in self.validators:
            result = await validator.validate(story)
            validation_results.append(result)

        # Check if story passes all required validations
        required_passed = all(
            result.passed for result in validation_results
            if result.validator_type in self.config.required_validators
        )

        # Check overall quality score
        overall_score = sum(result.score for result in validation_results) / len(validation_results)
        score_passed = overall_score >= self.config.min_quality_score

        return required_passed and score_passed

    def _initialize_validators(self) -> List[StoryValidator]:
        """Initialize story validators based on configuration."""
        validators = []

        if self.config.check_length:
            validators.append(LengthValidator(self.config.length_constraints))

        if self.config.check_vocabulary:
            validators.append(VocabularyValidator(self.config.vocabulary_constraints))

        if self.config.check_coherence:
            validators.append(CoherenceValidator(self.config.coherence_config))

        if self.config.check_required_words:
            validators.append(RequiredWordsValidator())

        if self.config.check_features:
            validators.append(FeatureValidator(self.config.feature_config))

        return validators

class StoryValidator(ABC):
    """Abstract base class for story validators."""

    @abstractmethod
    async def validate(self, story: GeneratedStory) -> ValidationResult:
        """Validate story and return result."""
        pass

class LengthValidator(StoryValidator):
    """Validates story length constraints."""

    def __init__(self, constraints: LengthConstraints):
        self.constraints = constraints

    async def validate(self, story: GeneratedStory) -> ValidationResult:
        word_count = story.metadata.word_count
        paragraph_count = story.metadata.paragraph_count

        # Check word count
        word_count_valid = (
            self.constraints.min_words <= word_count <= self.constraints.max_words
        )

        # Check paragraph count
        paragraph_count_valid = (
            self.constraints.min_paragraphs <= paragraph_count <= self.constraints.max_paragraphs
        )

        passed = word_count_valid and paragraph_count_valid
        score = 1.0 if passed else 0.0

        return ValidationResult(
            validator_type="length",
            passed=passed,
            score=score,
            details={
                "word_count": word_count,
                "paragraph_count": paragraph_count,
                "word_count_valid": word_count_valid,
                "paragraph_count_valid": paragraph_count_valid
            }
        )
```

### 5.2 Configuration Schema

```yaml
# configs/story_generation.yaml
generation:
  total_stories: 10000
  batch_size: 32
  strategy: "batch"  # or "branching", "adaptive"

  generation_params:
    max_new_tokens: 512
    temperature: 0.8
    top_p: 0.9
    repetition_penalty: 1.1
    do_sample: true

branching:  # Only used if strategy is "branching"
  max_variants_per_prompt: 5
  max_branching_depth: 3
  high_prob_threshold: 0.9
  low_prob_threshold: 0.01

quality_control:
  enabled: true
  min_quality_score: 0.7
  required_validators: ["length", "vocabulary", "required_words"]

  length_constraints:
    min_words: 50
    max_words: 300
    min_paragraphs: 2
    max_paragraphs: 5

  vocabulary_constraints:
    check_age_appropriateness: true
    max_complex_words: 5
    forbidden_words: []

  coherence_config:
    check_narrative_flow: true
    check_character_consistency: true
    min_coherence_score: 0.6

storage:
  type: "local"  # or "s3", "gcs"
  config:
    output_dir: "./generated_stories"
    format: "jsonl"
    compression: "gzip"
    backup_enabled: true

  incremental_save:
    enabled: true
    save_every_n_stories: 100
    checkpoint_enabled: true

monitoring:
  progress_reporting:
    enabled: true
    report_every_n_stories: 50

  quality_monitoring:
    enabled: true
    alert_on_quality_drop: true
    quality_threshold: 0.6

  performance_monitoring:
    enabled: true
    track_generation_speed: true
    track_memory_usage: true
```

### 5.3 Implementation Benefits

#### 5.3.1 Modularity and Extensibility
- **Easy Provider Swapping**: Switch between OpenAI API and local transformers with configuration change
- **Pluggable Strategies**: Add new generation strategies without modifying core code
- **Extensible Validation**: Add custom quality validators through plugin system
- **Flexible Storage**: Support multiple storage backends through provider pattern

#### 5.3.2 Scalability and Performance
- **Async Processing**: Non-blocking operations for improved throughput
- **Batch Processing**: Efficient parallel generation with configurable batch sizes
- **Incremental Saving**: Prevent data loss with incremental checkpointing
- **Resource Management**: Intelligent memory and GPU utilization

#### 5.3.3 Quality and Reliability
- **Comprehensive Validation**: Multi-layer quality control system
- **Error Handling**: Robust error handling with retry mechanisms
- **Progress Tracking**: Real-time monitoring of generation progress
- **Quality Monitoring**: Continuous quality assessment with alerting

#### 5.3.4 Maintainability and Testing
- **Type Safety**: Comprehensive type hints and runtime validation
- **Configuration Management**: Centralized, validated configuration system
- **Dependency Injection**: Loose coupling for easy testing and mocking
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 6. Usage Examples and Integration

### 6.1 Complete Pipeline Execution

```python
# Example: Complete dataset generation pipeline
from silly_llm.data.generation import StoryGenerationOrchestrator
from silly_llm.core.config import load_config
from silly_llm.core.container import DIContainer

async def main():
    # Load configuration
    config = load_config("configs/story_generation.yaml")

    # Setup dependency injection container
    container = DIContainer()
    container.register_config(config)

    # Get orchestrator from container
    orchestrator = container.get(StoryGenerationOrchestrator)

    # Generate dataset
    result = await orchestrator.generate_dataset()

    print(f"Generated {result.statistics.total_stories} stories")
    print(f"Total words: {result.statistics.total_words}")
    print(f"Average quality score: {result.statistics.quality_metrics['average_score']}")

# Run the pipeline
asyncio.run(main())
```

### 6.2 Custom Provider Integration

```python
# Example: Using custom LLM provider
class CustomAPIProvider(LLMProvider):
    """Custom API provider for proprietary models."""

    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    async def generate(self, prompt: str, **kwargs) -> str:
        # Custom API integration
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"prompt": prompt, **kwargs}
            ) as response:
                result = await response.json()
                return result["generated_text"]

# Register custom provider
container.register_provider(LLMProvider, CustomAPIProvider)
```

### 6.3 Command-Line Interface

```bash
# Generate vocabulary
python -m silly_llm.data.vocabulary \
    --config configs/vocabulary.yaml \
    --output-path ./data/vocabulary.json

# Generate stories
python -m silly_llm.data.generation \
    --config configs/story_generation.yaml \
    --vocabulary-path ./data/vocabulary.json \
    --output-dir ./generated_stories

# Monitor generation progress
python -m silly_llm.data.monitor \
    --generation-dir ./generated_stories \
    --dashboard-port 8080
```

## 7. Testing and Validation Framework

### 7.1 Unit Testing Structure

```python
# Example test structure
class TestVocabularyManager:
    """Test suite for vocabulary management."""

    @pytest.fixture
    def mock_llm_provider(self):
        return MockLLMProvider()

    @pytest.fixture
    def vocab_manager(self, mock_llm_provider):
        config = VocabularyConfig(target_word_count=100)
        return VocabularyManager(mock_llm_provider, config)

    async def test_vocabulary_generation(self, vocab_manager):
        """Test basic vocabulary generation."""
        vocabulary = await vocab_manager.generate_vocabulary()

        assert len(vocabulary.get_all_words()) <= 100
        assert len(vocabulary.get_words_by_category(WordCategory.NOUN)) > 0
        assert len(vocabulary.get_words_by_category(WordCategory.VERB)) > 0
        assert len(vocabulary.get_words_by_category(WordCategory.ADJECTIVE)) > 0

    async def test_vocabulary_validation(self, vocab_manager):
        """Test vocabulary validation."""
        word = VocabularyWord(
            word="inappropriate",
            category=WordCategory.ADJECTIVE,
            confidence_score=0.8,
            age_appropriateness=0.3,  # Too low
            complexity_score=0.9,
            source="test"
        )

        result = vocab_manager.validator.validate_word(word)
        assert not result.is_valid
```

### 7.2 Integration Testing

```python
class TestGenerationPipeline:
    """Integration tests for story generation pipeline."""

    @pytest.mark.integration
    async def test_end_to_end_generation(self):
        """Test complete generation pipeline."""
        config = load_config("configs/test_generation.yaml")
        container = DIContainer()
        container.register_config(config)

        orchestrator = container.get(StoryGenerationOrchestrator)
        result = await orchestrator.generate_dataset()

        assert result.statistics.total_stories > 0
        assert all(story.quality_scores["overall"] > 0.5 for story in result.stories)
```

## 8. Performance Optimization and Monitoring

### 8.1 Performance Metrics

```python
class PerformanceMonitor:
    """Monitor generation performance and resource usage."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def record_generation_time(self, story_count: int, elapsed_time: float):
        """Record generation performance."""
        stories_per_second = story_count / elapsed_time
        self.metrics["stories_per_second"].append(stories_per_second)

    def record_memory_usage(self):
        """Record current memory usage."""
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.metrics["memory_usage_mb"].append(memory_mb)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        return {
            "average_stories_per_second": np.mean(self.metrics["stories_per_second"]),
            "peak_memory_usage_mb": max(self.metrics["memory_usage_mb"]),
            "total_runtime": time.time() - self.start_time
        }
```

### 8.2 Resource Management

```python
class ResourceManager:
    """Manage computational resources during generation."""

    def __init__(self, config: ResourceConfig):
        self.config = config
        self.gpu_monitor = GPUMonitor()
        self.memory_monitor = MemoryMonitor()

    async def optimize_batch_size(self, current_batch_size: int) -> int:
        """Dynamically optimize batch size based on resource usage."""
        gpu_utilization = self.gpu_monitor.get_utilization()
        memory_usage = self.memory_monitor.get_usage_percentage()

        if gpu_utilization < 0.7 and memory_usage < 0.8:
            # Can increase batch size
            return min(current_batch_size + 4, self.config.max_batch_size)
        elif memory_usage > 0.9:
            # Need to decrease batch size
            return max(current_batch_size - 4, self.config.min_batch_size)

        return current_batch_size
```

## 9. Future Extensions and Roadmap

### 9.1 Planned Enhancements

1. **Multi-Modal Generation**: Support for generating stories with accompanying images
2. **Interactive Generation**: Real-time story generation with user feedback
3. **Curriculum Learning**: Progressive complexity in generated stories
4. **Cross-Lingual Support**: Generate stories in multiple languages
5. **Domain Adaptation**: Specialized story generation for different domains

### 9.2 Research Integration

1. **Novel Generation Strategies**: Integration of latest research in text generation
2. **Quality Assessment**: Advanced quality metrics using latest evaluation methods
3. **Efficiency Improvements**: Integration of model compression and acceleration techniques
4. **Personalization**: Adaptive story generation based on user preferences

This modular architecture provides a solid foundation for current needs while enabling future extensions and research integration.
