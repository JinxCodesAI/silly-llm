"""LLM provider interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import logging

from .data_models import GenerationConfig, GeneratedStory, KShotExample

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_batch(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        """Generate responses for a batch of prompts."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and metadata."""
        pass


class TransformersProvider(LLMProvider):
    """Transformers-based LLM provider using efficient batched generation."""

    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        """Initialize the transformers provider.

        Args:
            model_name: Name of the model to load
            device: Device to use ("auto", "cuda", "cpu")
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_kwargs = kwargs
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of model and tokenizer."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "torch and transformers are required for TransformersProvider. "
                "Install them with: pip install torch transformers"
            ) from e

        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **self.model_kwargs
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if self.device != "auto" else "auto",
            trust_remote_code=True,
            **self.model_kwargs
        )

        self._initialized = True
        logger.info(f"Model loaded on device: {self.model.device}")
    
    async def generate_batch(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        """Generate responses for a batch of prompts using efficient batching."""
        if not prompts:
            return []

        # Ensure model is loaded
        self._ensure_initialized()

        import torch

        # Prepare batch inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Reasonable input limit
        ).to(self.model.device)
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": config.use_cache,
            "return_dict_in_generate": True,
            "output_scores": False
        }
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        generation_time = time.time() - start_time
        
        # Decode generated text
        generated_texts = []
        for i, output_ids in enumerate(outputs.sequences):
            # Find input length for this sequence
            input_length = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum().item()
            
            # Extract only the generated tokens
            generated_tokens = output_ids[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            generated_texts.append(generated_text.strip())
        
        logger.info(f"Generated {len(generated_texts)} responses in {generation_time:.2f}s")
        
        return generated_texts
    
    def generate_with_messages(self, messages_batch: List[List[Dict[str, str]]], config: GenerationConfig) -> List[str]:
        """Generate responses using chat template format."""
        # Ensure model is loaded
        self._ensure_initialized()

        # Convert messages to text using chat template
        texts_batch = []
        for messages in messages_batch:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking for Qwen models
            )
            texts_batch.append(text)

        # Use the regular batch generation
        import asyncio
        return asyncio.run(self.generate_batch(texts_batch, config))
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if not self._initialized:
            return 0.0
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3
        except ImportError:
            pass
        return 0.0

    def clear_memory(self):
        """Clear GPU memory cache."""
        if not self._initialized:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and metadata."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if self.model else "unknown",
            "supports_batching": True,
            "supports_chat_template": True,
            "max_batch_size": 32,  # Conservative estimate
            "memory_usage_gb": self.get_memory_usage()
        }


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without requiring torch/transformers."""

    def __init__(self, model_name: str = "mock-model", **kwargs):
        """Initialize mock provider."""
        self.model_name = model_name
        self.generation_count = 0
        self.tokenizer = None  # Mock providers don't have real tokenizers

    async def generate_batch(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        """Generate mock responses."""
        self.generation_count += len(prompts)

        # Generate simple mock stories
        mock_responses = []
        for i, prompt in enumerate(prompts):
            # Extract words from prompt if possible
            words = []
            if "containing words" in prompt:
                # Simple extraction - look for quoted words or common patterns
                import re
                word_matches = re.findall(r'"([^"]*)"', prompt)
                if not word_matches:
                    word_matches = re.findall(r'\b\w+\b', prompt.split("containing words")[1].split("\n")[0])
                words = word_matches[:3] if word_matches else ["cat", "happy", "sleep"]

            # Generate a simple story
            story = self._generate_mock_story(words, i)
            mock_responses.append(story)

        # Simulate some processing time
        import asyncio
        await asyncio.sleep(0.1)

        return mock_responses

    def _generate_mock_story(self, words: List[str], index: int) -> str:
        """Generate a simple mock story."""
        templates = [
            "Once upon a time, there was a little {0} who loved to {1}. Every night, the {0} would feel {2} and dream of wonderful adventures. The {0} lived in a cozy house with soft pillows and warm blankets. When bedtime came, the {0} would {1} around the room, feeling so {2} and excited. Then the {0} would snuggle into bed, close their eyes, and drift off to sleep with the sweetest dreams. The end.",

            "In a magical forest, a {2} {0} discovered how to {1}. All the forest animals were amazed and felt {2} too. The wise old owl hooted with joy, the rabbits clapped their paws, and the squirrels chattered excitedly. The {0} taught everyone how to {1}, and soon the whole forest was filled with {2} sounds. As the sun set behind the trees, all the animals gathered in a circle to {1} together. They felt so {2} and grateful for their new friend. They all lived happily ever after.",

            "Little Emma found a {0} that could {1}. She felt so {2} that she shared her discovery with all her friends. They gathered in the garden where the flowers bloomed and the butterflies danced. The {0} would {1} in the most amazing ways, making everyone laugh and feel {2}. Emma's friends were delighted and wanted to learn how to make their own {0} {1} too. Together they spent the afternoon playing and learning, feeling {2} and content. When it was time to go home, they promised to meet again tomorrow for more fun adventures."
        ]

        template = templates[index % len(templates)]

        # Use provided words or defaults
        word1 = words[0] if len(words) > 0 else "bunny"
        word2 = words[1] if len(words) > 1 else "dance"
        word3 = words[2] if len(words) > 2 else "happy"

        return template.format(word1, word2, word3)

    def get_memory_usage(self) -> float:
        """Get mock memory usage (always 0)."""
        return 0.0

    def clear_memory(self):
        """Mock memory clearing (no-op)."""
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Get mock provider capabilities."""
        return {
            "model_name": self.model_name,
            "device": "mock",
            "supports_batching": True,
            "supports_chat_template": True,
            "max_batch_size": 100,
            "memory_usage_gb": 0.0,
            "generation_count": self.generation_count
        }
