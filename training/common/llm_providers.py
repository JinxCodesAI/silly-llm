"""LLM provider interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import logging
import os
import asyncio

from .data_models import GenerationConfig, GeneratedStory, KShotExample, LLMRequest

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        """Generate responses for a batch of requests."""
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

        # Prepare model kwargs with flash attention
        model_kwargs = (self.model_kwargs or {}).copy()
        model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if self.device != "auto" else "auto",
            trust_remote_code=True,
            **model_kwargs
        )

        self._initialized = True
        logger.info(f"Model loaded on device: {self.model.device}")
    
    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        """Generate responses for a batch of requests using efficient batching."""
        if not requests:
            return []

        # Ensure model is loaded
        self._ensure_initialized()

        import torch
        import gc

        # Initialize variables for cleanup
        inputs = None
        outputs = None
        generated_texts = []

        try:
            # Convert LLMRequest objects to text format
            texts_batch = []
            for request in requests:
                messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

                if self.tokenizer.chat_template:
                    try:
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                    except Exception as e:
                        logger.warning(f"Chat template failed, using fallback: {e}")
                        # Fallback to simple concatenation
                        text = request.to_simple_prompt()
                else:
                    # Fallback to simple concatenation
                    text = request.to_simple_prompt()

                texts_batch.append(text)

            # Prepare batch inputs
            inputs = self.tokenizer(
                texts_batch,
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
            for i, output_ids in enumerate(outputs.sequences):
                # Find input length for this sequence
                input_length = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum().item()

                # Extract only the generated tokens
                generated_tokens = output_ids[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                generated_texts.append(generated_text.strip())

            logger.info(f"Generated {len(generated_texts)} responses in {generation_time:.2f}s")

            return generated_texts

        finally:
            # Explicit cleanup of tensors and variables
            if inputs is not None:
                del inputs
            if outputs is not None:
                del outputs

            # Clear intermediate variables
            if 'texts_batch' in locals():
                del texts_batch
            if 'generation_kwargs' in locals():
                del generation_kwargs

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    

    
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
        """Clear GPU memory cache and force garbage collection."""
        if not self._initialized:
            return
        try:
            import torch
            import gc

            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Reset peak memory stats for better tracking
                torch.cuda.reset_peak_memory_stats()
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()

            # Force garbage collection to free Python objects
            gc.collect()

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

    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        """Generate mock responses."""
        self.generation_count += len(requests)

        # Generate simple mock stories
        mock_responses = []
        for i, request in enumerate(requests):
            # Analyze k-shot examples for better mock responses
            k_shot_context = [msg for msg in request.messages if msg.role == "assistant"]
            current_prompt = request.messages[-1].content if request.messages else ""

            # Extract words from prompt if possible
            words = []
            if "containing" in current_prompt and "words" in current_prompt:
                # Simple extraction - look for the pattern "containing X words word1 word2 word3"
                import re
                # Look for pattern like "containing 3 English words word1 word2 word3"
                pattern = r'containing\s+\d+\s+\w*\s*words\s+([^\n]+)'
                match = re.search(pattern, current_prompt, re.IGNORECASE)
                if match:
                    words_line = match.group(1).strip()
                    # Split by whitespace and take first 3 words
                    word_matches = words_line.split()
                    words = [w for w in word_matches[:3] if w and len(w) > 1 and w.isalpha()]

                if not words:
                    # Fallback: look for quoted words or common patterns
                    word_matches = re.findall(r'"([^"]*)"', current_prompt)
                    if not word_matches:
                        # Try to extract words after "containing words"
                        if "containing words" in current_prompt:
                            after_containing = current_prompt.split("containing words")[1].split("\n")[0]
                            word_matches = re.findall(r'\b[a-zA-Z]+\b', after_containing)
                    words = [w for w in word_matches[:3] if w and len(w) > 1] or ["cat", "happy", "sleep"]

            if not words:
                words = ["cat", "happy", "sleep"]

            # Generate a mock story considering k-shot context
            story = self._generate_mock_story_with_context(words, i, k_shot_context)
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

    def _generate_mock_story_with_context(self, words: List[str], index: int, k_shot_context: List) -> str:
        """Generate a mock story considering k-shot context."""
        # If we have k-shot examples, try to mimic their style
        if k_shot_context:
            # Analyze the k-shot examples for patterns
            context_length = sum(len(msg.content.split()) for msg in k_shot_context)
            avg_length = context_length // len(k_shot_context) if k_shot_context else 50

            # Generate a story with similar length characteristics
            base_story = self._generate_mock_story(words, index)

            # Adjust length to match k-shot examples
            if avg_length < 30:
                # Shorter story
                sentences = base_story.split('. ')
                return '. '.join(sentences[:2]) + '.'
            elif avg_length > 80:
                # Longer story
                return base_story + " And they all lived happily ever after in their magical world."
            else:
                return base_story
        else:
            # No k-shot context, use regular generation
            return self._generate_mock_story(words, index)

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


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible API provider for remote model inference."""

    def __init__(self, model_name: str, api_base_url: str = "https://openrouter.ai/api/v1", **kwargs):
        """Initialize OpenAI-compatible provider.

        Args:
            model_name: Name of the model to use
            api_base_url: Base URL for the API (default: OpenAI)
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = os.getenv('AI_API_KEY')
        self.generation_count = 0

        if not self.api_key:
            raise ValueError(
                "AI_API_KEY environment variable is required for OpenAI-compatible provider. "
                "Set it with: export AI_API_KEY=your_api_key"
            )

        # Validate API base URL
        if not self.api_base_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid API base URL: {self.api_base_url}")

        logger.info(f"Initialized OpenAI-compatible provider with model: {self.model_name}")
        logger.info(f"API base URL: {self.api_base_url}")

    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        """Generate responses for a batch of requests using OpenAI-compatible API."""
        if not requests:
            return []

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for OpenAI-compatible provider. "
                "Install it with: pip install httpx"
            ) from e

        logger.info(f"Generating {len(requests)} responses using OpenAI-compatible API")

        # Process requests in a loop to simulate batch generation
        responses = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, request in enumerate(requests):
                try:
                    response = await self._generate_single(client, request, config)
                    responses.append(response)
                    logger.debug(f"Generated response {i+1}/{len(requests)}")

                    # Small delay to avoid rate limiting
                    if i < len(requests) - 1:
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Failed to generate response {i+1}: {e}")
                    # Add empty response to maintain batch size
                    responses.append("")

        self.generation_count += len(requests)
        logger.info(f"Completed batch generation: {len([r for r in responses if r])} successful")

        return responses

    async def _generate_single(self, client: "httpx.AsyncClient", request: LLMRequest, config: GenerationConfig) -> str:
        """Generate a single response using the API."""
        # Convert LLMRequest messages directly to OpenAI format - FIXES THE CRITICAL BUG!
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        print("max_tokens", config.max_new_tokens)
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,  # ← PROPER K-SHOT SUPPORT!
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": False
        }

        # Add optional parameters if they're supported
        if hasattr(config, 'repetition_penalty') and config.repetition_penalty != 1.0:
            payload["frequency_penalty"] = (config.repetition_penalty - 1.0) * 0.5

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Make the API request
        response = await client.post(
            f"{self.api_base_url}/chat/completions",
            json=payload,
            headers=headers
        )

        if response.status_code != 200:
            error_msg = f"API request failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Parse the response
        result = response.json()

        if "choices" not in result or not result["choices"]:
            raise Exception("No choices in API response")

        content = result["choices"][0]["message"]["content"]
        return content.strip()

    def get_memory_usage(self) -> float:
        """Get memory usage (always 0 for API providers)."""
        return 0.0

    def clear_memory(self):
        """Clear memory (no-op for API providers)."""
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and metadata."""
        return {
            "model_name": self.model_name,
            "device": "api",
            "supports_batching": True,
            "supports_chat_template": True,
            "max_batch_size": 100,  # Can be adjusted based on rate limits
            "memory_usage_gb": 0.0,
            "api_base_url": self.api_base_url,
            "generation_count": self.generation_count
        }
