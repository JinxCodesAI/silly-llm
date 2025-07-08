"""LLM provider interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        self._load_model(**kwargs)
    
    def _load_model(self, **kwargs):
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **kwargs
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
            **kwargs
        )
        
        logger.info(f"Model loaded on device: {self.model.device}")
    
    async def generate_batch(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        """Generate responses for a batch of prompts using efficient batching."""
        if not prompts:
            return []
        
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
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def clear_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
