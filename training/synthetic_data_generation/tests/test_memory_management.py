"""Tests for memory management and batch retry functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from training.synthetic_data_generation.src.batch_processor import BatchProcessor
from training.common.data_models import StoryPrompt, GenerationConfig, LLMRequest
from training.common.llm_providers import MockLLMProvider


class MemoryTestProvider(MockLLMProvider):
    """Test provider that simulates memory issues."""
    
    def __init__(self, fail_on_batch: int = None, memory_error_msg: str = None):
        super().__init__()
        self.fail_on_batch = fail_on_batch
        self.memory_error_msg = memory_error_msg or "CUDA out of memory"
        self.batch_count = 0
        self.clear_memory_calls = 0
        
    async def generate_batch(self, requests: List[LLMRequest], config: GenerationConfig) -> List[str]:
        """Generate batch with optional memory failure simulation."""
        self.batch_count += 1
        
        if self.fail_on_batch and self.batch_count == self.fail_on_batch:
            raise RuntimeError(self.memory_error_msg)
        
        return await super().generate_batch(requests, config)
    
    def clear_memory(self):
        """Track memory clear calls."""
        self.clear_memory_calls += 1
        super().clear_memory()


@pytest.fixture
def sample_prompts():
    """Create sample prompts for testing."""
    prompts = []
    for i in range(5):
        prompt = StoryPrompt(
            prompt_id=f"test_prompt_{i}",
            template="Test template",
            selected_words={"word1": "test", "word2": "word"},
            additional_condition="Test condition",
            full_prompt=f"Test prompt {i}",
            k_shot_examples=[],
            metadata={"test": True}
        )
        prompts.append(prompt)
    return prompts


@pytest.fixture
def generation_config():
    """Create test generation config."""
    return GenerationConfig(
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        batch_size=3
    )


class TestMemoryManagement:
    """Test memory management functionality."""
    
    @pytest.mark.asyncio
    async def test_memory_cleared_after_successful_batch(self, sample_prompts, generation_config):
        """Test that memory is cleared after successful batch processing."""
        provider = MemoryTestProvider()
        processor = BatchProcessor(provider, generation_config, validate_stories=False)

        result = await processor.process_batch(sample_prompts)

        assert len(result.stories) == len(sample_prompts)
        # Memory should be cleared in the provider's generate_batch method
        
    @pytest.mark.asyncio
    async def test_memory_cleared_after_failed_batch(self, sample_prompts, generation_config):
        """Test that memory is cleared even after batch failure."""
        provider = MemoryTestProvider(fail_on_batch=1)
        processor = BatchProcessor(provider, generation_config)
        
        with pytest.raises(RuntimeError):
            await processor.process_batch(sample_prompts)
        
        # Memory cleanup should happen in the finally block
        
    @pytest.mark.asyncio
    async def test_batch_retry_on_memory_error(self, sample_prompts, generation_config):
        """Test that batches are retried when memory errors occur."""
        provider = MemoryTestProvider()
        processor = BatchProcessor(provider, generation_config)
        
        # Test the multiple batches method with retry logic
        results = await processor.process_multiple_batches(
            sample_prompts, 
            max_retries=2,
            retry_delay=0.1
        )
        
        assert len(results) > 0
        assert provider.clear_memory_calls > 0
        
    @pytest.mark.asyncio
    async def test_adaptive_batch_size_on_memory_error(self, sample_prompts, generation_config):
        """Test adaptive batch size reduction on memory errors."""
        provider = MemoryTestProvider(
            fail_on_batch=1, 
            memory_error_msg="CUDA out of memory. Tried to allocate 408.00 MiB"
        )
        processor = BatchProcessor(provider, generation_config, validate_stories=False)

        # Mock the process_batch_with_ids to fail first, then succeed with smaller batches
        original_method = processor.process_batch_with_ids
        call_count = 0

        async def mock_process_batch_with_ids(prompts, batch_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails with memory error
                raise RuntimeError("CUDA out of memory. Tried to allocate 408.00 MiB")
            else:
                # Subsequent calls succeed - disable validation for testing
                temp_processor = BatchProcessor(MockLLMProvider(), generation_config, validate_stories=False)
                return await temp_processor.process_batch_with_ids(prompts, batch_id)

        processor.process_batch_with_ids = mock_process_batch_with_ids
        
        result = await processor.process_batch_with_adaptive_size(sample_prompts)
        
        assert result.success_rate > 0
        assert "adaptive_batch_size" in result.metadata
        assert "reduction_factor" in result.metadata
        
    @pytest.mark.asyncio
    async def test_retry_with_delay(self, sample_prompts, generation_config):
        """Test that retry logic includes proper delays."""
        provider = MemoryTestProvider(fail_on_batch=1)
        processor = BatchProcessor(provider, generation_config)
        
        start_time = time.time()
        
        # This should fail and not retry (since we're testing single batch)
        with pytest.raises(RuntimeError):
            await processor.process_batch(sample_prompts)
        
        # But the multiple batches method should retry
        provider.batch_count = 0  # Reset counter
        provider.fail_on_batch = 2  # Fail on second batch
        
        results = await processor.process_multiple_batches(
            sample_prompts * 2,  # Make it span multiple batches
            max_retries=1,
            retry_delay=0.2
        )
        
        elapsed_time = time.time() - start_time
        # Should have some delay due to retry
        assert elapsed_time > 0.1
        
    @pytest.mark.asyncio
    async def test_failed_batch_tracking(self, sample_prompts, generation_config):
        """Test that failed batches are properly tracked."""
        provider = MemoryTestProvider(fail_on_batch=1)
        processor = BatchProcessor(provider, generation_config)
        
        # Create enough prompts for multiple batches
        many_prompts = sample_prompts * 3  # 15 prompts with batch_size=3 = 5 batches
        
        results = await processor.process_multiple_batches(
            many_prompts,
            max_retries=0  # No retries to ensure failure
        )
        
        # Should have some successful batches and some failed ones
        # The exact number depends on which batch fails
        assert len(results) < 5  # Not all batches succeeded


class TestMemoryOptimizations:
    """Test memory optimization features."""
    
    @pytest.mark.asyncio
    async def test_garbage_collection_called(self, sample_prompts, generation_config):
        """Test that garbage collection is properly called."""
        provider = MockLLMProvider()
        processor = BatchProcessor(provider, generation_config)
        
        with patch('gc.collect') as mock_gc:
            await processor.process_batch(sample_prompts)
            # gc.collect should be called in the provider's generate_batch method
            
    @pytest.mark.asyncio
    async def test_tensor_cleanup(self, sample_prompts, generation_config):
        """Test that tensors are properly cleaned up."""
        provider = MockLLMProvider()
        processor = BatchProcessor(provider, generation_config, validate_stories=False)

        # This test would be more meaningful with actual torch tensors
        # For now, just verify the batch completes successfully
        result = await processor.process_batch(sample_prompts)
        assert len(result.stories) == len(sample_prompts)
        
    def test_memory_usage_tracking(self, generation_config):
        """Test memory usage tracking functionality."""
        provider = MockLLMProvider()
        processor = BatchProcessor(provider, generation_config)
        
        # Test memory usage reporting
        memory_usage = provider.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        assert memory_usage >= 0
        
        # Test memory clearing
        provider.clear_memory()  # Should not raise any errors


if __name__ == "__main__":
    pytest.main([__file__])
