"""Tests for custom validation system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ..validation.base import BaseValidator, CustomValidationResult
from ..validation.quality_validator import QualityValidator
from ...common.llm_providers import MockLLMProvider
from ...common.data_models import GenerationConfig


class TestCustomValidationResult:
    """Test CustomValidationResult model."""

    def test_validation_result_creation(self):
        """Test creating CustomValidationResult."""
        result = CustomValidationResult(
            is_valid=True,
            score=0.8,
            details={"test": "data"},
            reasoning="Test reasoning"
        )
        
        assert result.is_valid is True
        assert result.score == 0.8
        assert result.details == {"test": "data"}
        assert result.reasoning == "Test reasoning"
    
    def test_validation_result_score_bounds(self):
        """Test score validation bounds."""
        # Valid scores
        CustomValidationResult(is_valid=True, score=0.0)
        CustomValidationResult(is_valid=True, score=1.0)
        CustomValidationResult(is_valid=True, score=0.5)

        # Invalid scores should raise validation error
        with pytest.raises(ValueError):
            CustomValidationResult(is_valid=True, score=-0.1)

        with pytest.raises(ValueError):
            CustomValidationResult(is_valid=True, score=1.1)


class TestBaseValidator:
    """Test BaseValidator abstract class."""
    
    def test_base_validator_initialization(self):
        """Test BaseValidator initialization."""
        provider = MockLLMProvider()
        config = {"test": "config"}
        
        # Create a concrete implementation for testing
        class TestValidator(BaseValidator):
            def get_validation_prompt(self, story_content: str) -> str:
                return f"Validate: {story_content}"

            def parse_validation_response(self, response: str) -> CustomValidationResult:
                return CustomValidationResult(is_valid=True, score=1.0)
        
        validator = TestValidator(provider, config)
        assert validator.provider == provider
        assert validator.config == config
    
    def test_parse_validation_response_answer_yes(self):
        """Test parsing ANSWER:YES response."""
        provider = MockLLMProvider()
        
        class TestValidator(BaseValidator):
            def get_validation_prompt(self, story_content: str) -> str:
                return f"Validate: {story_content}"

            def parse_validation_response(self, response: str) -> CustomValidationResult:
                return CustomValidationResult(is_valid=True, score=1.0)
        
        validator = TestValidator(provider, {})
        
        result = validator.parse_validation_response("Think step by step... ANSWER:YES")
        assert result.is_valid is True
        assert result.score == 1.0
    
    def test_parse_validation_response_answer_no(self):
        """Test parsing ANSWER:NO response."""
        provider = MockLLMProvider()
        
        class TestValidator(BaseValidator):
            async def validate(self, story_content: str) -> CustomValidationResult:
                return CustomValidationResult(is_valid=True, score=1.0)
            
            def get_validation_prompt(self, story_content: str) -> str:
                return f"Validate: {story_content}"
        
        validator = TestValidator(provider, {})
        
        result = validator.parse_validation_response("Analysis... ANSWER:NO")
        assert result.is_valid is False
        assert result.score == 0.0


class TestQualityValidator:
    """Test QualityValidator implementation."""
    
    def test_quality_validator_initialization(self):
        """Test QualityValidator initialization."""
        provider = MockLLMProvider()
        config = {
            "generation": {
                "max_new_tokens": 64,
                "temperature": 0.2
            }
        }
        
        validator = QualityValidator(provider, config)
        assert validator.provider == provider
        assert validator.generation_config.max_new_tokens == 64
        assert validator.generation_config.temperature == 0.2
    
    def test_get_validation_prompt(self):
        """Test validation prompt generation."""
        provider = MockLLMProvider()
        validator = QualityValidator(provider, {})
        
        story = "Once upon a time, there was a cat."
        prompt = validator.get_validation_prompt(story)
        
        assert "Does provided story contain only English words" in prompt
        assert "ANSWER:NO or ANSWER:YES" in prompt
        assert story in prompt
    
    @pytest.mark.asyncio
    async def test_validate_success(self):
        """Test successful validation."""
        provider = MockLLMProvider()
        # Mock the provider to return a positive validation response
        provider.generate_batch = AsyncMock(return_value=["Analysis complete. ANSWER:YES"])
        
        validator = QualityValidator(provider, {})
        
        story = "Once upon a time, there was a happy cat who loved to play."
        result = await validator.validate(story)
        
        assert result.is_valid is True
        assert result.score == 1.0
        assert "story_length" in result.details
        assert result.details["provider_model"] == provider.model_name
    
    @pytest.mark.asyncio
    async def test_validate_failure(self):
        """Test validation failure."""
        provider = MockLLMProvider()
        # Mock the provider to return a negative validation response
        provider.generate_batch = AsyncMock(return_value=["Analysis shows issues. ANSWER:NO"])
        
        validator = QualityValidator(provider, {})
        
        story = "Some story with issues."
        result = await validator.validate(story)
        
        assert result.is_valid is False
        assert result.score == 0.0
        assert "ANSWER:NO" in result.details["pattern_found"]
    
    @pytest.mark.asyncio
    async def test_validate_error_handling(self):
        """Test validation error handling."""
        provider = MockLLMProvider()
        # Mock the provider to raise an exception
        provider.generate_batch = AsyncMock(side_effect=Exception("Test error"))
        
        validator = QualityValidator(provider, {})
        
        story = "Test story"
        result = await validator.validate(story)
        
        assert result.is_valid is False
        assert result.score == 0.0
        assert "error" in result.details
        assert "Test error" in result.details["error"]
    
    @pytest.mark.asyncio
    async def test_validate_no_response(self):
        """Test validation with no response."""
        provider = MockLLMProvider()
        # Mock the provider to return empty response
        provider.generate_batch = AsyncMock(return_value=[])
        
        validator = QualityValidator(provider, {})
        
        story = "Test story"
        result = await validator.validate(story)
        
        assert result.is_valid is False
        assert result.score == 0.0
        assert "No response from validation provider" in result.details["error"]


if __name__ == "__main__":
    pytest.main([__file__])
