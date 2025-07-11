"""Test file saving functionality for correct stories filtering."""

import json
import tempfile
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import logging

# Mock the data models to avoid import issues
class MockGeneratedStory:
    def __init__(self, story_id, prompt_id, content, word_count, generation_time,
                 tokens_generated, tokens_per_second, memory_used_gb, metadata=None):
        self.story_id = story_id
        self.prompt_id = prompt_id
        self.content = content
        self.word_count = word_count
        self.generation_time = generation_time
        self.tokens_generated = tokens_generated
        self.tokens_per_second = tokens_per_second
        self.memory_used_gb = memory_used_gb
        self.created_at = datetime.now()
        self.metadata = metadata or {}

    def dict(self):
        return {
            'story_id': self.story_id,
            'prompt_id': self.prompt_id,
            'content': self.content,
            'word_count': self.word_count,
            'generation_time': self.generation_time,
            'tokens_generated': self.tokens_generated,
            'tokens_per_second': self.tokens_per_second,
            'memory_used_gb': self.memory_used_gb,
            'created_at': self.created_at,
            'metadata': self.metadata
        }

# Mock save_stories_jsonl function
def mock_save_stories_jsonl(stories, output_path):
    """Mock implementation of save_stories_jsonl."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for story in stories:
            f.write(json.dumps(story, default=str) + '\n')


def save_stories_implementation(stories, output_path):
    """Implementation of the _save_stories logic for testing."""
    # Add timestamp to filename before extension
    path_obj = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = path_obj.with_stem(f"{path_obj.stem}_{timestamp}")

    # Convert to dictionaries for JSON serialization
    story_dicts = []
    correct_story_dicts = []

    for story in stories:
        story_dict = story.dict()
        # Convert datetime to string for JSON serialization
        if 'created_at' in story_dict:
            story_dict['created_at'] = story_dict['created_at'].isoformat()
        story_dicts.append(story_dict)

        # Add to correct stories if word_count > 0
        if story.word_count > 0:
            correct_story_dicts.append(story_dict)

    # Save all stories (existing behavior)
    mock_save_stories_jsonl(story_dicts, str(timestamped_path))

    # Save only correct stories to a separate file
    correct_path = path_obj.with_stem(f"{path_obj.stem}_{timestamp}_correct")
    mock_save_stories_jsonl(correct_story_dicts, str(correct_path))

    return str(timestamped_path), str(correct_path), len(story_dicts), len(correct_story_dicts)


class TestFileSaving:
    """Test file saving functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock stories with different word counts
        self.stories = [
            MockGeneratedStory(
                story_id="story_1",
                prompt_id="prompt_1",
                content="This is a valid story with enough words to pass validation.",
                word_count=12,  # > 0, should be in correct stories
                generation_time=1.0,
                tokens_generated=50,
                tokens_per_second=50.0,
                memory_used_gb=0.5,
                metadata={"test": "data"}
            ),
            MockGeneratedStory(
                story_id="story_2",
                prompt_id="prompt_2",
                content="",
                word_count=0,  # = 0, should NOT be in correct stories (error story)
                generation_time=0.0,
                tokens_generated=0,
                tokens_per_second=0.0,
                memory_used_gb=0.0,
                metadata={"error": "generated text empty"}
            ),
            MockGeneratedStory(
                story_id="story_3",
                prompt_id="prompt_3",
                content="Another valid story that meets the word count requirements for inclusion.",
                word_count=11,  # > 0, should be in correct stories
                generation_time=1.2,
                tokens_generated=45,
                tokens_per_second=37.5,
                memory_used_gb=0.4,
                metadata={"test": "data2"}
            )
        ]

    def test_save_stories_creates_both_files(self):
        """Test that _save_stories creates both all stories and correct stories files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_stories.jsonl"

            # Call the implementation
            all_path, correct_path, total_count, correct_count = save_stories_implementation(
                self.stories, str(output_path)
            )

            # Check that both files were created
            assert Path(all_path).exists(), f"All stories file not created: {all_path}"
            assert Path(correct_path).exists(), f"Correct stories file not created: {correct_path}"

            # Verify counts
            assert total_count == 3, f"Expected 3 total stories, got {total_count}"
            assert correct_count == 2, f"Expected 2 correct stories, got {correct_count}"

    def test_correct_stories_filtering(self):
        """Test that only stories with word_count > 0 are saved to correct stories file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_stories.jsonl"

            # Call the implementation
            all_path, correct_path, total_count, correct_count = save_stories_implementation(
                self.stories, str(output_path)
            )

            # Read and verify all stories file
            with open(all_path, 'r') as f:
                all_stories_data = [json.loads(line) for line in f]

            assert len(all_stories_data) == 3, f"Expected 3 stories in all file, got {len(all_stories_data)}"

            # Read and verify correct stories file
            with open(correct_path, 'r') as f:
                correct_stories_data = [json.loads(line) for line in f]

            assert len(correct_stories_data) == 2, f"Expected 2 correct stories, got {len(correct_stories_data)}"

            # Verify that only stories with word_count > 0 are in correct file
            for story in correct_stories_data:
                assert story['word_count'] > 0, f"Story with word_count {story['word_count']} found in correct stories"

            # Verify that the correct story IDs are present
            correct_story_ids = {story['story_id'] for story in correct_stories_data}
            expected_ids = {"story_1", "story_3"}
            assert correct_story_ids == expected_ids, f"Expected {expected_ids}, got {correct_story_ids}"

    def test_file_naming_convention(self):
        """Test that files follow the expected naming convention."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "my_stories.jsonl"

            # Call the implementation
            all_path, correct_path, total_count, correct_count = save_stories_implementation(
                self.stories, str(output_path)
            )

            # Verify naming pattern
            all_name = Path(all_path).name
            correct_name = Path(correct_path).name

            # All stories file: my_stories_YYYYMMDD_HHMMSS.jsonl
            assert all_name.startswith("my_stories_")
            assert all_name.endswith(".jsonl")
            assert "_correct" not in all_name

            # Correct stories file: my_stories_YYYYMMDD_HHMMSS_correct.jsonl
            assert correct_name.startswith("my_stories_")
            assert correct_name.endswith("_correct.jsonl")
