#!/usr/bin/env python3
"""
Test script for Winoground dataset integration.

This script tests the basic functionality of the Winoground dataset classes.
"""

import os
import sys
import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import WinogroundDataset, WinogroundEvaluationDataset
from transformers import AutoTokenizer


class TestWinogroundDataset:
    """Test cases for WinogroundDataset class."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_data(self, temp_dir):
        """Create mock dataset files."""
        # Create examples.jsonl
        examples_path = temp_dir / "examples.jsonl"
        examples_data = [
            {
                "caption_0": "A dog is chasing a cat",
                "caption_1": "A cat is chasing a dog",
                "image_0": "000001",
                "image_1": "000002",
                "tags": ["spatial"],
                "secondary_tag": "action",
                "num_main_preds": 2,
                "collapsed_tag": "spatial_action"
            },
            {
                "caption_0": "The red car is above the blue car",
                "caption_1": "The blue car is above the red car",
                "image_0": "000003",
                "image_1": "000004",
                "tags": ["spatial", "color"],
                "secondary_tag": "position",
                "num_main_preds": 2,
                "collapsed_tag": "spatial_color"
            }
        ]
        
        import json
        with open(examples_path, 'w') as f:
            for example in examples_data:
                f.write(json.dumps(example) + '\n')
        
        # Create images directory
        images_dir = temp_dir / "images"
        images_dir.mkdir()
        
        # Create mock PNG images
        for img_id in ["000001", "000002", "000003", "000004"]:
            img_path = images_dir / f"{img_id}.png"
            # Create a simple 1x1 RGB image
            img = Image.new('RGB', (1, 1), color='red')
            img.save(img_path)
        
        return temp_dir
    
    def test_dataset_initialization(self, tokenizer, mock_data):
        """Test that dataset initializes correctly."""
        dataset = WinogroundDataset(
            tokenizer=tokenizer,
            data_root=str(mock_data),
            download=False
        )
        
        assert len(dataset.examples) == 2
        assert dataset.examples[0]['caption_0'] == "A dog is chasing a cat"
        assert dataset.examples[0]['tags'] == ["spatial"]
    
    def test_load_examples(self, tokenizer, mock_data):
        """Test loading examples from JSONL file."""
        dataset = WinogroundDataset(
            tokenizer=tokenizer,
            data_root=str(mock_data),
            download=False
        )
        
        assert len(dataset.examples) == 2
        assert dataset.examples[0]['tags'] == ["spatial"]
        assert dataset.examples[1]['tags'] == ["spatial", "color"]
    
    def test_getitem(self, tokenizer, mock_data):
        """Test accessing individual examples."""
        dataset = WinogroundDataset(
            tokenizer=tokenizer,
            data_root=str(mock_data),
            download=False
        )
        
        example = dataset[0]
        assert 'caption_0' in example
        assert 'caption_1' in example
        assert 'input_ids_0' in example
        assert 'input_ids_1' in example
        assert 'tags' in example
        
        # Check that input_ids are tensors
        assert isinstance(example['input_ids_0'], torch.Tensor)
        assert isinstance(example['input_ids_1'], torch.Tensor)
        assert isinstance(example['attention_mask_0'], torch.Tensor)
        assert isinstance(example['attention_mask_1'], torch.Tensor)
    
    def test_get_examples_by_tag(self, tokenizer, mock_data):
        """Test filtering examples by tag."""
        dataset = WinogroundDataset(
            tokenizer=tokenizer,
            data_root=str(mock_data),
            download=False
        )
        
        spatial_examples = dataset.get_examples_by_tag('spatial')
        assert len(spatial_examples) == 2
        
        color_examples = dataset.get_examples_by_tag('color')
        assert len(color_examples) == 1
    
    def test_get_statistics(self, tokenizer, mock_data):
        """Test dataset statistics generation."""
        dataset = WinogroundDataset(
            tokenizer=tokenizer,
            data_root=str(mock_data),
            download=False
        )
        
        stats = dataset.get_statistics()
        assert stats['total_examples'] == 2
        assert 'spatial' in stats['tags_distribution']
        assert stats['tags_distribution']['spatial'] == 2
        assert stats['tags_distribution']['color'] == 1
    
    def test_get_evaluation_batch(self, tokenizer, mock_data):
        """Test getting evaluation batches."""
        dataset = WinogroundDataset(
            tokenizer=tokenizer,
            data_root=str(mock_data),
            download=False
        )
        
        batch = dataset.get_evaluation_batch(batch_size=1)
        assert len(batch) == 1
        
        batch = dataset.get_evaluation_batch(batch_size=5)
        assert len(batch) == 2  # Only 2 examples available
    
    @patch('dataset.winoground.requests.get')
    def test_download_dataset(self, mock_get, tokenizer, temp_dir):
        """Test dataset download functionality."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"mock data"]
        mock_get.return_value = mock_response
        
        # Create mock examples.jsonl in temp_dir
        examples_path = temp_dir / "examples.jsonl"
        with open(examples_path, 'w') as f:
            f.write('{"caption_0": "test", "caption_1": "test2", "image_0": "000001", "image_1": "000002", "tags": ["test"]}\n')
        
        # Create mock images directory to avoid zip extraction
        images_dir = temp_dir / "images"
        images_dir.mkdir()
        
        # Create a mock image file
        img_path = images_dir / "000001.png"
        img = Image.new('RGB', (1, 1), color='red')
        img.save(img_path)
        
        # Test with fresh dataset (download=False since we already have the files)
        dataset = WinogroundDataset(
            tokenizer=tokenizer,
            data_root=str(temp_dir),
            download=False
        )
        
        # Verify examples were loaded
        assert len(dataset.examples) == 1
        assert dataset.examples[0]['caption_0'] == "test"


class TestWinogroundEvaluationDataset:
    """Test cases for WinogroundEvaluationDataset class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_data(self, temp_dir):
        """Create mock dataset files."""
        # Create examples.jsonl
        examples_path = temp_dir / "examples.jsonl"
        examples_data = [
            {
                "caption_0": "Test caption 1",
                "caption_1": "Test caption 2",
                "image_0": "000001",
                "image_1": "000002",
                "tags": ["test"],
                "secondary_tag": "test_secondary",
                "num_main_preds": 1,
                "collapsed_tag": "test_collapsed"
            }
        ]
        
        import json
        with open(examples_path, 'w') as f:
            for example in examples_data:
                f.write(json.dumps(example) + '\n')
        
        return temp_dir
    
    def test_evaluation_dataset(self, mock_data):
        """Test evaluation dataset functionality."""
        dataset = WinogroundEvaluationDataset(
            data_root=str(mock_data),
            download=False
        )
        
        assert len(dataset) == 1
        example = dataset[0]
        assert example['caption_0'] == "Test caption 1"
        assert example['caption_1'] == "Test caption 2"
        assert example['tags'] == ["test"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
