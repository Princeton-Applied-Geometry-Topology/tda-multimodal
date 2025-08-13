import pytest
import torch
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
import tempfile
import shutil

# Import dataset classes
from dataset.questions import QuestionAnswerDataset
from dataset.multi_task_questions import MultiTaskQuestionAnswerDataset
from dataset.wikipedia import WikipediaDataset
from dataset.graphs import RingGraphDataset, GridGraphDataset


class TestDatasetBase:
    """Base class for dataset tests with common functionality."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return AutoTokenizer.from_pretrained('gpt2')
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def check_dataset_structure(self, dataset, expected_length):
        """Check that dataset has expected structure and properties."""
        # Check length
        assert len(dataset) == expected_length
        
        # Check output keys
        expected_keys = ['input_ids', 'attention_mask', 'token_labels', 'text']
        assert set(dataset.output_keys) == set(expected_keys)
        
        # Check that cache is initialized
        assert hasattr(dataset, 'cache')
        assert set(dataset.cache.keys()) == set(expected_keys)
    
    def check_dataset_item(self, dataset, item_idx=0):
        """Check that dataset item has correct structure."""
        item = dataset[item_idx]
        
        # Check item keys
        expected_keys = ['input_ids', 'attention_mask', 'token_labels', 'text']
        assert set(item.keys()) == set(expected_keys)
        
        # Check tensor shapes and types
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        assert isinstance(item['token_labels'], list)
        assert isinstance(item['text'], str)
        
        # Check tensor shapes match
        assert item['input_ids'].shape == item['attention_mask'].shape
        assert len(item['token_labels']) == len(item['input_ids'])
        
        return item
    
    def check_save_functionality(self, dataset, temp_dir):
        """Check that dataset can save data correctly."""
        # Get some items to populate cache
        for i in range(min(3, len(dataset))):
            _ = dataset[i]
        
        # Save data
        dataset.save_run(temp_dir)
        
        # Check that files were created
        expected_files = ['input_ids.pt', 'attention_mask.pt', 'token_labels.npy', 'text.txt']
        for file_name in expected_files:
            assert (temp_dir / file_name).exists()
        
        # Check that saved data has correct format
        input_ids = torch.load(temp_dir / 'input_ids.pt')
        attention_mask = torch.load(temp_dir / 'attention_mask.pt')
        token_labels = np.load(temp_dir / 'token_labels.npy', allow_pickle=True)
        text = (temp_dir / 'text.txt').read_text()
        
        assert input_ids.shape[0] == min(3, len(dataset))
        assert attention_mask.shape[0] == min(3, len(dataset))
        assert len(token_labels) == min(3, len(dataset))
        assert len(text.split('\n')) == min(3, len(dataset))


class TestQuestionAnswerDataset(TestDatasetBase):
    """Test the QuestionAnswerDataset class."""
    
    def test_dataset_initialization(self, tokenizer):
        """Test that dataset initializes correctly."""
        # Mock data directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data file
            data_dir = Path(temp_dir) / 'data' / 'abstractive'
            data_dir.mkdir(parents=True)
            
            mock_data = [
                {"input": "What is the opposite of hot?", "output": "cold"},
                {"input": "What is the opposite of big?", "output": "small"},
                {"input": "What is the opposite of fast?", "output": "slow"}
            ]
            
            import json
            (data_dir / 'antonym.json').write_text(json.dumps(mock_data))
            
            # Change to temp directory
            import os
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                dataset = QuestionAnswerDataset(
                    n_examples=2,
                    n_seqs=5,
                    tokenizer=tokenizer,
                    question_type='abstractive',
                    task_name='antonym',
                    shuffle=False
                )
                
                self.check_dataset_structure(dataset, 5)
                
            finally:
                os.chdir(old_cwd)
    
    def test_dataset_item_structure(self, tokenizer):
        """Test that dataset items have correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data
            data_dir = Path(temp_dir) / 'data' / 'abstractive'
            data_dir.mkdir(parents=True)
            
            mock_data = [
                {"input": "What is the opposite of hot?", "output": "cold"},
                {"input": "What is the opposite of big?", "output": "small"}
            ]
            
            import json
            (data_dir / 'antonym.json').write_text(json.dumps(mock_data))
            
            import os
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                dataset = QuestionAnswerDataset(
                    n_examples=2,
                    n_seqs=3,
                    tokenizer=tokenizer,
                    question_type='abstractive',
                    task_name='antonym',
                    shuffle=False
                )
                
                self.check_dataset_item(dataset, 0)
                
            finally:
                os.chdir(old_cwd)


class TestMultiTaskQuestionAnswerDataset(TestDatasetBase):
    """Test the MultiTaskQuestionAnswerDataset class."""
    
    def test_multi_task_dataset_initialization(self, tokenizer):
        """Test that multi-task dataset initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data files
            abs_dir = Path(temp_dir) / 'data' / 'abstractive'
            ext_dir = Path(temp_dir) / 'data' / 'extractive'
            abs_dir.mkdir(parents=True)
            ext_dir.mkdir(parents=True)
            
            mock_abs_data = [
                {"input": "What is the opposite of hot?", "output": "cold"},
                {"input": "What is the opposite of big?", "output": "small"}
            ]
            
            mock_ext_data = [
                {"input": "Is this an animal: dog", "output": "yes"},
                {"input": "Is this an animal: car", "output": "no"}
            ]
            
            import json
            (abs_dir / 'antonym.json').write_text(json.dumps(mock_abs_data))
            (ext_dir / 'animal_classification.json').write_text(json.dumps(mock_ext_data))
            
            import os
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                dataset = MultiTaskQuestionAnswerDataset(
                    n_examples_per_task=1,
                    n_seqs=3,
                    tokenizer=tokenizer,
                    question_types=['abstractive', 'extractive'],
                    task_names=['antonym', 'animal_classification'],
                    shuffle=False
                )
                
                self.check_dataset_structure(dataset, 3)
                
            finally:
                os.chdir(old_cwd)


class TestWikipediaDataset(TestDatasetBase):
    """Test the WikipediaDataset class."""
    
    def test_wikipedia_dataset_initialization(self, tokenizer):
        """Test that Wikipedia dataset initializes correctly."""
        # Use small parameters for testing
        dataset = WikipediaDataset(
            tokenizer=tokenizer,
            seq_len=50,
            n_seqs=3,
            max_tokens=100,
            cache_dir='/tmp/test_cache',
            shuffle=False
        )
        
        self.check_dataset_structure(dataset, 3)
    
    def test_wikipedia_item_structure(self, tokenizer):
        """Test that Wikipedia dataset items have correct structure."""
        dataset = WikipediaDataset(
            tokenizer=tokenizer,
            seq_len=50,
            n_seqs=3,
            max_tokens=100,
            cache_dir='/tmp/test_cache',
            shuffle=False
        )
        
        item = self.check_dataset_item(dataset, 0)
        
        # Check that all tokens are labeled as 'content'
        assert all(label == 'content' for label in item['token_labels'])
    
    def test_wikipedia_shuffle_functionality(self, tokenizer):
        """Test that Wikipedia dataset shuffle works correctly."""
        dataset_no_shuffle = WikipediaDataset(
            tokenizer=tokenizer,
            seq_len=50,
            n_seqs=2,
            max_tokens=100,
            cache_dir='/tmp/test_cache',
            shuffle=False
        )
        
        dataset_shuffle = WikipediaDataset(
            tokenizer=tokenizer,
            seq_len=50,
            n_seqs=2,
            max_tokens=100,
            cache_dir='/tmp/test_cache',
            shuffle=True
        )
        
        # Both should have same length
        assert len(dataset_no_shuffle) == len(dataset_shuffle)
        
        # Get items from both
        item_no_shuffle = dataset_no_shuffle[0]
        item_shuffle = dataset_shuffle[0]
        
        # Allow up to 5% difference in token count
        len1 = len(item_no_shuffle['input_ids'])
        len2 = len(item_shuffle['input_ids'])
        assert abs(len1 - len2) / max(len1, len2) < 0.05  # Allow up to 5% difference
    
    def test_wikipedia_save_functionality(self, tokenizer, temp_dir):
        """Test that Wikipedia dataset can save data correctly."""
        dataset = WikipediaDataset(
            tokenizer=tokenizer,
            seq_len=50,
            n_seqs=3,
            max_tokens=100,
            cache_dir='/tmp/test_cache',
            shuffle=False
        )
        
        self.check_save_functionality(dataset, temp_dir)


class TestRingGraphDataset(TestDatasetBase):
    """Test the RingGraphDataset class."""
    
    def test_ring_graph_dataset_initialization(self, tokenizer):
        """Test that ring graph dataset initializes correctly."""
        dataset = RingGraphDataset(
            tokenizer=tokenizer,
            n_nodes=10,
            path_length=5,
            n_seqs=3,
            seq_len=100,
            shuffle=False
        )
        
        self.check_dataset_structure(dataset, 3)
        
        # Check that graph was created
        assert hasattr(dataset, 'graph')
        assert dataset.graph.number_of_nodes() == 10
    
    def test_ring_graph_item_structure(self, tokenizer):
        """Test that ring graph dataset items have correct structure."""
        dataset = RingGraphDataset(
            tokenizer=tokenizer,
            n_nodes=10,
            path_length=5,
            n_seqs=3,
            seq_len=100,
            shuffle=False
        )
        
        item = self.check_dataset_item(dataset, 0)
        
        # Check that all tokens are labeled as 'path'
        assert all(label == 'path' for label in item['token_labels'])
        
        # Check that text contains path information
        assert 'Path:' in item['text']
        assert 'Node_' in item['text']
    
    def test_ring_graph_save_functionality(self, tokenizer, temp_dir):
        """Test that ring graph dataset can save data correctly."""
        dataset = RingGraphDataset(
            tokenizer=tokenizer,
            n_nodes=10,
            path_length=5,
            n_seqs=3,
            seq_len=100,
            shuffle=False
        )
        
        self.check_save_functionality(dataset, temp_dir)


class TestGridGraphDataset(TestDatasetBase):
    """Test the GridGraphDataset class."""
    
    def test_grid_graph_dataset_initialization(self, tokenizer):
        """Test that grid graph dataset initializes correctly."""
        dataset = GridGraphDataset(
            tokenizer=tokenizer,
            grid_size=3,
            path_length=5,
            n_seqs=3,
            seq_len=100,
            shuffle=False
        )
        
        self.check_dataset_structure(dataset, 3)
        
        # Check that graph was created
        assert hasattr(dataset, 'graph')
        assert dataset.graph.number_of_nodes() == 9  # 3x3 grid
    
    def test_grid_graph_item_structure(self, tokenizer):
        """Test that grid graph dataset items have correct structure."""
        dataset = GridGraphDataset(
            tokenizer=tokenizer,
            grid_size=3,
            path_length=5,
            n_seqs=3,
            seq_len=100,
            shuffle=False
        )
        
        item = self.check_dataset_item(dataset, 0)
        
        # Check that all tokens are labeled as 'path'
        assert all(label == 'path' for label in item['token_labels'])
        
        # Check that text contains grid path information
        assert 'Grid Path:' in item['text']
        assert '(' in item['text'] and ')' in item['text']  # Coordinate format
    
    def test_grid_graph_save_functionality(self, tokenizer, temp_dir):
        """Test that grid graph dataset can save data correctly."""
        dataset = GridGraphDataset(
            tokenizer=tokenizer,
            grid_size=3,
            path_length=5,
            n_seqs=3,
            seq_len=100,
            shuffle=False
        )
        
        self.check_save_functionality(dataset, temp_dir)


class TestDatasetCompatibility:
    """Test that all datasets work with the main pipeline."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        return AutoTokenizer.from_pretrained('gpt2')
    
    def test_all_datasets_have_same_interface(self, tokenizer):
        """Test that all datasets have the same interface."""
        # Create instances of all datasets
        datasets = []
        
        # Wikipedia dataset
        wikipedia_dataset = WikipediaDataset(
            tokenizer=tokenizer,
            seq_len=50,
            n_seqs=2,
            max_tokens=100,
            cache_dir='/tmp/test_cache',
            shuffle=False
        )
        datasets.append(wikipedia_dataset)
        
        # Ring graph dataset
        ring_dataset = RingGraphDataset(
            tokenizer=tokenizer,
            n_nodes=10,
            path_length=5,
            n_seqs=2,
            seq_len=100,
            shuffle=False
        )
        datasets.append(ring_dataset)
        
        # Grid graph dataset
        grid_dataset = GridGraphDataset(
            tokenizer=tokenizer,
            grid_size=3,
            path_length=5,
            n_seqs=2,
            seq_len=100,
            shuffle=False
        )
        datasets.append(grid_dataset)
        
        # Check that all datasets have the same interface
        for dataset in datasets:
            # Check required attributes
            assert hasattr(dataset, 'output_keys')
            assert hasattr(dataset, 'cache')
            assert hasattr(dataset, 'save_run')
            
            # Check that __len__ and __getitem__ work
            assert len(dataset) == 2
            item = dataset[0]
            
            # Check that items have consistent structure
            expected_keys = ['input_ids', 'attention_mask', 'token_labels', 'text']
            assert set(item.keys()) == set(expected_keys)
            
            # Check tensor types
            assert isinstance(item['input_ids'], torch.Tensor)
            assert isinstance(item['attention_mask'], torch.Tensor)
            assert isinstance(item['token_labels'], list)
            assert isinstance(item['text'], str)


if __name__ == '__main__':
    pytest.main([__file__]) 