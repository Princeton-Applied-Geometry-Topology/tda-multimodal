import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA before importing torch
import unittest
import torch
from pathlib import Path
import json
from transformers import AutoTokenizer
from dataset.dataset_utils import ICLTokenLabeler

# Force CPU usage
torch.set_default_device('cpu')

class TestTokenLabelExtraction(unittest.TestCase):
    def setUp(self):
        """Set up the tokenizers and data root."""
        # Define all model tokenizers used in experiments
        self.model_tokenizers = {
            'gpt2': 'gpt2',
            'qwen': 'Qwen/Qwen2.5-7B',
            'llama': 'meta-llama/Llama-3.1-8B', 
            'pythia': 'EleutherAI/pythia-6.9b-deduped'
        }
        self.data_root = Path('data')

    def test_all_tokenizers_and_examples(self):
        """
        Tests the ICLTokenLabeler on all JSON files in the data directory
        with all model tokenizers used in experiments.
        Validates that the answer extracted from token labels matches the
        original answer for ALL examples (not just ex1), and compares
        without stripping to catch delimiter token issues.
        """
        data_subdirs = ['abstractive', 'extractive', 'compositional']
        
        for model_name, tokenizer_path in self.model_tokenizers.items():
            print(f"\n=== Testing with {model_name} tokenizer ===")
            
            try:
                # Load tokenizer with error handling for missing models
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True,
                    local_files_only=False  # Allow downloading if not cached
                )
                token_labeler = ICLTokenLabeler(tokenizer)
            except Exception as e:
                print(f"Skipping {model_name} tokenizer due to error: {e}")
                continue
            
            for subdir in data_subdirs:
                data_dir = self.data_root / subdir
                if not data_dir.is_dir():
                    continue

                for task_file in data_dir.glob('*.json'):
                    print(f"  Testing {task_file.name} with {model_name}")
                    
                    with open(task_file, 'r', encoding='utf-8') as f:
                        try:
                            examples = json.load(f)
                        except json.JSONDecodeError:
                            self.fail(f"Failed to decode JSON from {task_file}")

                    # Check if the file contains a list of examples
                    if not isinstance(examples, list):
                        self.skipTest(f"Skipping {task_file} due to unsupported format (not a list).")
                        continue

                    if not examples:
                        continue
                    
                    # Validate that examples have the required keys
                    if not all('input' in ex and 'output' in ex for ex in examples):
                        self.skipTest(f"Skipping {task_file} due to missing 'input' or 'output' keys.")
                        continue

                    # Test with multiple examples to verify all example indices work
                    num_examples_to_test = min(3, len(examples))  # Test up to 3 examples
                    test_examples = examples[:num_examples_to_test]
                    
                    with self.subTest(tokenizer=model_name, task=task_file.name):
                        # Convert to new format (input/output -> query/answer)
                        converted_examples = [
                            {"query": ex["input"], "answer": ex["output"]} 
                            for ex in test_examples
                        ]
                        
                        # Use new interface: label_sequence returns (token_ids, token_labels, example_labels)
                        token_ids, token_labels, example_labels = token_labeler.label_sequence(converted_examples)
                        
                        # Validate each example's answer tokens
                        for example_idx in range(num_examples_to_test):
                            expected_output = test_examples[example_idx]['output']
                            
                            # Find answer tokens for this example
                            answer_label = f'answer_{example_idx}'
                            answer_token_indices = [
                                i for i, label in enumerate(token_labels) 
                                if label == answer_label
                            ]
                            
                            if not answer_token_indices:
                                # If no answer tokens found, the expected output should be empty
                                self.assertEqual(
                                    expected_output, "", 
                                    f"In {task_file.name} with {model_name} tokenizer (example {example_idx}), "
                                    f"no answer tokens were found but expected: '{expected_output}'"
                                )
                                continue

                            # Extract and decode the answer tokens
                            answer_tokens = [token_ids[i].item() for i in answer_token_indices]
                            decoded_answer = tokenizer.decode(answer_tokens)
                            
                            # Test exact match (this is the new behavior we want)
                            self.assertEqual(
                                decoded_answer, expected_output,
                                f"Answer mismatch in {task_file.name} with {model_name} tokenizer (example {example_idx})\n"
                                f"Expected: '{expected_output}'\n"
                                f"Got: '{decoded_answer}'\n"
                                f"The new ICLTokenizer should handle tokenizer differences automatically."
                            )
                            
                            print(f"    ✅ {task_file.name} example {example_idx}: Perfect match!")
                            print(f"       Expected: {repr(expected_output)}")
                            print(f"       Got: {repr(decoded_answer)}")

if __name__ == '__main__':
    unittest.main() 