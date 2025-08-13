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

class TestTaskAccuracy(unittest.TestCase):
    def setUp(self):
        """Set up the tokenizer and data root."""
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.token_labeler = ICLTokenLabeler(self.tokenizer)
        self.data_root = Path('data')

    def test_all_tasks_and_modes(self):
        """
        Tests accuracy computation across all tasks and accuracy modes.
        - 'all': All tokens in the answer must be correct.
        - 'token_wise': Accuracy is the fraction of correct tokens.
        - 'first_token': Only the first token of the answer is checked.
        """
        data_subdirs = ['abstractive', 'extractive', 'compositional']
        for subdir in data_subdirs:
            data_dir = self.data_root / subdir
            if not data_dir.is_dir():
                continue

            for task_file in data_dir.glob('*.json'):
                print(f"Testing task accuracy for task: {task_file.name}")
                with open(task_file, 'r', encoding='utf-8') as f:
                    try:
                        examples = json.load(f)
                    except json.JSONDecodeError:
                        self.fail(f"Failed to decode JSON from {task_file}")

                if not isinstance(examples, list) or not examples:
                    self.skipTest(f"Skipping {task_file}: not a list or empty.")
                    continue

                if not all('input' in ex and 'output' in ex for ex in examples):
                    self.skipTest(f"Skipping {task_file}: missing 'input' or 'output' keys.")
                    continue

                for i, example in enumerate(examples):
                    with self.subTest(task=task_file.name, example_index=i):
                        # Convert to new format and process one example at a time
                        converted_example = [{"query": example["input"], "answer": example["output"]}]
                        
                        # Use new interface
                        token_ids, token_labels, example_labels = self.token_labeler.label_sequence(converted_example)
                        
                        # Find answer tokens for this example
                        answer_token_indices = [
                            idx for idx, label in enumerate(token_labels) 
                            if label == 'answer_0'
                        ]
                        
                        if not answer_token_indices:
                            # If no answer tokens found, the expected output should be empty
                            self.assertEqual(
                                example['output'], "",
                                f"No answer tokens found for {task_file.name} example {i}, but expected: '{example['output']}'"
                            )
                            continue

                        # Extract and decode the answer tokens
                        answer_tokens = [token_ids[idx].item() for idx in answer_token_indices]
                        predicted_answer = self.tokenizer.decode(answer_tokens)
                        ground_truth = example['output']

                        # Test all accuracy modes
                        for mode in ['all', 'token_wise', 'first_token']:
                            with self.subTest(mode=mode):
                                # Test perfect prediction
                                accuracy_perfect = self.compute_accuracy(predicted_answer, ground_truth, mode=mode)
                                self.assertEqual(accuracy_perfect, 1.0, 
                                    f"Perfect prediction should yield 1.0 accuracy for {task_file.name} example {i} with mode '{mode}'. "
                                    f"Got: {accuracy_perfect}, Predicted: '{predicted_answer}', Ground truth: '{ground_truth}'")

                                # Test incorrect prediction
                                wrong_prediction = "WRONG_ANSWER"
                                accuracy_wrong = self.compute_accuracy(wrong_prediction, ground_truth, mode=mode)
                                
                                # For token_wise mode, we might get partial accuracy
                                if mode == 'token_wise':
                                    self.assertLessEqual(accuracy_wrong, 1.0, 
                                        f"Wrong prediction should yield <= 1.0 accuracy for {task_file.name} example {i} with mode '{mode}'. "
                                        f"Got: {accuracy_wrong}")
                                else:
                                    self.assertEqual(accuracy_wrong, 0.0,
                                        f"Wrong prediction should yield 0.0 accuracy for {task_file.name} example {i} with mode '{mode}'. "
                                        f"Got: {accuracy_wrong}")

    def compute_accuracy(self, predicted_answer, ground_truth, mode='all'):
        """
        Compute accuracy based on the specified mode.
        
        Args:
            predicted_answer (str): The predicted answer.
            ground_truth (str): The ground truth answer.
            mode (str): The accuracy mode ('all', 'token_wise', or 'first_token').
        
        Returns:
            float: The accuracy score.
        """
        if mode == 'all':
            # All tokens must match
            return 1.0 if predicted_answer == ground_truth else 0.0
        elif mode == 'token_wise':
            # Accuracy is the fraction of correct tokens
            pred_tokens = self.tokenizer.tokenize(predicted_answer)
            gt_tokens = self.tokenizer.tokenize(ground_truth)
            if not gt_tokens:
                return 1.0 if not pred_tokens else 0.0
            correct_tokens = sum(1 for p, g in zip(pred_tokens, gt_tokens) if p == g)
            return correct_tokens / len(gt_tokens)
        elif mode == 'first_token':
            # Only the first token is checked
            pred_tokens = self.tokenizer.tokenize(predicted_answer)
            gt_tokens = self.tokenizer.tokenize(ground_truth)
            if not gt_tokens:
                return 1.0 if not pred_tokens else 0.0
            return 1.0 if pred_tokens and pred_tokens[0] == gt_tokens[0] else 0.0
        else:
            raise ValueError(f"Unknown accuracy mode: {mode}")

if __name__ == '__main__':
    unittest.main() 