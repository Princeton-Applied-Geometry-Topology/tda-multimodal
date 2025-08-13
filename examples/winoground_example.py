import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import WinogroundDataset, WinogroundEvaluationDataset
from transformers import AutoTokenizer


def main():
    """Main function demonstrating Winoground dataset usage."""
    
    # Initialize tokenizer (you can use any Hugging Face tokenizer)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check for Hugging Face token in environment
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print(f"Found HF_TOKEN environment variable: {hf_token[:8]}...")
        use_auth_token = hf_token
    else:
        print("No HF_TOKEN environment variable found.")
        print("To set it, run: export HF_TOKEN='your_token_here'")
        use_auth_token = None
    
    # Initialize Winoground dataset with automatic download
    print("\nInitializing Winoground dataset...")
    if not use_auth_token:
        print("Note: Winoground dataset requires authentication.")
        print("If you get a 401 error, you need to:")
        print("1. Visit: https://huggingface.co/datasets/facebook/winoground")
        print("2. Click 'Access repository' and accept the terms")
        print("3. Get your access token from: https://huggingface.co/settings/tokens")
        print("4. Set the token: export HF_TOKEN='your_token_here'")
        print("5. Or use it directly: use_auth_token='your_token_here'")
        print()
    
    try:
        dataset = WinogroundDataset(
            tokenizer=tokenizer,
            data_root="./data/winoground",
            download=True,
            max_length=256,
            use_auth_token=use_auth_token
        )
        
        print(f"Dataset loaded with {len(dataset)} examples")
        
        # Get dataset statistics
        print("\nDataset statistics:")
        stats = dataset.get_statistics()
        print(f"Total examples: {stats['total_examples']}")
        print(f"Tags distribution: {stats['tags_distribution']}")
        print(f"Secondary tags distribution: {stats['secondary_tags_distribution']}")
        
        # Get a few examples
        print("\nSample examples:")
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"\nExample {i}:")
            print(f"  Caption 0: {example['caption_0']}")
            print(f"  Caption 1: {example['caption_1']}")
            print(f"  Image 0 ID: {example['image_0_id']}")
            print(f"  Image 1 ID: {example['image_1_id']}")
            print(f"  Tags: {example['tags']}")
            print(f"  Secondary tag: {example['secondary_tag']}")
            print(f"  Input IDs shape: {example['input_ids_0'].shape}")
        
        # Filter examples by tag
        print("\nExamples with 'spatial' tag:")
        spatial_examples = dataset.get_examples_by_tag('spatial')
        print(f"Found {len(spatial_examples)} examples with 'spatial' tag")
        
        if spatial_examples:
            example = spatial_examples[0]
            print(f"  Sample spatial example:")
            print(f"    Caption 0: {example['caption_0']}")
            print(f"    Caption 1: {example['caption_1']}")
        
        # Get evaluation batch
        print("\nGetting evaluation batch...")
        eval_batch = dataset.get_evaluation_batch(batch_size=4)
        print(f"Evaluation batch size: {len(eval_batch)}")
        
        # Initialize evaluation dataset (simplified version)
        print("\nInitializing evaluation dataset...")
        eval_dataset = WinogroundEvaluationDataset(
            data_root="./data/winoground",
            download=False  # Already downloaded
        )
        
        print(f"Evaluation dataset loaded with {len(eval_dataset)} examples")
        
        # Show evaluation example
        eval_example = eval_dataset[0]
        print(f"\nEvaluation example:")
        print(f"  Caption 0: {eval_example['caption_0']}")
        print(f"  Caption 1: {eval_example['caption_1']}")
        print(f"  Tags: {eval_example['tags']}")
        
        print("\nWinoground dataset setup complete!")
        
    except RuntimeError as e:
        print(f"\nSetup failed: {e}")
        print("\nTo use the dataset with authentication:")
        print("1. Get your Hugging Face access token")
        print("2. Set it: export HF_TOKEN='your_token'")
        print("3. Or use it directly: use_auth_token='your_token'")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check the error message above and ensure you have proper access to the dataset.")


if __name__ == "__main__":
    main()
