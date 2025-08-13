import os
import json
import zipfile
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from PIL import Image
import requests
from tqdm import tqdm
from dataset.dataset_utils import ICLTokenLabeler


class WinogroundDataset(Dataset):
    """
    Winoground dataset for evaluating visio-linguistic compositional reasoning.
    
    The dataset contains pairs of images and captions where both captions contain
    identical words but in different order, testing the model's ability to understand
    compositional relationships between vision and language.
    
    Paper: https://arxiv.org/abs/2204.03162
    Dataset: https://huggingface.co/datasets/facebook/winoground
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_root: str = "./data/winoground",
        use_auth_token: Optional[str] = None,
        download: bool = True,
        max_length: int = 512,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.use_auth_token = use_auth_token
        self.max_length = max_length
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_root, exist_ok=True)
        
        # Download and prepare dataset if requested
        if download:
            self._download_dataset()
        
        # Load examples
        self.examples = self._load_examples()
        
        # Initialize tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _download_dataset(self):
        """Download the Winoground dataset from Hugging Face."""
        print("Downloading Winoground dataset...")
        
        # Check if we already have the data
        examples_path = os.path.join(self.data_root, "examples.jsonl")
        images_dir = os.path.join(self.data_root, "images")
        
        if os.path.exists(examples_path) and os.path.exists(images_dir):
            print("Dataset already exists, skipping download.")
            return
        
        # Download examples.jsonl
        examples_url = "https://huggingface.co/datasets/facebook/winoground/resolve/main/data/examples.jsonl"
        
        if not os.path.exists(examples_path):
            print("Downloading examples.jsonl...")
            try:
                if self.use_auth_token:
                    headers = {"Authorization": f"Bearer {self.use_auth_token}"}
                    response = requests.get(examples_url, headers=headers, stream=True)
                else:
                    response = requests.get(examples_url, stream=True)
                
                response.raise_for_status()
                
                with open(examples_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Downloaded examples.jsonl")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    print("ERROR: Authentication required for Winoground dataset.")
                    print("Please visit: https://huggingface.co/datasets/facebook/winoground")
                    print("Click 'Access repository' and accept the terms.")
                    print("Then get your access token from: https://huggingface.co/settings/tokens")
                    print("Use the token when initializing the dataset:")
                    print("dataset = WinogroundDataset(tokenizer=tokenizer, use_auth_token='your_token_here')")
                    raise RuntimeError("Authentication required for Winoground dataset. See error message above.") from e
                else:
                    raise e
            except Exception as e:
                print(f"Error downloading examples.jsonl: {e}")
                raise e
        
        # Download images.zip
        images_url = "https://huggingface.co/datasets/facebook/winoground/resolve/main/data/images.zip"
        images_zip_path = os.path.join(self.data_root, "images.zip")
        
        if not os.path.exists(images_dir):
            if not os.path.exists(images_zip_path):
                print("Downloading images.zip...")
                try:
                    if self.use_auth_token:
                        headers = {"Authorization": f"Bearer {self.use_auth_token}"}
                        response = requests.get(images_url, headers=headers, stream=True)
                    else:
                        response = requests.get(images_url, stream=True)
                    
                    response.raise_for_status()
                    
                    with open(images_zip_path, 'wb') as f:
                        for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading images"):
                            f.write(chunk)
                    print("Downloaded images.zip")
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 401:
                        print("ERROR: Authentication required for Winoground dataset images.")
                        print("Please visit: https://huggingface.co/datasets/facebook/winoground")
                        print("Click 'Access repository' and accept the terms.")
                        print("Then get your access token from: https://huggingface.co/settings/tokens")
                        print("Use the token when initializing the dataset:")
                        print("dataset = WinogroundDataset(tokenizer=tokenizer, use_auth_token='your_token_here')")
                        raise RuntimeError("Authentication required for Winoground dataset. See error message above.") from e
                    else:
                        raise e
                except Exception as e:
                    print(f"Error downloading images.zip: {e}")
                    raise e
            
            # Extract images
            print("Extracting images...")
            with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            print("Extracted images")
            
            # Clean up zip file
            os.remove(images_zip_path)
        
        print("Dataset download complete!")
    
    def _load_examples(self) -> List[Dict]:
        """Load examples from the JSONL file."""
        examples_path = os.path.join(self.data_root, "examples.jsonl")
        
        if not os.path.exists(examples_path):
            raise FileNotFoundError(f"Examples file not found at {examples_path}. Run with download=True first.")
        
        examples = []
        with open(examples_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        print(f"Loaded {len(examples)} examples from Winoground dataset")
        return examples
    
    def _load_image(self, image_id: str) -> Optional[Image.Image]:
        """Load an image by ID."""
        image_path = os.path.join(self.data_root, "images", f"{image_id}.png")
        if os.path.exists(image_path):
            try:
                return Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None
        return None
    
    def _create_prompt(self, caption_0: str, caption_1: str) -> str:
        """Create a prompt for the model to choose between two captions."""
        return f"Caption 1: {caption_0}\nCaption 2: {caption_1}\n\nWhich caption better describes the image? Answer with 'Caption 1' or 'Caption 2'."
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single example from the dataset."""
        example = self.examples[idx]
        
        # Extract data from example
        caption_0 = example.get('caption_0', '')
        caption_1 = example.get('caption_1', '')
        image_0_id = example.get('image_0', '')
        image_1_id = example.get('image_1', '')
        
        # Load images
        image_0 = self._load_image(image_0_id)
        image_1 = self._load_image(image_1_id)
        
        # Create prompts for both image-caption pairs
        prompt_0 = self._create_prompt(caption_0, caption_1)
        prompt_1 = self._create_prompt(caption_0, caption_1)
        
        # Tokenize prompts
        encoding_0 = self.tokenizer(
            prompt_0,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        encoding_1 = self.tokenizer(
            prompt_1,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Ensure we have tensors and handle both single and batch outputs
        def process_encoding(encoding):
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # Convert to tensor if it's a list
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask)
            
            # Squeeze batch dimension if present
            if input_ids.dim() > 1:
                input_ids = input_ids.squeeze(0)
            if attention_mask.dim() > 1:
                attention_mask = attention_mask.squeeze(0)
                
            return input_ids, attention_mask
        
        input_ids_0, attention_mask_0 = process_encoding(encoding_0)
        input_ids_1, attention_mask_1 = process_encoding(encoding_1)
        
        # Create item
        item = {
            'input_ids_0': input_ids_0,
            'attention_mask_0': attention_mask_0,
            'input_ids_1': input_ids_1,
            'attention_mask_1': attention_mask_1,
            'caption_0': caption_0,
            'caption_1': caption_1,
            'image_0': image_0,
            'image_1': image_1,
            'image_0_id': image_0_id,
            'image_1_id': image_1_id,
            'tags': example.get('tags', []),
            'secondary_tag': example.get('secondary_tag', ''),
            'num_main_preds': example.get('num_main_preds', 0),
            'collapsed_tag': example.get('collapsed_tag', ''),
        }
        
        return item
    
    def get_evaluation_batch(self, batch_size: int = 8) -> List[Dict]:
        """Get a batch of examples for evaluation."""
        indices = torch.randperm(len(self))[:batch_size]
        return [self[i] for i in indices]
    
    def get_examples_by_tag(self, tag: str) -> List[Dict]:
        """Get examples filtered by a specific tag."""
        filtered_examples = []
        for example in self.examples:
            if tag in example.get('tags', []):
                filtered_examples.append(example)
        return filtered_examples
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        tags_count = {}
        secondary_tags_count = {}
        
        for example in self.examples:
            for tag in example.get('tags', []):
                tags_count[tag] = tags_count.get(tag, 0) + 1
            
            secondary_tag = example.get('secondary_tag', '')
            if secondary_tag:
                secondary_tags_count[secondary_tag] = secondary_tags_count.get(secondary_tag, 0) + 1
        
        return {
            'total_examples': len(self.examples),
            'tags_distribution': tags_count,
            'secondary_tags_distribution': secondary_tags_count,
        }


class WinogroundEvaluationDataset(Dataset):
    """
    Simplified version of Winoground dataset focused on evaluation tasks.
    This version is optimized for running evaluation metrics and scoring.
    """
    
    def __init__(
        self,
        data_root: str = "./data/winoground",
        download: bool = True,
        **kwargs,
    ):
        self.data_root = data_root
        
        if download:
            self._download_dataset()
        
        self.examples = self._load_examples()
    
    def _download_dataset(self):
        """Download the Winoground dataset from Hugging Face."""
        # Reuse the download logic from WinogroundDataset
        temp_dataset = WinogroundDataset(
            tokenizer=None,  # We don't need tokenizer for evaluation
            data_root=self.data_root,
            download=True
        )
    
    def _load_examples(self) -> List[Dict]:
        """Load examples from the JSONL file."""
        examples_path = os.path.join(self.data_root, "examples.jsonl")
        
        if not os.path.exists(examples_path):
            raise FileNotFoundError(f"Examples file not found at {examples_path}. Run with download=True first.")
        
        examples = []
        with open(examples_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single example for evaluation."""
        example = self.examples[idx]
        
        return {
            'caption_0': example.get('caption_0', ''),
            'caption_1': example.get('caption_1', ''),
            'image_0': example.get('image_0', ''),
            'image_1': example.get('image_1', ''),
            'tags': example.get('tags', []),
            'secondary_tag': example.get('secondary_tag', ''),
            'num_main_preds': example.get('num_main_preds', 0),
            'collapsed_tag': example.get('collapsed_tag', ''),
        }
