import torch
from transformers import AutoTokenizer
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class LLM:
    '''Base class for language models'''
    
    def __init__(
        self,
        weights_path: str,
        tokenizer: AutoTokenizer,
        max_output_tokens: int = 256,
        device: str = 'auto',
        torch_dtype: str = 'auto',
        output_dir: str = None,
        model_name: str = None,
        task_name: str = None,
        max_input_tokens: int = None,
        ablate_heads: bool = False,
        n_ablate_heads: int = None
    ):
        self.max_output_tokens = max_output_tokens
        self.weights_path = weights_path
        self.torch_dtype = torch_dtype
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_name = task_name
        self.output_dir = output_dir
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.ablate_heads = ablate_heads
        self.n_ablate_heads = n_ablate_heads
        
        # Tokenizer setup
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Model will be initialized in subclasses
        self.model = None
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor]:
        '''Run forward pass and return predictions
        
        Args:
            batch: Dictionary containing input_ids and attention_mask tensors
            
        Returns:
            Tuple of (decoded_outputs, predicted_tokens)
        '''
        raise NotImplementedError('Subclasses must implement forward method')
    
    def get_representations(self, batch: Dict[str, torch.Tensor], layer_name: str) -> torch.Tensor:
        '''Get model representations for a specific layer
        
        Args:
            batch: Dictionary containing input_ids and attention_mask tensors
            layer_name: Name of the layer to extract representations from
            
        Returns:
            Tensor of representations
        '''
        raise NotImplementedError('Subclasses must implement get_representations method')