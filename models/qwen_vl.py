import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
import ast
import csv
import gc


class QwenVLModel:
    def __init__(self, 
                 task: Any,
                 model_path: str,
                 model_name: str,
                 device: str = "cuda",
                 torch_dtype: str = "auto",
                 max_new_tokens: int = 5,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 do_sample: bool = False,
                 device_map: str = None,
                 **kwargs):
        self.task = task
        self.model_path = model_path
        self.model_name = model_name
        # FORCE CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cuda:0"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Set torch dtype
        if torch_dtype == "auto":
            dtype = torch.bfloat16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        print(f"Loading Qwen model from {self.model_path}")
        print(f"FORCING device: {self.device}")
        
        # First load to get the modules
        _ = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="meta"
        )
        
        # Import the model class
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        
        # Load model WITHOUT device_map first
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        # EXPLICITLY MOVE MODEL TO CUDA
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model explicitly moved to {self.device}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Set output directory
        self.output_dir = Path(self.task.output_dir) / self.model_name / self.task.task_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Model loaded. Output will be saved to {self.output_dir}")
    
    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    def process_single_image_trial(self, image_path: str, prompt: str) -> str:
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # EXPLICITLY MOVE EVERY TENSOR TO CUDA
        inputs_cuda = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_cuda[k] = v.to(self.device)
            else:
                inputs_cuda[k] = v
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs_cuda,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_cuda['input_ids'], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip().replace('\n', ' ').replace('\r', ' ')
    
    def process_multi_image_trial(self, image_paths: List[str], prompt: str) -> str:
        content = []
        for img_path in image_paths:
            if not os.path.isabs(img_path):
                img_path = os.path.abspath(img_path)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            content.append({"type": "image", "image": f"file://{img_path}"})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # EXPLICITLY MOVE EVERY TENSOR TO CUDA
        inputs_cuda = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_cuda[k] = v.to(self.device)
            else:
                inputs_cuda[k] = v
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs_cuda,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_cuda['input_ids'], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip().replace('\n', ' ').replace('\r', ' ')
    
    def run(self):
        metadata_df = pd.read_csv(self.task.metadata_path)
        print(f"Loaded {len(metadata_df)} trials from metadata")
        
        if hasattr(self.task, 'prompt'):
            prompt_template = self.task.prompt
        elif hasattr(self.task, 'prompt_path') and self.task.prompt_path:
            with open(self.task.prompt_path, 'r') as f:
                prompt_template = f.read().strip()
        else:
            prompt_template = "Are these two images the same object or different objects?"
        
        print(f"Using prompt: {prompt_template}")
        print(f"Max tokens: {self.max_new_tokens}")
        print("-" * 80)
        
        results_path = self.output_dir / 'results.csv'
        
        start_idx = 0
        if results_path.exists():
            existing_df = pd.read_csv(results_path)
            start_idx = len(existing_df)
            print(f"Resuming from trial {start_idx}")
        
        first_row = metadata_df.iloc[0].to_dict()
        first_row['response'] = ''
        fieldnames = list(first_row.keys())
        
        mode = 'a' if start_idx > 0 else 'w'
        with open(results_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            if start_idx == 0:
                writer.writeheader()
            
            for idx in range(start_idx, len(metadata_df)):
                row = metadata_df.iloc[idx]
                
                if idx > 0 and idx % 100 == 0:
                    print(f"\nClearing GPU memory at trial {idx}")
                    self.clear_gpu_memory()
                
                try:
                    # Parse paths
                    if isinstance(row['path'], str):
                        if row['path'].startswith('['):
                            image_paths = ast.literal_eval(row['path'])
                        else:
                            image_paths = [row['path']]
                    else:
                        image_paths = row['path']
                    
                    # Update paths
                    image_paths = [p.replace('/scratch/gpfs/nb0564/vlm-behavior/', 
                                           '/scratch/gpfs/nb0564/vlm_interp/') 
                                 for p in image_paths]
                    
                    print(f"\nTrial {idx + 1}/{len(metadata_df)}")
                    print(f"Images: {[os.path.basename(p) for p in image_paths]}")
                    print(f"Correct answer: {row.get('correct_choice', 'N/A')}")
                    
                    if len(image_paths) == 1:
                        response = self.process_single_image_trial(image_paths[0], prompt_template)
                    else:
                        response = self.process_multi_image_trial(image_paths, prompt_template)
                    
                    print(f"Model response: {response}")
                    
                    row_dict = row.to_dict()
                    row_dict['response'] = str(response)
                    writer.writerow(row_dict)
                    csvfile.flush()
                    
                except Exception as e:
                    print(f"\nError processing trial {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    row_dict = row.to_dict()
                    row_dict['response'] = f"ERROR: {str(e)}"
                    writer.writerow(row_dict)
                    csvfile.flush()
        
        print(f"\nAll trials completed! Results saved to {results_path}")