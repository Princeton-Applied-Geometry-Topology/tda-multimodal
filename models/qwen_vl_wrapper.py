from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
from models.model import LLM
from PIL import Image


class QwenVL(LLM):
	"""Wrapper for Qwen2-VL models compatible with the pipeline.
	- Uses AutoProcessor for multimodal batching (text + images).
	- Computes metrics from hidden states if requested.
	"""

	def __init__(
		self,
		weights_path: str = 'Qwen/Qwen2-VL-2B-Instruct',
		processor: Optional[AutoProcessor] = None,
		tokenizer: Optional[AutoTokenizer] = None,
		max_output_tokens: int = 5,
		device: str = 'auto',
		torch_dtype: str = 'auto',
		output_dir: str = None,
		model_name: str = 'qwen2-vl-2b',
		task_name: str = None,
		max_input_tokens: int = None,
		ablate_heads: bool = False,
		n_ablate_heads: int = None,
		trust_remote_code: bool = True,
		local_files_only: bool = True,
		**kwargs
	):
		super().__init__(
			weights_path=weights_path,
			tokenizer=tokenizer,
			max_output_tokens=max_output_tokens,
			device=device,
			torch_dtype=torch_dtype,
			output_dir=output_dir,
			model_name=model_name,
			task_name=task_name,
			max_input_tokens=max_input_tokens,
			ablate_heads=ablate_heads,
			n_ablate_heads=n_ablate_heads,
		)
		# Load processor and model
		self.processor = processor or AutoProcessor.from_pretrained(
			weights_path,
			trust_remote_code=trust_remote_code,
			local_files_only=local_files_only,
		)
		dtype = torch.float16 if torch_dtype == 'auto' else getattr(torch, torch_dtype)
		self.model = Qwen2VLForConditionalGeneration.from_pretrained(
			weights_path,
			trust_remote_code=trust_remote_code,
			torch_dtype=dtype,
			local_files_only=local_files_only,
		).to(self.device)
		self.model.eval()

	def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		texts: List[str] = batch.get('text', [])
		images = batch.get('image', None)
		if images is not None and len(images) > 0:
			# Load images from paths if needed
			pil_images: List[Image.Image] = []
			for img in images:
				if isinstance(img, Image.Image):
					pil_images.append(img)
				elif isinstance(img, str):
					pil_images.append(Image.open(img).convert('RGB'))
				else:
					pil_images.append(None)
			# Build chat-formatted texts per sample
			chat_texts: List[str] = []
			for txt in texts:
				messages = [{
					"role": "user",
					"content": [
						{"type": "image"},
						{"type": "text", "text": txt},
					],
				}]
				chat_texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
			inputs = self.processor(
				text=chat_texts,
				images=pil_images,
				return_tensors='pt',
				padding=True
			)
		else:
			inputs = self.processor(
				text=texts,
				return_tensors='pt',
				padding=True
			)
		for k, v in list(inputs.items()):
			if isinstance(v, torch.Tensor):
				inputs[k] = v.to(self.device)
		return inputs

	def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor]:
		with torch.no_grad():
			inputs = self._prepare_inputs(batch)
			outputs = self.model(**inputs, output_hidden_states=False)
			logits = outputs.logits
			predicted_tokens = torch.argmax(logits, dim=-1)
			decoded_outputs = self.tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
		return decoded_outputs, predicted_tokens

	def forward_with_metrics(
		self,
		batch: Dict[str, torch.Tensor],
		metrics_to_compute: Dict[str, callable]
	) -> Tuple[List[str], torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
		with torch.no_grad():
			inputs = self._prepare_inputs(batch)
			outputs = self.model(
				**inputs,
				output_hidden_states=True,
				return_dict=True
			)
			logits = outputs.logits
			predicted_tokens = torch.argmax(logits, dim=-1).cpu()
			decoded_outputs = self.tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)

			all_metric_results: Dict[str, Dict[str, torch.Tensor]] = {name: {} for name in metrics_to_compute}
			if outputs.hidden_states is not None and len(metrics_to_compute) > 0:
				for layer_idx, activations in enumerate(outputs.hidden_states):
					act = activations.detach().to('cpu')
					layer_key = f'layer-{layer_idx}'
					for metric_name, metric_fn in metrics_to_compute.items():
						try:
							metric_scores = metric_fn(act)
							all_metric_results[metric_name][layer_key] = metric_scores
						except Exception:
							continue
		return decoded_outputs, predicted_tokens, all_metric_results 