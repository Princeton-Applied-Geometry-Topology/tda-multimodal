import csv
import os
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from dataset.dataset_utils import ICLTokenLabeler


class ConMeDataset(Dataset):
	"""Text-only ConMe dataset that reads CSV rows and constructs QA prompts.
	Yields items compatible with the pipeline: input_ids, attention_mask, token_labels, text.
	"""

	def __init__(
		self,
		tokenizer: PreTrainedTokenizer,
		csv_path: str,
		images_dir: Optional[str] = None,
		prompt_template: str = "Question: {question}\nOptions: {options}\nAnswer:",
		seq_len: int = 256,
		n_seqs: int = 500,
		shuffle: bool = False,
		max_tokens: Optional[int] = None,
		data_root: Optional[str] = None,
		**kwargs,
	):
		self.tokenizer = tokenizer
		self.labeler = ICLTokenLabeler(tokenizer)
		self.csv_path = csv_path
		self.images_dir = images_dir
		self.prompt_template = prompt_template
		self.seq_len = seq_len
		self.n_seqs = n_seqs
		self.shuffle = shuffle
		self.max_tokens = max_tokens
		self.data_root = data_root

		self.output_keys = ["input_ids", "attention_mask", "token_labels", "text"]
		self.cache = {key: [] for key in self.output_keys}

		self.examples = self._load_examples()
		if self.shuffle:
			import random
			random.shuffle(self.examples)
		# Cap to n_seqs if provided
		if self.n_seqs is not None:
			self.examples = self.examples[: self.n_seqs]

	def _load_examples(self) -> List[Dict[str, str]]:
		rows: List[Dict[str, str]] = []
		if not os.path.isabs(self.csv_path) and self.data_root is not None:
			csv_path = os.path.join(self.data_root, self.csv_path)
		else:
			csv_path = self.csv_path
		with open(csv_path, newline='', encoding='utf-8') as f:
			reader = csv.DictReader(f)
			self._raw_rows = []
			for row in reader:
				self._raw_rows.append(row)
				# Expect fields: image, question, correct_option, incorrect_option, base_question, answer, etc.
				question = row.get('question', '').strip()
				# Build two-option string; label correct as A by convention
				opt_a = row.get('correct_option', '').strip()
				opt_b = row.get('incorrect_option', '').strip()
				options = f"A) {opt_a} B) {opt_b}"
				# Ground truth answer text exactly as expected output tokens
				gt_answer = opt_a
				text = self.prompt_template.format(question=question, options=options)
				rows.append({
					"query": text,
					"answer": gt_answer,
				})
		return rows

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		# Single-example ICL: one QA per item
		example = self.examples[idx]
		sequence = [example]
		input_ids, token_labels, _ = self.labeler.label_sequence(sequence)

		# Truncate if needed
		if self.max_tokens is not None:
			input_ids = input_ids[: self.max_tokens]
			if len(token_labels) > self.max_tokens:
				token_labels = token_labels[: self.max_tokens]

		attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

		item = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"token_labels": token_labels,
			"text": example["query"],
		}
		# Attach image if use_images is True and image column exists
		try:
			if getattr(self, 'use_images', False):
				# CSV has COCO id under 'image'
				img_id = self._raw_rows[idx].get('image', None) if hasattr(self, '_raw_rows') else None
				if img_id is not None and self.images_dir:
					img_name = f"{int(img_id):012d}.jpg"
					img_path = os.path.join(self.images_dir, img_name)
					if os.path.exists(img_path):
						item['image'] = img_path
		except Exception:
			pass

		for key in self.output_keys:
			self.cache[key].append(item.get(key, [] if key == 'token_labels' else item.get(key)))
		return item

	def save_run(self, save_dir: str):
		import numpy as np
		os.makedirs(save_dir, exist_ok=True)
		n_items = len(self.cache['input_ids'])
		if n_items == 0:
			raise ValueError("No items in cache to save. Populate cache by iterating over the dataset before calling save_run.")
		max_len = max(x.shape[0] for x in self.cache['input_ids'])
		pad_id = self.tokenizer.pad_token_id
		input_ids_padded = [
			torch.cat([x, torch.full((max_len - x.shape[0],), pad_id, dtype=x.dtype)]) if x.shape[0] < max_len else x
			for x in self.cache['input_ids']
		]
		attention_mask_padded = [
			torch.cat([x, torch.zeros(max_len - x.shape[0], dtype=x.dtype)]) if x.shape[0] < max_len else x
			for x in self.cache['attention_mask']
		]
		input_ids_tensor = torch.stack(input_ids_padded)
		attention_mask_tensor = torch.stack(attention_mask_padded)
		torch.save(input_ids_tensor, os.path.join(save_dir, 'input_ids.pt'))
		torch.save(attention_mask_tensor, os.path.join(save_dir, 'attention_mask.pt'))
		np_labels = np.array(self.cache['token_labels'], dtype=object)
		import numpy as np
		np.save(os.path.join(save_dir, 'token_labels.npy'), np_labels)
		with open(os.path.join(save_dir, 'text.txt'), 'w') as f:
			f.write('\n'.join(self.cache['text'])) 