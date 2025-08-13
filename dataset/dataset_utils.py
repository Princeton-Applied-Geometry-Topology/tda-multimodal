import torch
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple


class ICLTokenLabeler:
	"""
	Tokenize and label simple in-context learning sequences of the form:
	Q: <query>\n
	A: <answer>\n\n
	This returns:
	- concatenated input_ids (torch.Tensor)
	- per-token string labels (List[str]) like 'query_0', 'answer_0', etc.
	- concatenated example index mask (torch.Tensor)
	"""

	def __init__(self, tokenizer: PreTrainedTokenizer):
		self.tokenizer = tokenizer
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
		self.query_delim = "Q: "
		self.answer_delim = "A: "
		self.query_spacing = "\n"
		self.example_spacing = "\n\n"

	def _label_tokens(self, text: str, tag: str, ex_idx: int) -> Tuple[List[int], List[str]]:
		ids = self.tokenizer(text, add_special_tokens=False).input_ids
		labels = [f"{tag}_{ex_idx}"] * len(ids)
		return ids, labels

	def label_example(self, example: Dict[str, str], ex_idx: int) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
		parts = [
			(self.query_delim, "query_delimiter"),
			(example["query"], "query"),
			(self.query_spacing, "query_spacing"),
			(self.answer_delim, "answer_delimiter"),
			(example["answer"], "answer"),
			(self.example_spacing, "example_delimiter"),
		]
		ids_all: List[int] = []
		labels_all: List[str] = []
		for text, tag in parts:
			ids, labels = self._label_tokens(text, tag, ex_idx)
			ids_all.extend(ids)
			labels_all.extend(labels)
		mask = [ex_idx] * len(ids_all)
		return torch.tensor(ids_all, dtype=torch.long), labels_all, torch.tensor(mask, dtype=torch.long)

	def label_sequence(self, sequence: List[Dict[str, str]]) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
		ids_list: List[torch.Tensor] = []
		labels_list: List[str] = []
		mask_list: List[torch.Tensor] = []
		for i, ex in enumerate(sequence):
			ids, labels, mask = self.label_example(ex, i)
			ids_list.append(ids)
			labels_list.extend(labels)
			mask_list.append(mask)
		return torch.cat(ids_list), labels_list, torch.cat(mask_list)

	def tokenize_sequences(self, all_inputs: List[List[Dict[str, str]]]) -> Tuple[torch.Tensor, List[List[str]], torch.Tensor]:
		ids_batch: List[torch.Tensor] = []
		labels_batch: List[List[str]] = []
		mask_batch: List[torch.Tensor] = []
		for seq in all_inputs:
			ids, labels, mask = self.label_sequence(seq)
			ids_batch.append(ids)
			labels_batch.append(labels)
			mask_batch.append(mask)
		padded_ids = pad_sequence(ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
		padded_mask = pad_sequence(mask_batch, batch_first=True, padding_value=-1)
		max_len = padded_ids.shape[1]
		padded_labels = [row + ["PAD"] * (max_len - len(row)) for row in labels_batch]
		return padded_ids, padded_labels, padded_mask 