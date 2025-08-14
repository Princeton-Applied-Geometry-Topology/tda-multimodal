import os
import csv
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from dataset.dataset_utils import ICLTokenLabeler


class SugarCrepePlusPlusDataset(Dataset):
    """
    Image+Text SUGARCREPE++ dataset loader.

    Supports two sources:
      1) Local CSV (default)
      2) Hugging Face datasets (e.g., "Aman-J/SugarCrepe_pp") when enabled

    CSV expects columns:
      - id: unique row id (string or int)
      - image_id: COCO 2017 val image id (int or str, no extension)
      - pos1: first positive caption
      - pos2: second positive caption
      - neg: hard negative caption
      - neg_type: manipulation type/category for the negative (e.g., replace_obj)

    This expands each row into two items: (pos1 vs neg) and (pos2 vs neg).
    Each item yields the standard keys required by the pipeline:
      - input_ids (Tensor)
      - attention_mask (Tensor)
      - token_labels (List[str])
      - text (str)
    And attaches the image path under key 'image' for VLMs.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        csv_path: Optional[str] = None,
        coco_images_dir: str = "/scratch/gpfs/nb0564/tda-multimodal/datasets/COCO/val2017",
        prompt_template: str = "Question: Which caption correctly describes the image?\nOptions: A) {positive} B) {negative}\nAnswer:",
        seq_len: int = 256,
        n_seqs: Optional[int] = None,
        shuffle: bool = False,
        max_tokens: Optional[int] = None,
        data_root: Optional[str] = None,
        # Field mapping
        image_field: str = "image_id",
        pos1_field: str = "pos1",
        pos2_field: str = "pos2",
        neg_field: str = "neg",
        neg_type_field: str = "neg_type",
        id_field: str = "id",
        use_images: bool = True,
        # Optional HF dataset loading
        use_hf: bool = False,
        hf_repo_id: Optional[str] = None,
        hf_subsets: Optional[List[str]] = None,
        hf_split: str = "test",
        use_auth_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.labeler = ICLTokenLabeler(tokenizer)
        self.csv_path = csv_path
        self.coco_images_dir = coco_images_dir
        self.prompt_template = prompt_template
        self.seq_len = seq_len
        self.n_seqs = n_seqs
        self.shuffle = shuffle
        self.max_tokens = max_tokens
        self.data_root = data_root
        self.image_field = image_field
        self.pos1_field = pos1_field
        self.pos2_field = pos2_field
        self.neg_field = neg_field
        self.neg_type_field = neg_type_field
        self.id_field = id_field
        self.use_images = use_images
        # HF options
        self.use_hf = use_hf
        self.hf_repo_id = hf_repo_id or "Aman-J/SugarCrepe_pp"
        self.hf_subsets = hf_subsets or [
            "replace_attribute",
            "replace_object",
            "replace_relation",
        ]
        self.hf_split = hf_split
        self.use_auth_token = use_auth_token
        self.hf_cache_dir = hf_cache_dir

        self.output_keys = ["input_ids", "attention_mask", "token_labels", "text"]
        self.cache = {key: [] for key in self.output_keys}

        self.examples = self._load_examples()
        if self.shuffle:
            import random
            random.shuffle(self.examples)
        if self.n_seqs is not None:
            self.examples = self.examples[: self.n_seqs]

    def _resolve_paths(self) -> (Optional[str], str):
        # Resolve CSV path if provided
        csv_path = None
        if self.csv_path:
            if not os.path.isabs(self.csv_path) and self.data_root is not None:
                csv_path = os.path.join(self.data_root, self.csv_path)
            else:
                csv_path = self.csv_path
        # Resolve COCO images dir
        coco_dir = self.coco_images_dir
        if not os.path.isabs(coco_dir) and self.data_root is not None:
            coco_dir = os.path.join(self.data_root, coco_dir)
        if self.use_images and not os.path.isdir(coco_dir):
            raise FileNotFoundError(f"COCO images directory not found: {coco_dir}")
        return csv_path, coco_dir

    def _coco_img_path(self, coco_dir: str, img_id: str) -> Optional[str]:
        try:
            img_int = int(img_id)
            img_name = f"{img_int:012d}.jpg"
        except Exception:
            # If already a filename without extension normalization
            base = os.path.basename(str(img_id))
            if not base.lower().endswith('.jpg'):
                base = base + '.jpg'
            img_name = base
        path = os.path.join(coco_dir, img_name)
        return path if os.path.exists(path) else None

    def _read_csv_rows(self, csv_path: str, coco_dir: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trial_id = str(row.get(self.id_field, len(rows)))
                image_id = str(row.get(self.image_field, '')).strip()
                pos1 = row.get(self.pos1_field, '').strip()
                pos2 = row.get(self.pos2_field, '').strip()
                neg = row.get(self.neg_field, '').strip()
                neg_type = row.get(self.neg_type_field, '').strip()

                img_path = self._coco_img_path(coco_dir, image_id) if self.use_images else None
                if self.use_images and not img_path:
                    raise FileNotFoundError(
                        f"Image for COCO id {image_id} not found under {coco_dir}."
                    )

                text1 = self.prompt_template.format(positive=pos1, negative=neg)
                rows.append({
                    "trial_id": trial_id,
                    "condition": "normal_pos1",
                    "manipulation_type": neg_type,
                    "image_path": img_path,
                    "query": text1,
                    "answer": pos1,
                })
                text2 = self.prompt_template.format(positive=pos2, negative=neg)
                rows.append({
                    "trial_id": trial_id,
                    "condition": "normal_pos2",
                    "manipulation_type": neg_type,
                    "image_path": img_path,
                    "query": text2,
                    "answer": pos2,
                })
        return rows

    def _read_hf_rows(self, coco_dir: str) -> List[Dict[str, str]]:
        from datasets import load_dataset, concatenate_datasets
        parts = []
        for subset in self.hf_subsets:
            ds = load_dataset(
                self.hf_repo_id,
                subset,
                split=self.hf_split,
                use_auth_token=self.use_auth_token,
                cache_dir=self.hf_cache_dir,
            )
            parts.append(ds)
        if len(parts) > 1:
            ds_all = concatenate_datasets(parts)
        else:
            ds_all = parts[0]
        rows: List[Dict[str, str]] = []
        for ex in ds_all:
            # Robust field access with fallbacks
            def get_field(example: Dict, keys: List[str], default: str = "") -> str:
                for k in keys:
                    if k in example and example[k] is not None:
                        return str(example[k])
                return default

            trial_id = get_field(ex, [self.id_field, "id", "uid", "example_id"], str(len(rows)//2))
            image_id = get_field(ex, [self.image_field, "imageId", "image_id"], "").strip()
            pos1 = get_field(ex, [self.pos1_field, "pos1", "caption_pos1", "positive1", "caption_1"], "").strip()
            pos2 = get_field(ex, [self.pos2_field, "pos2", "caption_pos2", "positive2", "caption_2"], "").strip()
            neg = get_field(ex, [self.neg_field, "neg", "negative", "hard_negative"], "").strip()
            neg_type = get_field(ex, [self.neg_type_field, "neg_type", "type", "manipulation_type"], "").strip()

            img_path = self._coco_img_path(coco_dir, image_id) if self.use_images else None
            if self.use_images and not img_path:
                raise FileNotFoundError(
                    f"Image for COCO id {image_id} not found under {coco_dir}."
                )

            text1 = self.prompt_template.format(positive=pos1, negative=neg)
            rows.append({
                "trial_id": trial_id,
                "condition": "normal_pos1",
                "manipulation_type": neg_type,
                "image_path": img_path,
                "query": text1,
                "answer": pos1,
            })
            text2 = self.prompt_template.format(positive=pos2, negative=neg)
            rows.append({
                "trial_id": trial_id,
                "condition": "normal_pos2",
                "manipulation_type": neg_type,
                "image_path": img_path,
                "query": text2,
                "answer": pos2,
            })
        return rows

    def _load_examples(self) -> List[Dict[str, str]]:
        csv_path, coco_dir = self._resolve_paths()
        rows: List[Dict[str, str]]
        if csv_path and os.path.exists(csv_path):
            rows = self._read_csv_rows(csv_path, coco_dir)
        elif self.use_hf:
            rows = self._read_hf_rows(coco_dir)
        else:
            raise FileNotFoundError(
                "No valid CSV found and use_hf is False. Provide csv_path or set use_hf=true with hf_repo_id/subsets."
            )
        return rows

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        sequence = [{"query": example["query"], "answer": example["answer"]}]
        input_ids, token_labels, _ = self.labeler.label_sequence(sequence)

        if self.max_tokens is not None:
            input_ids = input_ids[: self.max_tokens]
            if len(token_labels) > self.max_tokens:
                token_labels = token_labels[: self.max_tokens]

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        item: Dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_labels": token_labels,
            "text": example["query"],
        }
        if self.use_images and example.get("image_path"):
            item["image"] = example["image_path"]
        # Carry metadata through for post-hoc alignment
        item["trial_id"] = example.get("trial_id")
        item["condition"] = example.get("condition")
        item["manipulation_type"] = example.get("manipulation_type")

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
        np.save(os.path.join(save_dir, 'token_labels.npy'), np_labels)
        with open(os.path.join(save_dir, 'text.txt'), 'w') as f:
            f.write('\n'.join(self.cache['text'])) 