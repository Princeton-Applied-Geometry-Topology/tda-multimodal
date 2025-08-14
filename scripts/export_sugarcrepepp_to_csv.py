import argparse
import csv
import os
import re
from typing import List, Tuple, Any


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Export SUGARCREPE++ subsets to a normalized CSV")
	p.add_argument("--out", required=True, help="Output CSV path")
	p.add_argument("--subsets", nargs="+", default=["replace_attribute", "replace_object", "replace_relation"], help="HF subsets to include")
	p.add_argument("--repo_id", default="Aman-J/SugarCrepe_pp", help="HF dataset repo id")
	p.add_argument("--split", default="train", help="Split to load from HF")
	p.add_argument("--cache_dir", default=None, help="HF cache dir")
	p.add_argument("--auth_token", default=None, help="HF auth token")
	return p.parse_args()


def pick(ex, keys: List[str], default: str = "") -> str:
	for k in keys:
		if k in ex and ex[k] is not None:
			val = ex[k]
			if not isinstance(val, (list, tuple)):
				return str(val)
	return default


def pick_list(ex, keys: List[str]) -> List[str]:
	for k in keys:
		if k in ex and ex[k] is not None and isinstance(ex[k], (list, tuple)):
			return [str(v) for v in ex[k] if v is not None]
	return []


def derive_image_id(ex) -> str:
	# Prefer explicit numeric ids
	img = pick(ex, ["image_id", "imageId", "image", "img_id", "coco_id", "cocoid"]).strip()
	if img:
		return img
	# Fall back to filename like '000000123456.jpg' or 'COCO_val2017_000000123456.jpg'
	filename = pick(ex, ["filename", "file_name", "image_name"]).strip()
	if filename:
		m = re.search(r"(\d{12})", filename)
		if m:
			return m.group(1)
		# if only digits without 12 length, still return
		digits = re.sub(r"\D", "", filename)
		if digits:
			return digits
	return ""


def derive_pos_pair(ex) -> Tuple[str, str]:
	# Map observed HF fields first
	cap1 = pick(ex, ["caption", "caption1", "caption_1"]).strip()
	cap2 = pick(ex, ["caption2", "caption_2"]).strip()
	if cap1 and cap2:
		return cap1, cap2
	# Prior fallbacks
	pos1 = pick(ex, ["pos1", "positive1", "caption_pos1", "pos"]).strip()
	pos2 = pick(ex, ["pos2", "positive2", "caption_pos2", "pos_aug", "positive_aug"]).strip()
	if pos1 and pos2:
		return pos1, pos2
	# List-style
	cands = pick_list(ex, [
		"positive_captions", "positives", "positive_list", "positive",
		"captions_pos", "captions_positive", "pos_captions",
	])
	if len(cands) >= 2:
		return cands[0].strip(), cands[1].strip()
	return "", ""


def derive_neg(ex) -> str:
	# Observed field
	neg = pick(ex, ["negative_caption"]).strip()
	if neg:
		return neg
	# Fallbacks
	neg = pick(ex, ["neg", "negative", "hard_negative"]).strip()
	if neg:
		return neg
	# List-style
	cands = pick_list(ex, ["negative_captions", "negatives", "negative_list"])
	return cands[0].strip() if cands else ""


def main():
	args = parse_args()
	from datasets import load_dataset, concatenate_datasets

	parts = []
	for name in args.subsets:
		ds = load_dataset(
			args.repo_id,
			name,
			split=args.split,
			cache_dir=args.cache_dir,
			use_auth_token=args.auth_token,
		)
		parts.append(ds)
	if len(parts) > 1:
		ds_all = concatenate_datasets(parts)
	else:
		ds_all = parts[0]

	rows = []
	skipped_missing_image = 0
	skipped_missing_pos = 0
	skipped_missing_neg = 0
	printed_keys = False
	for ex in ds_all:
		if not printed_keys:
			try:
				print("Sample keys:", list(ex.keys()))
			except Exception:
				pass
			printed_keys = True
		image_id = derive_image_id(ex)
		pos1, pos2 = derive_pos_pair(ex)
		neg = derive_neg(ex)
		neg_type = pick(ex, ["neg_type", "type", "manipulation", "manipulation_type"]).strip()
		uid = pick(ex, ["id", "uid", "example_id"]).strip()

		if not image_id:
			skipped_missing_image += 1
			continue
		if not (pos1 and pos2):
			skipped_missing_pos += 1
			continue
		if not neg:
			skipped_missing_neg += 1
			continue

		rows.append({
			"id": uid,
			"image_id": image_id,
			"pos1": pos1,
			"pos2": pos2,
			"neg": neg,
			"neg_type": neg_type,
		})

	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	with open(args.out, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["id", "image_id", "pos1", "pos2", "neg", "neg_type"])
		writer.writeheader()
		writer.writerows(rows)
	print(f"Wrote {args.out} with {len(rows)} rows from subsets: {args.subsets}")
	print(f"Skipped (no image_id): {skipped_missing_image}; (no pos1/pos2): {skipped_missing_pos}; (no neg): {skipped_missing_neg}")


if __name__ == "__main__":
	main() 