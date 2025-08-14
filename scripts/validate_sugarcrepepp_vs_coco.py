import argparse
import csv
import os
from collections import Counter


def parse_args():
	p = argparse.ArgumentParser(description="Validate SUGARCREPE++ CSV against COCO images dir")
	p.add_argument("--csv", required=True, help="Path to triplets.csv")
	p.add_argument("--coco_dir", required=True, help="Path to COCO val2017 images directory (folder with *.jpg)")
	return p.parse_args()


def coco_path(coco_dir: str, image_id: str) -> str:
	try:
		img_int = int(str(image_id))
		name = f"{img_int:012d}.jpg"
	except Exception:
		name = str(image_id)
		if not name.lower().endswith('.jpg'):
			name += '.jpg'
	return os.path.join(coco_dir, name)


def main():
	args = parse_args()
	missing = []
	seen = Counter()
	with open(args.csv, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			img_id = row.get('image_id', '').strip()
			if not img_id:
				continue
			seen[img_id] += 1
			path = coco_path(args.coco_dir, img_id)
			if not os.path.exists(path):
				missing.append((img_id, path))
		total_ids = sum(seen.values())
		unique_ids = len(seen)
		print(f"Total rows with image_id: {total_ids} | Unique image_ids: {unique_ids}")
		print(f"COCO dir: {args.coco_dir}")
		if missing:
			print(f"Missing {len(missing)} image files. First 10:")
			for mid, p in missing[:10]:
				print(f"  {mid} -> {p}")
			raise SystemExit(2)
		else:
			print("All image files found. Alignment OK.")


if __name__ == "__main__":
	main() 