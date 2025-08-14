import os

BASE_DIR = "/scratch/gpfs/nb0564/tda-multimodal/datasets/COCO"

try:
	import fiftyone as fo
	import fiftyone.zoo as foz
except ImportError:
	raise SystemExit("FiftyOne is not installed. Run: pip install fiftyone")

os.makedirs(BASE_DIR, exist_ok=True)

print(f"Downloading COCO-2017 validation split to base dir: {BASE_DIR}")
dataset = foz.load_zoo_dataset(
	"coco-2017",
	split="validation",
	dataset_dir=BASE_DIR,
	download_if_necessary=True,
)

images_dir = os.path.join(BASE_DIR, "coco-2017", "validation", "data")
print(f"Images directory: {images_dir}")

# Basic sanity check
if not os.path.isdir(images_dir):
	raise SystemExit(f"Images directory not found at expected path: {images_dir}")

num_samples = len(dataset)
print(f"COCO val2017 samples available: {num_samples}")
print("Done.") 