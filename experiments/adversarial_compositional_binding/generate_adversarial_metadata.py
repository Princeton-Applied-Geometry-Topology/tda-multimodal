# generate_adversarial_metadata.py
# Creates adversarial image-text pairs for compositional binding experiment
import json
import os
from itertools import product

# Get project root (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data/physics_experiment_6x6")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "adversarial_metadata.json")

COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta"]
SHAPES = ["cube", "sphere", "pyramid", "cone", "torus", "cylinder"]

print(f"Loading base metadata from {METADATA_PATH}...")
with open(METADATA_PATH, 'r') as f:
    base_metadata = json.load(f)

# Create lookup for images
image_lookup = {}
for item in base_metadata:
    if item["type"] == "bound":
        key = (item["color"], item["shape"])
        image_lookup[key] = item["image_path"]

# Generate all adversarial conditions
adversarial_samples = []

print("Generating adversarial conditions...")
for img_color, img_shape in product(COLORS, SHAPES):
    image_path = image_lookup.get((img_color, img_shape))
    if not image_path:
        print(f"Warning: No image found for {img_color} {img_shape}")
        continue
    
    base_id = f"{img_color}_{img_shape}"
    
    # Condition 1: MATCHED (control)
    adversarial_samples.append({
        "id": f"{base_id}_matched",
        "base_id": base_id,
        "image_path": image_path,
        "prompt": f"a photo of a {img_color} {img_shape}",
        "condition": "matched",
        "img_color": img_color,
        "img_shape": img_shape,
        "txt_color": img_color,
        "txt_shape": img_shape,
        "color_match": True,
        "shape_match": True
    })
    
    # Condition 2: COLOR MISMATCH
    for txt_color in COLORS:
        if txt_color != img_color:
            adversarial_samples.append({
                "id": f"{base_id}_color_{txt_color}",
                "base_id": base_id,
                "image_path": image_path,
                "prompt": f"a photo of a {txt_color} {img_shape}",
                "condition": "color_mismatch",
                "img_color": img_color,
                "img_shape": img_shape,
                "txt_color": txt_color,
                "txt_shape": img_shape,
                "color_match": False,
                "shape_match": True
            })
    
    # Condition 3: SHAPE MISMATCH
    for txt_shape in SHAPES:
        if txt_shape != img_shape:
            adversarial_samples.append({
                "id": f"{base_id}_shape_{txt_shape}",
                "base_id": base_id,
                "image_path": image_path,
                "prompt": f"a photo of a {img_color} {txt_shape}",
                "condition": "shape_mismatch",
                "img_color": img_color,
                "img_shape": img_shape,
                "txt_color": img_color,
                "txt_shape": txt_shape,
                "color_match": True,
                "shape_match": False
            })
    
    # Condition 4: BOTH MISMATCH (systematic subset to avoid explosion)
    # For each base, create a few strategic both-mismatch examples
    # Option A: Same color family but different shape
    other_colors = [c for c in COLORS if c != img_color]
    other_shapes = [s for s in SHAPES if s != img_shape]
    
    # Create a balanced subset: one per color mismatch + one per shape mismatch
    # This gives us systematic coverage without too many samples
    for txt_color, txt_shape in product(other_colors[:3], other_shapes[:3]):  # Limit to 3x3 = 9 per base
        adversarial_samples.append({
            "id": f"{base_id}_both_{txt_color}_{txt_shape}",
            "base_id": base_id,
            "image_path": image_path,
            "prompt": f"a photo of a {txt_color} {txt_shape}",
            "condition": "both_mismatch",
            "img_color": img_color,
            "img_shape": img_shape,
            "txt_color": txt_color,
            "txt_shape": txt_shape,
            "color_match": False,
            "shape_match": False
        })

print(f"\nGenerated {len(adversarial_samples)} adversarial samples:")
condition_counts = {}
for sample in adversarial_samples:
    cond = sample["condition"]
    condition_counts[cond] = condition_counts.get(cond, 0) + 1

for cond, count in sorted(condition_counts.items()):
    print(f"  {cond}: {count} samples")

print(f"\nSaving to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, 'w') as f:
    json.dump(adversarial_samples, f, indent=2)

print("Done!")

