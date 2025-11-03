# generate_dataset.py (v3 - More Data)
import json
import os
from PIL import Image, ImageDraw

# --- NEW 6x6 CONFIG ---
COLORS = {
    "red": (255, 60, 60),
    "green": (60, 255, 60),
    "blue": (60, 60, 255),
    "yellow": (255, 255, 60),
    "cyan": (60, 255, 255),
    "magenta": (255, 60, 255),
    "grey": (128, 128, 128) # For controls
}
SHAPES = ["cube", "sphere", "pyramid", "cone", "torus", "cylinder"]
DATA_DIR = "data/physics_experiment_6x6" # <-- New directory
IMG_DIR = os.path.join(DATA_DIR, "images")

# Ensure directories exist
os.makedirs(IMG_DIR, exist_ok=True)

def create_image(img_path, color_name, shape_name):
    """Creates a simple image with a colored shape."""
    img = Image.new('RGB', (200, 200), color='grey')
    draw = ImageDraw.Draw(img)
    
    color_rgb = COLORS[color_name]
    
    if shape_name == "cube":
        draw.rectangle([50, 50, 150, 150], fill=color_rgb, outline="black")
    elif shape_name == "sphere":
        draw.ellipse([50, 50, 150, 150], fill=color_rgb, outline="black")
    elif shape_name == "pyramid":
        draw.polygon([(100, 50), (50, 150), (150, 150)], fill=color_rgb, outline="black")
    elif shape_name == "cone":
        # A cone is just a pyramid with an ellipse base, but we'll cheat
        draw.polygon([(100, 50), (40, 150), (160, 150)], fill=color_rgb, outline="black")
    elif shape_name == "torus":
        # Draw a thick ellipse
        draw.ellipse([50, 50, 150, 150], fill=None, outline=color_rgb, width=20)
    elif shape_name == "cylinder":
        draw.rectangle([60, 50, 140, 150], fill=color_rgb, outline="black")
        draw.ellipse([60, 40, 140, 60], fill=color_rgb, outline="black")
        
    img.save(img_path)

def generate_data():
    """Generates all combinations and saves metadata."""
    metadata = []
    
    # 1. Create the 'Cloud_Bound' dataset (6x6 = 36 samples)
    for color in [c for c in COLORS if c != 'grey']:
        for shape in SHAPES:
            img_id = f"{color}_{shape}"
            img_path = os.path.join(IMG_DIR, f"{img_id}.png")
            create_image(img_path, color, shape)
            
            metadata.append({
                "id": img_id,
                "image_path": img_path,
                "prompt": f"a photo of a {color} {shape}",
                "type": "bound",
                "color": color,
                "shape": shape
            })
            
    # 2. Create 'Cloud_Color' (6 samples)
    for color in [c for c in COLORS if c != 'grey']:
        img_id = f"{color}_object"
        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
        create_image(img_path, color, "cube") 
        
        metadata.append({
            "id": img_id,
            "image_path": img_path,
            "prompt": f"a photo of a {color} object",
            "type": "color_only",
            "color": color,
            "shape": "unknown"
        })

    # 3. Create 'Cloud_Shape' (6 samples)
    for shape in SHAPES:
        img_id = f"grey_{shape}"
        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
        create_image(img_path, "grey", shape)
        
        metadata.append({
            "id": img_id,
            "image_path": img_path,
            "prompt": f"a photo of a grey {shape}",
            "type": "shape_only",
            "color": "grey",
            "shape": shape
        })

    # Save metadata
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Generated {len(metadata)} samples in {DATA_DIR}")

if __name__ == "__main__":
    generate_data()