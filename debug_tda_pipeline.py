# debug_tda_pipeline.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shutil
from tqdm import tqdm
import umap
from ripser import ripser
from persim import plot_diagrams
import pandas as pd # <-- NEW IMPORT

# --- NEW IMPORT for Silhouette Score ---
from sklearn.metrics import silhouette_score

# --- Config ---
DATA_DIR = "data/physics_experiment_6x6"
ACTIVATIONS_PATH = os.path.join(DATA_DIR, "all_activations.pt")
POINT_CLOUD_TYPE = "bound"
MAX_DIM = 1
N_LAYERS = 32

# --- DEBUG OUTPUT SETUP ---
DEBUG_DIR = "tda_debug_output"
DIAGRAM_DIR = os.path.join(DEBUG_DIR, "diagrams")
CLOUD_DIR = os.path.join(DEBUG_DIR, "point_clouds_3d")

if os.path.exists(DEBUG_DIR):
    shutil.rmtree(DEBUG_DIR)
os.makedirs(DIAGRAM_DIR, exist_ok=True)
os.makedirs(CLOUD_DIR, exist_ok=True)

print(f"Debug output will be saved to: {DEBUG_DIR}")

# --- Load and Prepare Data ---
print(f"Loading activations from {ACTIVATIONS_PATH}...")
all_data = torch.load(ACTIVATIONS_PATH)

# --- NEW: Load metadata to get labels for scoring ---
metadata_file = os.path.join(DATA_DIR, "metadata.json")
with open(metadata_file, 'r') as f:
    all_metadata = json.load(f)

# Get sorted list of IDs, color labels, and shape labels
sample_ids = sorted([
    id for id, data in all_data.items() 
    if data["metadata"]["type"] == POINT_CLOUD_TYPE
])
# Create a map for quick lookup
metadata_map = {item['id']: item for item in all_metadata}
color_labels = [metadata_map[id]['color'] for id in sample_ids]
shape_labels = [metadata_map[id]['shape'] for id in sample_ids]
# --- End new label loading ---

N_SAMPLES = len(sample_ids)
print(f"Found {N_SAMPLES} samples for type '{POINT_CLOUD_TYPE}'")

layer_point_clouds_high_dim = []
for i in range(N_LAYERS):
    layer_name = f"layer_{i}"
    # Ensure we get activations in the same order as our labels
    cloud = [all_data[id]["activations"][layer_name] for id in sample_ids]
    cloud_np = torch.stack(cloud).numpy().astype(np.float64)
    layer_point_clouds_high_dim.append(cloud_np)

print(f"Prepared {len(layer_point_clouds_high_dim)} high-dim point clouds.")

# --- Main TDA Loop ---
all_layer_stats = []
summary_plot_data = {
    "n_h1": [],
    "max_h1_pers": [],
    "max_h0_pers": [],
    "shape_silhouette": [], # <-- NEW
    "color_silhouette": []  # <-- NEW
}

def get_persistence(dgms):
    if dgms.shape[0] == 0:
        return np.array([]), 0.0
    
    pers = dgms[:, 1] - dgms[:, 0]
    pers = pers[np.isfinite(pers)]
    
    if pers.shape[0] == 0:
        return np.array([]), 0.0
        
    return pers, np.max(pers)

print("\n--- Starting 'Obnoxiously Thorough' TDA Pipeline (v-final w/ Silhouette) ---")
for i in tqdm(range(N_LAYERS), desc="Processing Layers"):
    
    cloud_high_dim = layer_point_clouds_high_dim[i]
    
    reducer = umap.UMAP(
        n_neighbors=6,
        n_components=3,
        min_dist=0.1,
        random_state=42,
        metric='cosine'
    )
    
    cloud_low_dim = reducer.fit_transform(cloud_high_dim)
    
    cloud_save_path = os.path.join(CLOUD_DIR, f"layer_{i}_cloud.npy")
    np.save(cloud_save_path, cloud_low_dim)
    
    result = ripser(cloud_low_dim, maxdim=MAX_DIM)
    dgms = result['dgms']
    
    h0_pers, max_h0_pers = get_persistence(dgms[0])
    h1_pers, max_h1_pers = get_persistence(dgms[1])
    n_h1_features = len(h1_pers)

    # --- NEW: Calculate Silhouette Scores ---
    score_shape = silhouette_score(cloud_low_dim, shape_labels)
    score_color = silhouette_score(cloud_low_dim, color_labels)
    # --- End new score calculation ---

    layer_stats = {
        "layer": i,
        "n_h1_features": n_h1_features,
        "max_h1_persistence": max_h1_pers,
        "all_h1_persistence_values": h1_pers.tolist(),
        "n_h0_features": len(dgms[0]) - len(h0_pers),
        "max_h0_persistence": max_h0_pers,
        "silhouette_shape": score_shape, # <-- NEW
        "silhouette_color": score_color  # <-- NEW
    }
    all_layer_stats.append(layer_stats)
    
    print(f"\n--- Layer {i} Stats ---")
    print(f"  Max H1 Pers: {max_h1_pers:.4f} (n={n_h1_features})")
    print(f"  Max H0 Pers: {max_h0_pers:.4f}")
    print(f"  SILHOUETTE (Shape): {score_shape:.4f}") # <-- NEW
    print(f"  SILHOUETTE (Color): {score_color:.4f}") # <-- NEW
    
    plt.figure(figsize=(7, 7))
    plot_diagrams(dgms, show=False)
    plt.title(f"Layer {i} Diagram | Shape Score: {score_shape:.2f} | Color Score: {score_color:.2f}")
    diag_save_path = os.path.join(DIAGRAM_DIR, f"layer_{i}_diagram.png")
    plt.savefig(diag_save_path)
    plt.close()
    
    summary_plot_data["n_h1"].append(n_h1_features)
    summary_plot_data["max_h1_pers"].append(max_h1_pers)
    summary_plot_data["max_h0_pers"].append(max_h0_pers)
    summary_plot_data["shape_silhouette"].append(score_shape) # <-- NEW
    summary_plot_data["color_silhouette"].append(score_color) # <-- NEW

print("\n--- Pipeline Complete ---")

json_path = os.path.join(DEBUG_DIR, "summary_stats.json")
with open(json_path, 'w') as f:
    json.dump(all_layer_stats, f, indent=2)
print(f"Saved detailed stats to {json_path}")

# --- NEW: Updated 2x2 summary plot ---
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(range(N_LAYERS), summary_plot_data["max_h1_pers"], 'o-', color='r')
plt.title(f"Max $H_1$ Persistence vs. Layer")
plt.ylabel("Max Persistence (Death - Birth)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(range(N_LAYERS), summary_plot_data["n_h1"], 'o-', color='b')
plt.title(f"Number of $H_1$ Loops vs. Layer")
plt.ylabel("Number of $H_1$ Features")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(range(N_LAYERS), summary_plot_data["shape_silhouette"], 'o-', label='Shape Score', color='purple')
plt.plot(range(N_LAYERS), summary_plot_data["color_silhouette"], 'o-', label='Color Score', color='orange')
plt.title(f"Clustering Score vs. Layer")
plt.ylabel("Silhouette Score")
plt.xlabel("Model Layer")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(range(N_LAYERS), summary_plot_data["max_h0_pers"], 'o-', color='g')
plt.title(f"Max $H_0$ Persistence vs. Layer")
plt.ylabel("Max Persistence")
plt.xlabel("Model Layer")
plt.grid(True)

plt.tight_layout()
summary_plot_path = os.path.join(DEBUG_DIR, "summary_evolution_plot.png")
plt.savefig(summary_plot_path)
print(f"Saved summary plot to {summary_plot_path}")

peak_layer = np.argmax(summary_plot_data["shape_silhouette"])
print(f"\n--- Overall Result ---")
print(f"Peak *Shape Silhouette Score* is at layer: {peak_layer}")
print(f"Stats for peak layer {peak_layer}:")
print(json.dumps(all_layer_stats[peak_layer], indent=2))