# analyze_adversarial_tda.py
# Performs TDA analysis on adversarial conditions and compares results
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shutil
from tqdm import tqdm
from collections import defaultdict
import umap
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics import silhouette_score
import pandas as pd

# --- Config ---
# Get project root (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data/physics_experiment_6x6")
ACTIVATIONS_PATH = os.path.join(DATA_DIR, "adversarial_activations.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tda_adversarial_output")
N_LAYERS = 32
MAX_DIM = 1

# Create output directories
for subdir in ["matched", "color_mismatch", "shape_mismatch", "both_mismatch", "comparison"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir, "diagrams"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, subdir, "point_clouds"), exist_ok=True)

print(f"Loading activations from {ACTIVATIONS_PATH}...")
all_data = torch.load(ACTIVATIONS_PATH)

# Organize by condition
data_by_condition = defaultdict(list)
condition_metadata = defaultdict(list)

for sample_id, data in all_data.items():
    condition = data["metadata"]["condition"]
    data_by_condition[condition].append((sample_id, data))
    condition_metadata[condition].append(data["metadata"])

print("\nSamples per condition:")
for cond, samples in sorted(data_by_condition.items()):
    print(f"  {cond}: {len(samples)} samples")

def get_persistence(dgms):
    if dgms.shape[0] == 0:
        return np.array([]), 0.0
    pers = dgms[:, 1] - dgms[:, 0]
    pers = pers[np.isfinite(pers)]
    if pers.shape[0] == 0:
        return np.array([]), 0.0
    return pers, np.max(pers)

def compute_tda_for_condition(condition, samples, output_subdir):
    """Compute TDA for a specific condition."""
    print(f"\n--- Analyzing {condition} ---")
    
    # Sort samples by ID for consistent ordering
    samples = sorted(samples, key=lambda x: x[0])
    sample_ids = [s[0] for s in samples]
    
    # Get labels for silhouette scoring
    img_colors = [s[1]["metadata"]["img_color"] for s in samples]
    img_shapes = [s[1]["metadata"]["img_shape"] for s in samples]
    txt_colors = [s[1]["metadata"]["txt_color"] for s in samples]
    txt_shapes = [s[1]["metadata"]["txt_shape"] for s in samples]
    
    # Prepare high-dim point clouds
    layer_point_clouds = []
    for i in range(N_LAYERS):
        layer_name = f"layer_{i}"
        cloud = [s[1]["activations"][layer_name] for s in samples]
        cloud_np = torch.stack(cloud).numpy().astype(np.float64)
        layer_point_clouds.append(cloud_np)
    
    # TDA loop
    all_stats = []
    for i in tqdm(range(N_LAYERS), desc=f"Processing {condition}"):
        cloud_high_dim = layer_point_clouds[i]
        
        reducer = umap.UMAP(
            n_neighbors=min(6, len(samples)-1),
            n_components=3,
            min_dist=0.1,
            random_state=42,
            metric='cosine'
        )
        
        cloud_low_dim = reducer.fit_transform(cloud_high_dim)
        
        # Save point cloud
        cloud_save_path = os.path.join(output_subdir, "point_clouds", f"layer_{i}_cloud.npy")
        np.save(cloud_save_path, cloud_low_dim)
        
        # Compute persistence
        result = ripser(cloud_low_dim, maxdim=MAX_DIM)
        dgms = result['dgms']
        
        h0_pers, max_h0_pers = get_persistence(dgms[0])
        h1_pers, max_h1_pers = get_persistence(dgms[1])
        n_h1_features = len(h1_pers)
        
        # Silhouette scores
        score_img_color = silhouette_score(cloud_low_dim, img_colors)
        score_img_shape = silhouette_score(cloud_low_dim, img_shapes)
        score_txt_color = silhouette_score(cloud_low_dim, txt_colors)
        score_txt_shape = silhouette_score(cloud_low_dim, txt_shapes)
        
        stats = {
            "layer": i,
            "n_h1_features": n_h1_features,
            "max_h1_persistence": float(max_h1_pers),
            "max_h0_persistence": float(max_h0_pers),
            "silhouette_img_color": float(score_img_color),
            "silhouette_img_shape": float(score_img_shape),
            "silhouette_txt_color": float(score_txt_color),
            "silhouette_txt_shape": float(score_txt_shape)
        }
        all_stats.append(stats)
        
        # Plot persistence diagram
        plt.figure(figsize=(7, 7))
        plot_diagrams(dgms, show=False)
        plt.title(f"{condition} - Layer {i} | H1={n_h1_features} | Max Pers={max_h1_pers:.3f}")
        diag_path = os.path.join(output_subdir, "diagrams", f"layer_{i}_diagram.png")
        plt.savefig(diag_path)
        plt.close()
    
    # Save stats
    stats_path = os.path.join(output_subdir, "layer_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    return all_stats

# Analyze each condition
condition_stats = {}
for condition in ["matched", "color_mismatch", "shape_mismatch", "both_mismatch"]:
    if condition not in data_by_condition:
        print(f"Warning: No samples for {condition}")
        continue
    
    output_subdir = os.path.join(OUTPUT_DIR, condition)
    stats = compute_tda_for_condition(
        condition, 
        data_by_condition[condition],
        output_subdir
    )
    condition_stats[condition] = stats

# --- Comparison Analysis ---
print("\n--- Generating Comparison Plots ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Max H1 Persistence by condition
ax = axes[0, 0]
for condition in ["matched", "color_mismatch", "shape_mismatch", "both_mismatch"]:
    if condition in condition_stats:
        pers_values = [s["max_h1_persistence"] for s in condition_stats[condition]]
        ax.plot(range(N_LAYERS), pers_values, 'o-', label=condition, linewidth=2)
ax.set_title("Max H1 Persistence by Condition")
ax.set_xlabel("Layer")
ax.set_ylabel("Max Persistence")
ax.legend()
ax.grid(True)

# Plot 2: Number of H1 features
ax = axes[0, 1]
for condition in ["matched", "color_mismatch", "shape_mismatch", "both_mismatch"]:
    if condition in condition_stats:
        n_features = [s["n_h1_features"] for s in condition_stats[condition]]
        ax.plot(range(N_LAYERS), n_features, 'o-', label=condition, linewidth=2)
ax.set_title("Number of H1 Features by Condition")
ax.set_xlabel("Layer")
ax.set_ylabel("Number of Features")
ax.legend()
ax.grid(True)

# Plot 3: Image Color Silhouette (should cluster by actual image)
ax = axes[0, 2]
for condition in ["matched", "color_mismatch", "shape_mismatch", "both_mismatch"]:
    if condition in condition_stats:
        scores = [s["silhouette_img_color"] for s in condition_stats[condition]]
        ax.plot(range(N_LAYERS), scores, 'o-', label=condition, linewidth=2)
ax.set_title("Image Color Clustering (by actual image)")
ax.set_xlabel("Layer")
ax.set_ylabel("Silhouette Score")
ax.legend()
ax.grid(True)

# Plot 4: Text Color Silhouette (tests if model follows text)
ax = axes[1, 0]
for condition in ["matched", "color_mismatch", "shape_mismatch", "both_mismatch"]:
    if condition in condition_stats:
        scores = [s["silhouette_txt_color"] for s in condition_stats[condition]]
        ax.plot(range(N_LAYERS), scores, 'o-', label=condition, linewidth=2)
ax.set_title("Text Color Clustering (by text prompt)")
ax.set_xlabel("Layer")
ax.set_ylabel("Silhouette Score")
ax.legend()
ax.grid(True)

# Plot 5: Image Shape Silhouette
ax = axes[1, 1]
for condition in ["matched", "color_mismatch", "shape_mismatch", "both_mismatch"]:
    if condition in condition_stats:
        scores = [s["silhouette_img_shape"] for s in condition_stats[condition]]
        ax.plot(range(N_LAYERS), scores, 'o-', label=condition, linewidth=2)
ax.set_title("Image Shape Clustering")
ax.set_xlabel("Layer")
ax.set_ylabel("Silhouette Score")
ax.legend()
ax.grid(True)

# Plot 6: Difference between matched and mismatched (key metric!)
ax = axes[1, 2]
if "matched" in condition_stats:
    matched_pers = np.array([s["max_h1_persistence"] for s in condition_stats["matched"]])
    for condition in ["color_mismatch", "shape_mismatch", "both_mismatch"]:
        if condition in condition_stats:
            mismatch_pers = np.array([s["max_h1_persistence"] for s in condition_stats[condition]])
            diff = matched_pers - mismatch_pers
            ax.plot(range(N_LAYERS), diff, 'o-', label=f"{condition} disruption", linewidth=2)
ax.set_title("Persistence Disruption: Matched - Mismatched")
ax.set_xlabel("Layer")
ax.set_ylabel("Persistence Difference")
ax.legend()
ax.grid(True)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
comparison_path = os.path.join(OUTPUT_DIR, "comparison", "all_conditions_comparison.png")
plt.savefig(comparison_path)
print(f"Saved comparison plot to {comparison_path}")

# Save summary
summary = {
    "condition_stats": {k: v for k, v in condition_stats.items()},
    "n_samples_per_condition": {k: len(data_by_condition[k]) for k in data_by_condition.keys()}
}
summary_path = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n--- Analysis Complete ---")
print(f"Results saved to: {OUTPUT_DIR}")

