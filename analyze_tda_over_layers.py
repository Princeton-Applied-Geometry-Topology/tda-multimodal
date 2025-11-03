# analyze_tda_over_layers.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import os
from tqdm import tqdm
# --- NEW IMPORT ---
import umap # <-- We will use UMAP for dimensionality reduction

# --- Config ---
DATA_DIR = "data/physics_experiment"
ACTIVATIONS_PATH = os.path.join(DATA_DIR, "all_activations.pt")
POINT_CLOUD_TYPE = "bound" # 'bound', 'color_only', or 'shape_only'
MAX_DIM = 1 # We only care about H0 (clusters) and H1 (loops)
N_LAYERS = 32 # Qwen-VL has 32 layers

# --- Load and Prepare Data ---
print(f"Loading activations from {ACTIVATIONS_PATH}...")
all_data = torch.load(ACTIVATIONS_PATH)

# We will build one point cloud for each layer
layer_point_clouds = []

# Filter for the data type we want to analyze
sample_ids = [
    id for id, data in all_data.items() 
    if data["metadata"]["type"] == POINT_CLOUD_TYPE
]
print(f"Found {len(sample_ids)} samples for type '{POINT_CLOUD_TYPE}'")

# --- THE FIX: We need a UMAP reducer ---
# We'll use the *same* reducer for all layers to compare them fairly
# n_neighbors=5 is a good default for 12 points. min_dist controls spread.
# n_components=3 gives us a 3D space to run TDA on.
# We set random_state for reproducibility.
reducer = umap.UMAP(
    n_neighbors=max(2, len(sample_ids) // 2), # Heuristic: ~half the points
    n_components=3, 
    min_dist=0.1,
    random_state=42,
    metric='cosine' # Use cosine distance *for the UMAP input*
)
print(f"Initialized UMAP reducer (metric=cosine, n_components=3)")

for i in range(N_LAYERS):
    layer_name = f"layer_{i}"
    cloud = [all_data[id]["activations"][layer_name] for id in sample_ids]
    cloud_np = torch.stack(cloud).numpy().astype(np.float64)
    layer_point_clouds.append(cloud_np)

# --- Run TDA per Layer ---
results_per_layer = []
print("Computing TDA per layer (with UMAP pre-processing)...")
for i in tqdm(range(N_LAYERS), desc="Computing TDA per layer"):
    cloud_high_dim = layer_point_clouds[i]
    
    # --- THE FIX: Apply UMAP first ---
    # Fit and transform the *first* layer, then just transform the rest
    # This keeps the "camera" (projection) the same.
    # A better way for deep layers: fit *all* data, then transform each.
    # Let's stack all layers, fit UMAP once, then analyze each transformed cloud.
    
    # --- REVISED, MORE ROBUST UMAP FITTING ---
    # (We'll fit on the *last* layer, which is often most structured)
    if i == 0:
        print("Fitting UMAP reducer on last layer (L31) data...")
        reducer.fit(layer_point_clouds[-1]) # Fit on the last layer
        
    # Now transform the *current* layer's data
    cloud_low_dim = reducer.transform(cloud_high_dim)
    
    # Run Ripser on the NEW 3-dimensional cloud
    # We use default Euclidean distance now, because in 3D it's meaningful.
    result = ripser(cloud_low_dim, maxdim=MAX_DIM) 
    results_per_layer.append(result)

# --- Analyze and Plot Evolution ---
# (This part is identical to before, but the results will be real)
print("Analyzing results...")

# 1. Number of H1 features (loops)
n_loops_per_layer = [len(res['dgms'][1]) for res in results_per_layer]

# 2. "Persistence" of the most persistent H1 feature
def get_max_persistence(dgms):
    if len(dgms) == 0:
        return 0
    lifetimes = dgms[:, 1] - dgms[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return np.max(lifetimes) if len(lifetimes) > 0 else 0

max_h1_persistence = [get_max_persistence(res['dgms'][1]) for res in results_per_layer]
max_h0_persistence = [get_max_persistence(res['dgms'][0]) for res in results_per_layer]

# 3. Plot the evolution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(N_LAYERS), n_loops_per_layer, 'o-')
plt.title(f"Number of $H_1$ Loops (Topology) vs. Layer\n(Point Cloud: {POINT_CLOUD_TYPE}, UMAP-3D)")
plt.xlabel("Model Layer")
plt.ylabel("Number of $H_1$ Features")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(N_LAYERS), max_h1_persistence, 'o-', color='r')
plt.title(f"Max $H_1$ Persistence (Loop 'Clarity') vs. Layer")
plt.xlabel("Model Layer")
plt.ylabel("Max $H_1$ Persistence (Death - Birth)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(N_LAYERS), max_h0_persistence, 'o-', color='g')
plt.title(f"Max $H_0$ Persistence ('Connectedness') vs. Layer")
plt.xlabel("Model Layer")
plt.ylabel("Max $H_0$ Persistence")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"tda_evolution_{POINT_CLOUD_TYPE}_umap.png")
print(f"Saved plot to tda_evolution_{POINT_CLOUD_TYPE}_umap.png")

# 4. Plot the persistence diagram for the "peak" layer
peak_layer = np.argmax(max_h1_persistence)
print(f"Peak $H_1$ persistence is at layer: {peak_layer}")

plt.figure()
plot_diagrams(results_per_layer[peak_layer]['dgms'], show=False)
plt.title(f"Persistence Diagram at Peak Layer {peak_layer} (UMAP-3D)")
plt.savefig(f"peak_layer_{peak_layer}_diagram_umap.png")
print(f"Saved diagram for peak layer {peak_layer}")