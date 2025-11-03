# visualize_peak_layer.py
import numpy as np
import plotly.express as px
import plotly.io as pio
import pandas as pd
import json
import os

# --- Config ---
PEAK_LAYER = 25  # <-- The layer we found from the log
DATA_DIR = "data/physics_experiment_6x6"
DEBUG_DIR = "tda-output"

# Set a default renderer for Plotly that works on clusters
pio.templates.default = "plotly_white"

# --- Load the 3D Point Cloud ---
cloud_file = os.path.join(DEBUG_DIR, "point_clouds_3d", f"layer_{PEAK_LAYER}_cloud.npy")
print(f"Loading 3D point cloud from {cloud_file}...")
cloud_3d = np.load(cloud_file)

# --- Load Metadata ---
metadata_file = os.path.join(DATA_DIR, "metadata.json")
print(f"Loading metadata from {metadata_file}...")
with open(metadata_file, 'r') as f:
    all_metadata = json.load(f)

# Filter for the 'bound' samples that are in our cloud
bound_metadata = [
    item for item in all_metadata if item["type"] == "bound"
]
print(f"Loaded {len(bound_metadata)} metadata entries.")

if len(bound_metadata) != cloud_3d.shape[0]:
    print(f"Error: Metadata count ({len(bound_metadata)}) does not match point cloud size ({cloud_3d.shape[0]})")
    exit()

# --- Create a Pandas DataFrame for Plotly ---
# This is the easiest way to pass data to Plotly
df = pd.DataFrame({
    'x': cloud_3d[:, 0],
    'y': cloud_3d[:, 1],
    'z': cloud_3d[:, 2],
    'color_label': [item['color'] for item in bound_metadata],
    'shape_label': [item['shape'] for item in bound_metadata],
    'hover_text': [item['id'] for item in bound_metadata] # Text to show on hover
})

# --- Plot 1: Colored by "Color" ---
print("Generating 3D plot colored by 'color'...")
fig_color = px.scatter_3d(
    df,
    x='x', y='y', z='z',
    color='color_label',       # Use the 'color_label' column for color
    symbol='shape_label',      # Use the 'shape_label' column for symbol
    hover_name='hover_text',   # Show the 'id' on hover
    title=f"Layer {PEAK_LAYER} UMAP Embedding (Colored by Color)"
)
fig_color.update_traces(marker_size=5)

# Save as an interactive HTML file
color_plot_file = os.path.join(DEBUG_DIR, f"layer_{PEAK_LAYER}_3D_plot_by_color.html")
fig_color.write_html(color_plot_file)
print(f"Saved color plot to {color_plot_file}")

# --- Plot 2: Colored by "Shape" ---
print("Generating 3D plot colored by 'shape'...")
fig_shape = px.scatter_3d(
    df,
    x='x', y='y', z='z',
    color='shape_label',       # Use the 'shape_label' column for color
    symbol='color_label',      # Use the 'color_label' column for symbol
    hover_name='hover_text',   # Show the 'id' on hover
    title=f"Layer {PEAK_LAYER} UMAP Embedding (Colored by Shape)"
)
fig_shape.update_traces(marker_size=5)

# Save as an interactive HTML file
shape_plot_file = os.path.join(DEBUG_DIR, f"layer_{PEAK_LAYER}_3D_plot_by_shape.html")
fig_shape.write_html(shape_plot_file)
print(f"Saved shape plot to {shape_plot_file}")