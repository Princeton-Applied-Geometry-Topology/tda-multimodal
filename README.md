# Geometric Phases of VLM Reasoning

This repository is an analysis of the internal geometric structure of Vision-Language Models (VLMs), specifically investigating how compositional concepts (like color and shape) are represented across layers.

The  hypothesis is that the model forms a "compositional manifold" (e.g., a product space like $S^1 \times S^1$) at a specific "binding phase" in its architecture. We use Topological Data Analysis (TDA) via UMAP and Persistent Homology (`ripser`) to find this structure.

## Project Structure

* `/data/physics_experiment_6x6/`: The synthetic 6x6 (color x shape) dataset.
    * `/images/`: 48 generated images.
    * `metadata.json`: Metadata for all 48 samples.
* `/qwen-vl-chat-local/`: A local copy of the `Qwen/Qwen-VL-Chat` model.
* `/tda_debug_output/`: The analysis results.
    * `/diagrams/`: Persistence diagrams for each layer.
    * `/point_clouds_3d/`: The 3D UMAP embeddings for each layer.
    * `summary_stats.json`: A log of TDA metrics for each layer.
    * `summary_evolution_plot.png`: A plot of persistence vs. layer.
* `generate_dataset.py`: Script to generate the synthetic 6x6 dataset.
* `download_model.py`: Script to download the VLM to the local directory.
* `extract_activations.py`: Script to run the model and save all layer activations.
* `debug_tda_pipeline.py`: **(Main Analysis)** Runs the UMAP+TDA pipeline on the activations and generates the debug results.
* `visualize_peak_layer.py`: **(Main Result)** Generates interactive 3D plots for the "peak" layer.

## Workflow & How to Run

1.  **Install Dependencies:**
    ```bash
    pip install torch transformers pillow numpy matplotlib ripser persim umap-learn plotly pandas scikit-learn
    ```

2.  **Generate Data:**
    ```bash
    python generate_dataset.py
    ```

3.  **Download Model (Run on Login Node):**
    ```bash
    python download_model.py
    ```

4.  **Extract Activations (Run on Compute Node):**
    * *Note: This script is pre-configured to use the `data/physics_experiment_6x6` directory and `qwen-vl-chat-local` model.*
    ```bash
    python extract_activations.py
    ```

5.  **Run TDA Analysis (Run on Compute Node):**
    * This script runs the full UMAP+TDA pipeline and saves results to `/tda_debug_output/`.
    ```bash
    python debug_tda_pipeline.py
    ```

6.  **Visualize Peak Layer (Run on Compute Node):**
    * Check the log from the previous step to find the `PEAK_LAYER`.
    * Edit `visualize_peak_layer.py` to set the `PEAK_LAYER` variable.
    ```bash
    python visualize_peak_layer.py
    ```
    * This generates `.html` files in `/tda_debug_output/`. Download and open these locally to see the 3D structure.