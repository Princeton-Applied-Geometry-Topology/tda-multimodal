import torch
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ripser
import persim


def plot_persistence_diagram(diagrams, output_path):
    """
    Generates and saves a persistence diagram plot.

    Args:
        diagrams (list): The list of persistence diagrams from ripser.
        output_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot H₀ and H₁
    persim.plot_diagrams(diagrams, show=False, title="Persistence Diagram")
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved persistence diagram to {output_path}")

def plot_pca_projection(data, output_path):
    """
    Generates and saves a 2D PCA projection scatter plot.

    Args:
        data (np.ndarray): The data points, shape (n_samples, n_features).
        output_path (str): Path to save the plot image.
    """
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7)
    plt.title("2D PCA Projection of Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.axis('equal')
    plt.grid(True)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PCA projection plot to {output_path}")

def generate_noisy_circle(n_points=100, noise=0.1):
    """Generates sample data in the shape of a noisy circle."""
    print("Generating sample noisy circle data...")
    thetas = np.linspace(0, 2 * np.pi, n_points)
    x = np.cos(thetas) + np.random.normal(0, noise, n_points)
    y = np.sin(thetas) + np.random.normal(0, noise, n_points)
    return np.stack([x, y], axis=1)

def main():
    """
    Main function to run the topology visualization pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Generate topological visualizations for high-dimensional data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        help="Path to the input data file (e.g., .pt or .npy)."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='topology_visuals',
        help="Directory to save the output plots."
    )
    parser.add_argument(
        '--homology_dim',
        type=int,
        default=1,
        help="Maximum homology dimension to compute (e.g., 1 for H₀ and H₁)."
    )
    parser.add_argument(
        '--max_points',
        type=int,
        default=5000,
        help="Maximum number of points to subsample from the data for TDA. Default is 5000."
    )
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load data or generate sample data
    if args.input_file:
        print(f"Loading data from {args.input_file}...")
        if args.input_file.endswith('.pt'):
            data_tensor = torch.load(args.input_file)
            # Assuming tensor can be on GPU, move to CPU and convert to numpy
            data = data_tensor.cpu().numpy()
        elif args.input_file.endswith('.npy'):
            data = np.load(args.input_file)
        else:
            raise ValueError("Unsupported file type. Please use .pt or .npy.")
        
        # Ensure data is 2D (n_samples, n_features) for topological analysis.
        # The raw data is likely (batch_size, seq_len, embed_dim).
        # We want to treat each token embedding as a point, so we reshape.
        if data.ndim == 3:
            print(f"Input data has 3 dimensions {data.shape}. Reshaping to (samples, features).")
            data = data.reshape(-1, data.shape[-1])
        elif data.ndim > 3:
            print(f"Warning: Input data has {data.ndim} dimensions. Flattening all but the last dimension.")
            data = data.reshape(-1, data.shape[-1])

    else:
        print("No input file provided.")
        data = generate_noisy_circle()

    # Subsample the data to make computation tractable
    if args.max_points and data.shape[0] > args.max_points:
        print(f"Data has {data.shape[0]} points. Subsampling to {args.max_points} points.")
        indices = np.random.choice(data.shape[0], args.max_points, replace=False)
        data = data[indices]

    print(f"Data shape for analysis: {data.shape}")

    # 1. Compute Persistence Homology
    print(f"Computing persistent homology up to H{args.homology_dim}...")
    diagrams = ripser.ripser(data, maxdim=args.homology_dim)['dgms']

    # 2. Visualize the Persistence Diagram
    diagram_path = os.path.join(args.output_dir, 'persistence_diagram.png')
    plot_persistence_diagram(diagrams, diagram_path)

    # 3. Visualize the PCA Projection
    pca_path = os.path.join(args.output_dir, 'pca_projection.png')
    plot_pca_projection(data, pca_path)
    
    print("\nDone. Check the 'topology_visuals' directory for your plots.")
    print("To run with your own data, use: ")
    print("python visualize_topology.py --input_file path/to/your/data.pt")

if __name__ == '__main__':
    main() 