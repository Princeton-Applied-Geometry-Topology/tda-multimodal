import torch
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ripser
import persim


def farthest_point_sampling(data, n_samples, distance_metric='euclidean'):
    """
    Implements Farthest Point Sampling to preserve topological structure.
    
    Args:
        data (np.ndarray): Input data points, shape (n_points, n_features)
        n_samples (int): Number of points to sample
        distance_metric (str): 'euclidean' or 'cosine'
        
    Returns:
        np.ndarray: Indices of sampled points
    """
    n_points = data.shape[0]
    
    if n_samples >= n_points:
        return np.arange(n_points)
    
    if distance_metric == 'cosine':
        # Ensure data is float32 for numerical stability
        if data.dtype != np.float32:
            data = data.astype(np.float32)
            
        # Normalize to unit vectors for cosine distance
        # Handle edge cases where norms might be very small or zero
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        
        # Check for problematic cases
        zero_norms = norms < 1e-10
        if np.any(zero_norms):
            print(f"Warning: {np.sum(zero_norms)} points have near-zero norms. These will be handled specially.")
            # For zero-norm points, use a small random vector to avoid division issues
            data_normalized = data.copy()
            data_normalized[zero_norms.flatten()] = np.random.normal(0, 1e-6, size=(np.sum(zero_norms), data.shape[1]))
            norms[zero_norms] = np.linalg.norm(data_normalized[zero_norms.flatten()], axis=1, keepdims=True)
        else:
            data_normalized = data.copy()
        
        # Normalize with robust division
        data_normalized = data_normalized / (norms + 1e-12)
        
        # Initialize with a random point
        indices = np.zeros(n_samples, dtype=int)
        indices[0] = np.random.randint(0, n_points)
        
        # Compute cosine distances (1 - cosine similarity)
        distances = 1 - np.dot(data_normalized, data_normalized[indices[0]])
        
        # Iteratively select farthest points
        for i in range(1, n_samples):
            farthest_idx = np.argmax(distances)
            indices[i] = farthest_idx
            
            # Update distances
            new_distances = 1 - np.dot(data_normalized, data_normalized[farthest_idx])
            distances = np.minimum(distances, new_distances)
            
    else:  # euclidean
        # Normalize data to prevent numerical overflow
        data_normalized = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        
        # Initialize with a random point
        indices = np.zeros(n_samples, dtype=int)
        indices[0] = np.random.randint(0, n_points)
        
        # Compute distances using more stable computation
        distances = np.sum((data_normalized - data_normalized[indices[0]])**2, axis=1)
        
        # Iteratively select farthest points
        for i in range(1, n_samples):
            farthest_idx = np.argmax(distances)
            indices[i] = farthest_idx
            
            # Update distances
            new_distances = np.sum((data_normalized - data_normalized[farthest_idx])**2, axis=1)
            distances = np.minimum(distances, new_distances)
    
    return indices


def uniform_subsampling(data, n_samples):
    """
    Simple uniform subsampling that's guaranteed to work with any data.
    
    Args:
        data (np.ndarray): Input data
        n_samples (int): Number of samples to keep
        
    Returns:
        np.ndarray: Subsampled data
    """
    n_points = data.shape[0]
    if n_samples >= n_points:
        return data
    
    # Use uniform spacing to ensure good coverage
    step = n_points / n_samples
    indices = np.array([int(i * step) for i in range(n_samples)])
    return data[indices]


def subsample_data(data, n_samples, method='fps', distance_metric='cosine'):
    """
    Subsample data using the specified method.
    
    Args:
        data (np.ndarray): Input data
        n_samples (int): Number of samples to keep
        method (str): 'fps' for Farthest Point Sampling, 'random' for random sampling, 'robust' for FPS with fallback, 'uniform' for uniform spacing
        distance_metric (str): 'euclidean' or 'cosine' (only used with FPS)
        
    Returns:
        np.ndarray: Subsampled data
    """
    if data.shape[0] <= n_samples:
        return data
    
    if method == 'fps':
        try:
            print(f"Using Farthest Point Sampling with {distance_metric} distance to subsample from {data.shape[0]} to {n_samples} points...")
            indices = farthest_point_sampling(data, n_samples, distance_metric=distance_metric)
        except Exception as e:
            print(f"FPS failed with error: {e}. Falling back to random sampling.")
            indices = np.random.choice(data.shape[0], n_samples, replace=False)
            
    elif method == 'robust':
        try:
            print(f"Attempting Farthest Point Sampling with {distance_metric} distance...")
            indices = farthest_point_sampling(data, n_samples, distance_metric=distance_metric)
        except Exception as e:
            print(f"FPS failed, falling back to uniform sampling: {e}")
            return uniform_subsampling(data, n_samples)
            
    elif method == 'uniform':
        print(f"Using uniform subsampling to subsample from {data.shape[0]} to {n_samples} points...")
        return uniform_subsampling(data, n_samples)
        
    elif method == 'random':
        print(f"Using random sampling to subsample from {data.shape[0]} to {n_samples} points...")
        indices = np.random.choice(data.shape[0], n_samples, replace=False)
    else:
        raise ValueError(f"Unknown subsampling method: {method}")
    
    return data[indices]


def preprocess_activations(data):
    """
    Preprocess activations to handle extreme values and ensure numerical stability.
    
    Args:
        data (np.ndarray): Raw activation data
        
    Returns:
        np.ndarray: Preprocessed data
    """
    print(f"Preprocessing activations: shape {data.shape}, dtype {data.dtype}")
    
    # Convert float16 to float32 for better numerical stability
    if data.dtype == np.float16:
        print("Converting float16 to float32 for better numerical stability...")
        data = data.astype(np.float32)
    
    # Check for extreme values
    if np.isfinite(data).all():
        print("All values are finite ✓")
    else:
        print(f"Warning: {np.sum(~np.isfinite(data))} non-finite values found. Replacing with zeros.")
        data = np.where(np.isfinite(data), data, 0.0)
    
    # Check for extreme magnitudes
    abs_data = np.abs(data)
    max_val = np.max(abs_data)
    min_val = np.min(abs_data)
    print(f"Value range: [{min_val:.2e}, {max_val:.2e}]")
    
    # If values are extremely large, consider clipping or scaling
    if max_val > 1e6:
        print(f"Warning: Very large values detected (max: {max_val:.2e}). Consider using cosine distance.")
    
    return data


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
    parser.add_argument(
        '--subsample_method',
        type=str,
        choices=['fps', 'random', 'robust', 'uniform'],
        default='robust',
        help="Subsampling method: 'fps' (Farthest Point Sampling), 'random', 'robust' (FPS with fallback), or 'uniform' (guaranteed uniform). Default is 'robust' for reliability."
    )
    parser.add_argument(
        '--distance_metric',
        type=str,
        choices=['cosine', 'euclidean'],
        default='cosine',
        help="Distance metric for FPS: 'cosine' (recommended for embeddings) or 'euclidean'. Default is 'cosine'."
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

    # Preprocess activations to handle extreme values
    data = preprocess_activations(data)

    # Subsample the data to make computation tractable
    if args.max_points and data.shape[0] > args.max_points:
        data = subsample_data(data, args.max_points, method=args.subsample_method, distance_metric=args.distance_metric)

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