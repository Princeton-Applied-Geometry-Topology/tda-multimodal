import torch
import numpy as np
import re

@torch.jit.script
def compute_effective_dimensionality(activations_batch: torch.Tensor) -> torch.Tensor:
    """Computes normalized effective dimensionality for each sample in a batch using singular values.
    
    Uses the participation ratio formula normalized by dimension: 
    Normalized ED = [(Σσᵢ)² / Σσᵢ²] / min_dim
    where σᵢ are the singular values and min_dim is min(n_samples, embed_dim).

    This gives a value between 0 and 1, where:
    - 1.0 = all dimensions equally used (uniform singular values)  
    - closer to 0 = fewer dimensions effectively used

    Args:
        activations_batch: Tensor of shape (batch_size, n_samples, embed_dim)
                           where n_samples could be sequence length.
    Returns:
        Tensor of shape (batch_size,) containing normalized effective dimensionality estimates.
    """
    # Ensure input is float32 for SVD
    activations_batch = activations_batch.to(torch.float32)

    # Compute SVD values for each matrix in the batch
    S = torch.linalg.svdvals(activations_batch) # Shape: (batch_size, min(n_samples, embed_dim))

    # Sum of singular values for each matrix
    sum_S = torch.sum(S, dim=1) # Shape: (batch_size,)
    # Sum of squared singular values for each matrix
    sum_S_squared = torch.sum(S**2, dim=1) # Shape: (batch_size,)

    # Calculate the participation ratio: (Σσᵢ)² / Σσᵢ²
    # Clamp sum_S_squared to avoid division by zero
    participation_ratio = (sum_S**2) / torch.clamp(sum_S_squared, min=1e-10)

    # Normalize by the number of singular values (min dimension) for interpretability
    min_dim = float(min(activations_batch.shape[1:])) # min(n_samples, embed_dim)
    
    # Clamp min_dim to avoid division by zero if dimensions are 0 or 1 in edge cases
    normalized_effective_dims = participation_ratio / max(min_dim, 1.0)

    return normalized_effective_dims


@torch.jit.script
def compute_fixed_window_ed(activations_batch: torch.Tensor, n_windows: int) -> torch.Tensor:
    """Computes effective dimensionality over N fixed, non-overlapping windows.

    Args:
        activations_batch: Tensor of shape (batch_size, seq_len, embed_dim).
        n_windows: The number of fixed windows to divide the sequence into.

    Returns:
        Tensor of shape (batch_size, n_windows) containing ED estimates for each window.
    """
    batch_size, seq_len, embed_dim = activations_batch.shape

    # Ensure n_windows is valid
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if n_windows > seq_len:
        # Handle cases where seq_len is very small or n_windows is large
        # Option: Return NaNs, zeros, or raise error. Let's return zeros for now.
        # Or compute ED for the full sequence as one window?
        # For consistency, let's compute for the full sequence if n_windows > seq_len
        # This might not be ideal, consider adjusting based on desired behavior.
        # Alternative: Pad? But padding affects ED.
        # Let's compute for the full sequence as a single window if n_windows > seq_len
        # This requires reshaping the output later.
        # Simpler: Truncate n_windows to seq_len if needed.
        n_windows = seq_len # Each token becomes a window

    window_size = seq_len // n_windows
    remainder = seq_len % n_windows

    # Truncate the sequence to be divisible by n_windows for simplicity
    # This drops the last 'remainder' tokens.
    # Alternative: Distribute remainder tokens, but complicates batching.
    truncated_len = n_windows * window_size
    if truncated_len == 0:
         # Handle edge case where window_size is 0 (seq_len < n_windows)
         # Return ED of the full sequence, perhaps repeated or averaged?
         # Let's return the full sequence ED repeated.
         full_seq_ed = compute_effective_dimensionality(activations_batch) # Shape (batch_size,)
         # Expand to (batch_size, n_windows), repeating the value.
         # This might be misleading. Consider returning NaNs or a specific indicator.
         # For now, let's return the full sequence ED repeated.
         return full_seq_ed.unsqueeze(1).expand(-1, n_windows)


    truncated_activations = activations_batch[:, :truncated_len, :]

    # Reshape into windows: (batch_size, n_windows, window_size, embed_dim)
    windows = truncated_activations.reshape(batch_size, n_windows, window_size, embed_dim)

    # Reshape for batch ED computation: (batch_size * n_windows, window_size, embed_dim)
    # Permute to bring n_windows first: (n_windows, batch_size, window_size, embed_dim)
    # Then reshape: (n_windows * batch_size, window_size, embed_dim)
    windows_reshaped = windows.permute(1, 0, 2, 3).reshape(n_windows * batch_size, window_size, embed_dim)

    # Compute ED for all windows across all batches
    ed_scores_flat = compute_effective_dimensionality(windows_reshaped) # Shape: (n_windows * batch_size,)

    # Reshape back to (n_windows, batch_size) then permute to (batch_size, n_windows)
    ed_scores = ed_scores_flat.view(n_windows, batch_size).permute(1, 0) # Shape: (batch_size, n_windows)

    return ed_scores


@torch.jit.script
def compute_intrinsic_dimensionality(data: torch.Tensor, discard_fraction: float = 0.1, eps: float = 1e-10) -> torch.Tensor:
    """Estimates the intrinsic dimensionality using the TwoNN algorithm with regression and outlier handling.

    This implementation follows the scikit-dimension approach:
    1. Compute mu = r2/r1 ratios for all points
    2. Discard fraction of largest ratios (outlier removal)
    3. Create empirical cumulative distribution 
    4. Linear regression of log(mu) vs -log(1-F_emp) to get dimension

    Args:
        data: Tensor of shape (batch_size, n_samples, embed_dim).
              n_samples should be > 5 for reliable estimation.
        discard_fraction: Fraction of largest mu ratios to discard (default 0.1)
        eps: Small epsilon value for numerical stability.

    Returns:
        Tensor of shape (batch_size,) containing intrinsic dimensionality estimates.
        Returns NaN for batches where computation fails.
    """
    batch_size, n_samples, embed_dim = data.shape
    device = data.device
    dtype = torch.float32

    # Ensure enough samples for reliable TwoNN estimation
    if n_samples <= 5:
        return torch.full((batch_size,), float('nan'), device=device, dtype=dtype)

    data = data.to(dtype)

    # Compute pairwise distances
    distances = torch.cdist(data, data, p=2.0) # Shape: (batch_size, n_samples, n_samples)

    # Set diagonal to infinity to exclude self-distance
    distances.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

    # Find the 2 nearest neighbors (r1, r2)
    k_nearest_distances, _ = torch.topk(distances, k=2, dim=-1, largest=False, sorted=True)
    r1 = k_nearest_distances[..., 0] # Shape: (batch_size, n_samples)
    r2 = k_nearest_distances[..., 1] # Shape: (batch_size, n_samples)

    # Compute mu = r2/r1 ratios, avoiding division by zero
    valid_mask = (r1 > eps) & (r2 > eps)
    mu = torch.where(valid_mask, r2 / r1, torch.tensor(float('inf'), device=device, dtype=dtype))

    # Process each batch item separately for sorting and regression
    id_estimates = torch.full((batch_size,), float('nan'), device=device, dtype=dtype)

    for b in range(batch_size):
        mu_batch = mu[b] # Shape: (n_samples,)
        valid_mu_mask = torch.isfinite(mu_batch)
        
        if torch.sum(valid_mu_mask.float()) < 5:  # Need minimum points
            continue
            
        mu_valid = mu_batch[valid_mu_mask]
        
        # Sort mu values and discard largest fraction (outlier removal)
        mu_sorted, _ = torch.sort(mu_valid)
        n_keep = int(len(mu_sorted) * (1.0 - discard_fraction))
        n_keep = max(n_keep, 5)  # Ensure minimum for regression
        
        if n_keep < 5:
            continue
            
        mu_kept = mu_sorted[:n_keep]
        
        # Create empirical cumulative distribution
        # F_emp[i] = (i+1) / n_samples where i is 0-indexed
        indices = torch.arange(1, n_keep + 1, dtype=dtype, device=device)
        f_emp = indices / float(n_samples)
        
        # Prepare regression variables: log(mu) vs -log(1-F_emp)
        log_mu = torch.log(mu_kept + eps)
        log_1_minus_f = torch.log(1.0 - f_emp + eps)
        y = -log_1_minus_f
        x = log_mu
        
        # Check for sufficient variance in x for regression
        if torch.var(x) < eps or torch.var(y) < eps:
            continue
            
        # Linear regression: y = slope * x (no intercept)
        # Using normal equations: slope = (x^T * y) / (x^T * x)
        numerator = torch.sum(x * y)
        denominator = torch.sum(x * x)
        
        if torch.abs(denominator) < eps:
            continue
            
        slope = numerator / denominator
        
        # The slope is the intrinsic dimension estimate
        if torch.isfinite(slope) and slope > 0.0 and slope < 1000.0:  # Sanity check
            id_estimates[b] = slope

    return id_estimates


@torch.jit.script
def compute_fixed_window_id(activations_batch: torch.Tensor, n_windows: int, discard_fraction: float = 0.1) -> torch.Tensor:
    """Computes intrinsic dimensionality over N fixed, non-overlapping windows using TwoNN.

    Args:
        activations_batch: Tensor of shape (batch_size, seq_len, embed_dim).
        n_windows: The number of fixed windows to divide the sequence into.
        discard_fraction: Fraction of largest distances to discard for outlier removal.

    Returns:
        Tensor of shape (batch_size, n_windows) containing ID estimates for each window.
    """
    batch_size, seq_len, embed_dim = activations_batch.shape
    device = activations_batch.device
    dtype = torch.float32

    # Ensure n_windows is valid
    if n_windows <= 0:
        # raise ValueError("n_windows must be positive") # JIT doesn't support raising exceptions easily
        # Return NaN tensor instead
        return torch.full((batch_size, n_windows), float('nan'), device=device, dtype=dtype)


    # Handle cases where seq_len is too small for windowing or ID calculation
    min_samples_needed = 6 # Need more samples for improved algorithm
    if seq_len < n_windows or seq_len < min_samples_needed:
        # Cannot reliably compute ID. Return NaNs.
        # print(f"Warning: seq_len ({seq_len}) is too small for {n_windows} windows or ID calculation. Returning NaNs.") # JIT doesn't support print
        return torch.full((batch_size, n_windows), float('nan'), device=device, dtype=dtype)

    window_size = seq_len // n_windows
    if window_size < min_samples_needed:
         # Window size too small for ID calculation
         # print(f"Warning: Window size ({window_size}) is less than minimum required ({min_samples_needed}) for ID. Returning NaNs.") # JIT doesn't support print
         return torch.full((batch_size, n_windows), float('nan'), device=device, dtype=dtype)

    # Truncate the sequence to be divisible by n_windows
    truncated_len = n_windows * window_size
    truncated_activations = activations_batch[:, :truncated_len, :]

    # Reshape into windows: (batch_size, n_windows, window_size, embed_dim)
    windows = truncated_activations.reshape(batch_size, n_windows, window_size, embed_dim)

    # Reshape for batch ID computation: (batch_size * n_windows, window_size, embed_dim)
    # Permute: (n_windows, batch_size, window_size, embed_dim)
    # Reshape: (n_windows * batch_size, window_size, embed_dim)
    windows_reshaped = windows.permute(1, 0, 2, 3).reshape(n_windows * batch_size, window_size, embed_dim)

    # Compute ID for all windows across all batches using improved algorithm
    id_scores_flat = compute_intrinsic_dimensionality(windows_reshaped, discard_fraction) # Shape: (n_windows * batch_size,)

    # Reshape back to (n_windows, batch_size) then permute to (batch_size, n_windows)
    id_scores = id_scores_flat.view(n_windows, batch_size).permute(1, 0) # Shape: (batch_size, n_windows)

    return id_scores


def compute_accuracy_by_example(
    gt_ids: torch.Tensor, 
    pred_ids: torch.Tensor, 
    token_labels: np.ndarray,
    accuracy_mode: str = 'all'
) -> torch.Tensor:
    """
    Computes token-level accuracy for each example index across a batch.

    Examples are identified by labels following the pattern 'ex<N>_answer', 
    where <N> is the example index (1-based).

    Args:
        gt_ids: Ground truth token IDs. Shape: (batch_size, seq_len).
        pred_ids: Predicted token IDs. Shape: (batch_size, seq_len).
        token_labels: NumPy array of string labels for each token. 
                      Shape: (batch_size, seq_len).
        accuracy_mode: 'all' requires all answer tokens to be correct.
                       'first_token' requires only the first answer token to be correct.
                       'token_wise' computes the fraction of correct tokens.

    Returns:
        Tensor of shape (batch_size, max_example_idx) containing the accuracy 
        for each example index for each item in the batch. Returns NaN if an 
        example index is not present for a batch item or has no answer tokens.
    """
    batch_size, seq_len = gt_ids.shape
    device = gt_ids.device
    dtype = torch.float32

    # Find the maximum integer mentioned in any label across the entire batch
    all_labels_str = ' '.join(map(str, token_labels.flatten()))
    all_digits = re.findall(r'\d+', all_labels_str)
    all_ints = [int(d) for d in all_digits]
    max_example_idx = max(all_ints) if all_ints else 0

    if max_example_idx == 0:
        # No integers found in any labels
        # Let's return a tensor with shape (batch_size, 0)
        return torch.empty((batch_size, 0), device=device, dtype=dtype)

    # Initialize output tensor with NaNs
    accuracies = torch.full((batch_size, max_example_idx), float('nan'), device=device, dtype=dtype)

    # Iterate over batch items and example indices
    for b in range(batch_size):
        labels_for_item = token_labels[b]
        for ex_idx in range(1, max_example_idx + 1):
            # Create boolean mask for the current example's answer tokens
            # Ensure comparison is with string representation of labels
            target_label = f'ex{ex_idx}_answer'
            mask = np.array([str(label) == target_label for label in labels_for_item])

            if np.any(mask):
                # Select ground truth and predictions using the mask
                gt = gt_ids[b][mask]
                pred = pred_ids[b][mask]

                # Calculate accuracy if there are tokens for this example
                if gt.numel() > 0:
                    if accuracy_mode == 'all':
                        is_correct = torch.all(gt == pred).float()
                    elif accuracy_mode == 'first_token':
                        is_correct = (gt[0] == pred[0]).float()
                    elif accuracy_mode == 'token_wise':
                        is_correct = (gt == pred).float().mean()
                    else:
                        raise ValueError(f"Invalid accuracy_mode: {accuracy_mode}")
                    
                    accuracies[b, ex_idx - 1] = is_correct

                # else: accuracy remains NaN (already initialized)
            # else: accuracy remains NaN if no tokens found for this ex_idx in this batch item

    return accuracies

@torch.jit.script 
def matrix_entropy(matrix: torch.Tensor, alpha: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """
    Computes the matrix-based Rényi entropy of a given matrix.
    The input matrix Z is of shape (..., N, D), where ... are batch dimensions.

    From "A unified measure of semantic information" by Skeen et al. (2023), 
    the formula is: S_alpha(Z) = (1/(1-alpha)) * log(sum_i (p_i^alpha)),
    where p_i are the normalized eigenvalues of the Gram matrix K = Z @ Z.T.

    For alpha = 1, it computes the Shannon entropy: -sum_i (p_i * log(p_i)).

    Args:
        matrix: Input tensor Z of shape (..., N, D).
        alpha: Order of the Rényi entropy. Defaults to 1.0 (Shannon entropy).
        eps: Small value for numerical stability. Defaults to 1e-10.

    Returns:
        Tensor containing the matrix entropy for each item in the batch.
    """
    # Ensure float32 for stability
    matrix = matrix.to(torch.float32)

    # Compute Gram matrix K = Z @ Z.T, shape: (..., N, N)
    K = torch.matmul(matrix, matrix.transpose(-2, -1))

    # Compute eigenvalues of K. `eigvalsh` is for symmetric/Hermitian matrices.
    # K is positive semi-definite, so eigenvalues are non-negative.
    eigenvalues = torch.linalg.eigvalsh(K) # Shape: (..., N)

    # Due to numerical precision, some eigenvalues might be slightly negative.
    eigenvalues = torch.clamp(eigenvalues, min=0)

    trace_K = torch.sum(eigenvalues, dim=-1) # Shape: (...)

    # Avoid division by zero if trace is zero.
    trace_K = trace_K + eps

    # Normalize eigenvalues to get probabilities p_i
    p = eigenvalues / trace_K.unsqueeze(-1) # Shape: (..., N)

    # Now compute entropy based on alpha
    if abs(alpha - 1.0) < eps:
        # Shannon entropy for alpha -> 1
        # Use torch.xlogy to handle p_i = 0 cases correctly (0 * log(0) = 0).
        # We need -sum(p * log(p)).
        entropy = -torch.sum(torch.xlogy(p, p), dim=-1)
    else:
        # Rényi entropy for alpha != 1
        p_alpha = torch.pow(p, alpha)
        sum_p_alpha = torch.sum(p_alpha, dim=-1)
        # Add eps to log for stability
        log_sum = torch.log(sum_p_alpha)
        entropy = log_sum / (1.0 - alpha)

    return entropy