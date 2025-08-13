import os
from pathlib import Path
from typing import Dict, List
import hydra
from hydra.utils import instantiate
from functools import partial
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pyrootutils
from metrics import (
    compute_effective_dimensionality,
    compute_fixed_window_ed,
    compute_intrinsic_dimensionality,
    compute_fixed_window_id,
    compute_accuracy_by_example,
    matrix_entropy
)

def identity_metric(activations_batch: torch.Tensor) -> torch.Tensor:
    """A dummy 'metric' that just returns the activations to be saved."""
    return activations_batch

# project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

def save_metric_results(metric_name: str, metric_data: Dict[str, List[torch.Tensor]], base_save_dir: Path):
    """Saves the collected results for a specific metric."""
    metric_save_dir = base_save_dir / metric_name
    os.makedirs(metric_save_dir, exist_ok=True)
    print(f"Saving {metric_name} results to {metric_save_dir}")

    # Concatenate and save the scores for each layer.
    for layer_name, scores_list in metric_data.items():
        # Pad variable-length tensors along dim=1 to the max length
        max_len = max(t.shape[1] if t.ndim > 1 else 1 for t in scores_list)
        padded_list = []
        for t in scores_list:
            if t.ndim == 1:
                # Expand to [B, 1] for consistency
                t2 = t.unsqueeze(1)
            else:
                t2 = t
            pad_width = max_len - t2.shape[1]
            if pad_width > 0:
                t2 = F.pad(t2, (0, 0) if t2.ndim == 3 else (0, 0), mode='constant', value=0)
                # If 3D [B, W, D], pad width dim (dim=1)
                if t2.ndim == 3:
                    t2 = F.pad(t2, (0, 0, 0, pad_width), mode='constant', value=0)
                elif t2.ndim == 2:
                    # 2D [B, W]
                    t2 = F.pad(t2, (0, pad_width), mode='constant', value=0)
            padded_list.append(t2)
        final_tensor = torch.cat(padded_list, dim=0)
        filename = f"{layer_name}.pt"
        save_path = metric_save_dir / filename
        torch.save(final_tensor, save_path)

@hydra.main(config_path='config', config_name='inference', version_base=None)
def main(cfg: DictConfig):

    print(f'cfg.model.output_dir: {cfg.model.output_dir}')
    print(f'cfg.dataset.task_name: {cfg.dataset.task_name}')
    print(f'cfg.model.model_name: {cfg.model.model_name}')
    print(f'cfg.model.ablate_heads: {cfg.model.ablate_heads}')
    print(f'cfg.dataset.shuffle: {cfg.dataset.shuffle}')
    print(f'cfg.dataset.batch_size: {cfg.dataset.batch_size}')

    # Setup save directory
    save_dir = os.path.join(
        cfg.model.output_dir, 
        cfg.dataset.task_name,
        cfg.model.model_name,
        f'shuffle={cfg.dataset.shuffle}' + f'_ablate={cfg.model.ablate_heads}'
    )
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # # Check if inference has already been run
    # if (save_dir / 'outputs.txt').exists(): # and (save_dir / 'text.txt').exists():
    #     print(f'Inference outputs already exist in {save_dir}. Skipping inference.')
    #     return

    # Load the model
    model = instantiate(cfg.model)

    # Create dataloader
    dataset = instantiate(cfg.dataset)
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        # batch is a list of dicts
        keys = batch[0].keys()
        out = {}
        for key in keys:
            if isinstance(batch[0][key], torch.Tensor):
                pad_val = dataset.tokenizer.pad_token_id if key == 'input_ids' else 0
                out[key] = pad_sequence([item[key] for item in batch], batch_first=True, padding_value=pad_val)
            else:
                out[key] = [item[key] for item in batch]
        return out

    dataloader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True, collate_fn=collate_fn)

    # Define metrics to compute
    metrics_to_compute = {}

    # Add full sequence ED if specified
    if OmegaConf.select(cfg, "metrics.full_sequence_ed.enabled", default=True):
         metrics_to_compute["full_sequence_ed"] = compute_effective_dimensionality

    # Add fixed window ED if specified
    if OmegaConf.select(cfg, "metrics.fixed_window_ed.enabled", default=True):
        n_windows_ed = OmegaConf.select(cfg, "metrics.fixed_window_ed.n_windows", default=10) # Default to 10 windows
        metrics_to_compute["fixed_window_ed"] = partial(compute_fixed_window_ed, n_windows=n_windows_ed)

    # Add full sequence ID if specified
    if OmegaConf.select(cfg, "metrics.full_sequence_id.enabled", default=True):
        metrics_to_compute["full_sequence_id"] = compute_intrinsic_dimensionality

    # Add fixed window ID if specified
    if OmegaConf.select(cfg, "metrics.fixed_window_id.enabled", default=True):
        n_windows_id = OmegaConf.select(cfg, "metrics.fixed_window_id.n_windows", default=10) # Default to 10 windows
        metrics_to_compute["fixed_window_id"] = partial(compute_fixed_window_id, n_windows=n_windows_id)

    # Add matrix entropy if specified
    if OmegaConf.select(cfg, "metrics.matrix_entropy.enabled", default=False):
        alpha = OmegaConf.select(cfg, "metrics.matrix_entropy.alpha", default=1.0)
        eps = OmegaConf.select(cfg, "metrics.matrix_entropy.eps", default=1e-10)
        metrics_to_compute["matrix_entropy"] = partial(matrix_entropy, alpha=alpha, eps=eps)

    # Add a metric to save activations if specified
    if OmegaConf.select(cfg, "metrics.save_activations.enabled", default=True):
        metrics_to_compute["saved_activations"] = identity_metric

    # Run inference and collect metrics
    all_outputs = []
    all_predicted_tokens = []
    all_metric_results = {name: {} for name in metrics_to_compute}

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference Batches")):
        # Get model outputs and metric scores
        decoded_outputs, predicted_tokens, batch_metric_results = model.forward_with_metrics(batch, metrics_to_compute)

        # Collect outputs and predictions
        all_outputs.extend(decoded_outputs)
        all_predicted_tokens.append(predicted_tokens.cpu()) # Ensure tokens are on CPU

        # Collect metric scores if they were computed
        if batch_metric_results:
            for metric_name, layer_results in batch_metric_results.items():
                for layer_name, scores in layer_results.items():
                    if layer_name not in all_metric_results[metric_name]:
                        all_metric_results[metric_name][layer_name] = []
                    all_metric_results[metric_name][layer_name].append(scores.cpu()) # Ensure scores are on CPU

    # --- Save Results ---

    # Save metrics
    print(f"\nSaving metric results to base directory: {save_dir}")
    for metric_name, metric_data in all_metric_results.items():
        if metric_name == "saved_activations":
            # Special handling for saved activations
            layers_to_save = OmegaConf.select(cfg, "metrics.save_activations.layers", default=['layer-15'])
            if layers_to_save:
                # Filter to save only the specified layers
                filtered_data = {layer: data for layer, data in metric_data.items() if layer in layers_to_save}
                if not filtered_data:
                    print(f"Warning: Specified layers {layers_to_save} not found in model's activation results.")
                    continue
                save_metric_results("saved_activations", filtered_data, save_dir)
            else:
                # If no specific layers are requested, don't save any by default to avoid large files.
                print("`metrics.save_activations.enabled` is true, but no layers specified in `metrics.save_activations.layers`. Skipping activation saving.")
        else:
            save_metric_results(metric_name, metric_data, save_dir)

    # Save standard outputs
    all_outputs = [''.join(char for char in output if ord(char) < 128) for output in all_outputs] # remove padding characters
    all_outputs_cleaned = [output.replace('\n', '\t\t') for output in all_outputs]
    (save_dir / 'outputs.txt').write_text('\n'.join(all_outputs_cleaned))
    
    # Concatenate predicted tokens with right-padding to the max sequence length
    max_pred_len = max(t.shape[1] for t in all_predicted_tokens)
    pad_val = dataset.tokenizer.pad_token_id if hasattr(dataset, 'tokenizer') and dataset.tokenizer.pad_token_id is not None else 0
    padded_pred_list = [F.pad(t, (0, max_pred_len - t.shape[1]), mode='constant', value=pad_val) if t.shape[1] < max_pred_len else t for t in all_predicted_tokens]
    predicted_token_ids = torch.cat(padded_pred_list, dim=0)
    torch.save(predicted_token_ids, save_dir / 'predicted_token_ids.pt')

    # --- Compute and Save Accuracy ---
    # Retrieve necessary data from dataset cache
    from torch.nn.utils.rnn import pad_sequence
    tokenizer = dataset.tokenizer
    input_ids_list = dataset.cache.get('input_ids')
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(predicted_token_ids.device)
    token_labels_list = dataset.cache.get('token_labels')
    max_len = input_ids.shape[1]
    padded_token_labels = [labels + ["PAD"] * (max_len - len(labels)) for labels in token_labels_list]
    token_labels = np.array(padded_token_labels)

    # Align GT, predictions, and labels for accuracy calculation
    # Model predicts next token, so shift GT and labels
    gt_ids_for_acc = input_ids[:, 1:]
    pred_ids_for_acc = predicted_token_ids[:, :-1]
    labels_for_acc = token_labels[:, 1:] # Align labels with the GT tokens being predicted

    # Ensure shapes match after slicing
    print("\nComputing accuracy by example...")

    # Get accuracy mode from config
    accuracy_mode = OmegaConf.select(cfg, "metrics.accuracy.mode", default='all')

    # Normalize token label patterns from 'answer_<idx>' to 'ex<idx+1>_answer'
    import re
    labels_flat = labels_for_acc.reshape(-1)
    labels_converted = np.array([
        (lambda m: f"ex{int(m.group(1)) + 1}_answer")(m) if (m := re.match(r"answer_(\d+)", str(lbl))) else str(lbl)
        for lbl in labels_flat
    ], dtype=object)
    labels_for_acc_norm = labels_converted.reshape(labels_for_acc.shape)

    # Align lengths across gt, preds, and labels to avoid shape mismatches
    min_len = min(gt_ids_for_acc.shape[1], pred_ids_for_acc.shape[1], labels_for_acc_norm.shape[1])
    if gt_ids_for_acc.shape[1] != min_len or pred_ids_for_acc.shape[1] != min_len or labels_for_acc_norm.shape[1] != min_len:
        gt_ids_for_acc = gt_ids_for_acc[:, :min_len]
        pred_ids_for_acc = pred_ids_for_acc[:, :min_len]
        labels_for_acc_norm = labels_for_acc_norm[:, :min_len]

    accuracy_scores = compute_accuracy_by_example(
        gt_ids=gt_ids_for_acc,
        pred_ids=pred_ids_for_acc,
        token_labels=labels_for_acc_norm,
        accuracy_mode=accuracy_mode
    )
    print(f'accuracy_scores: {accuracy_scores.mean(0)}')
    torch.save(accuracy_scores, save_dir / 'accuracy_by_example.pt')
    print(f"Accuracy scores saved to {save_dir / 'accuracy_by_example.pt'}")

    # Save dataset-specific data (like input text, token labels)
    dataset.save_run(save_dir)

if __name__ == '__main__':
    main()