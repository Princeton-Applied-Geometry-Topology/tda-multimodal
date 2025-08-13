from functools import partial
from pathlib import Path
from transformer_lens import HookedTransformer
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from model.model import LLM
import traceback
import os


class Pythia(LLM):
    '''Pythia model implementation using TransformerLens'''

    def __init__(
        self,
        base_model_name: str = 'pythia-6.9b-deduped',
        model_name: str = None,
        checkpoint_number: str = '143000',
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        # Construct model identifier and name before calling super
        self.base_model_name = base_model_name
        self.checkpoint_number = checkpoint_number
        self.revision = f'step{self.checkpoint_number}'
        self.model_identifier = f'EleutherAI/{self.base_model_name}'
        model_name = model_name

        super().__init__(weights_path=self.model_identifier, model_name=model_name, **kwargs)

        self.cache_dir = cache_dir

        # Load model using TransformerLens
        print(f"Loading {self.model_identifier} revision {self.revision}...")
        self.model = HookedTransformer.from_pretrained(
            model_name=self.model_identifier,
            revision=self.revision,
            device=self.device,
            dtype=torch.float16 if self.torch_dtype == 'auto' else getattr(torch, self.torch_dtype),
            fold_ln=False,  # Keep LayerNorm separate for better analysis
            center_unembed=False,
            center_writing_weights=False,
            cache_dir=self.cache_dir
        )
        self.model.eval()
        # Map transformer layers to standard names for accessibility
        self.layer_map = self._create_layer_map()

        # Prepare for head ablation if requested
        self.ablation_coords: List[Tuple[int, int]] = []
        if self.ablate_heads:
            self._load_and_prepare_ablation_coords()

    def _create_layer_map(self) -> Dict[str, str]:
        """Create mapping from friendly names to TransformerLens internal names"""
        layer_map = {}
        num_layers = self.model.cfg.n_layers

        for layer_idx in range(num_layers):
            layer_map[f'layer-{layer_idx}'] = f'blocks.{layer_idx}'

        return layer_map

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor]:
        """Run forward pass and return predictions"""
        with torch.no_grad():
            # Get input tensors
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None) # TransformerLens handles None mask

            # Run forward pass with TransformerLens
            logits = self.model(
                input_ids,
                attention_mask=attention_mask # Pass mask if available
            )

            # Get predictions (next token)
            predictions = torch.argmax(logits, dim=-1)

            # TransformerLens returns logits that predict the next token,
            # so predictions already aligned with input sequence
            predicted_tokens = predictions

            # Decode predictions to text
            decoded_outputs = self.tokenizer.batch_decode(
                predicted_tokens,
                skip_special_tokens=True
            )

        return decoded_outputs, predicted_tokens

    def get_representations(self, batch: Dict[str, torch.Tensor], layer_name: str) -> torch.Tensor:
        """Get model representations for a specific layer using TransformerLens"""
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)

            # Use TransformerLens hooks to get activations
            transformer_lens_name = self.layer_map.get(layer_name, layer_name)

            _, cache = self.model.run_with_cache(
                input_ids,
                attention_mask=attention_mask,
                names_filter=transformer_lens_name
            )

            # Extract the cached activations
            # Ensure the output tensor is on the same device as the input
            return cache[transformer_lens_name].to(input_ids.device)

    def _load_and_prepare_ablation_coords(self):
        """Loads AIE scores and identifies the top heads to ablate."""
        # Use self.model_name which includes base and step
        aie_scores_path = Path(f'aie_scores/{self.base_model_name}.pt')
        if not aie_scores_path.exists():
            raise FileNotFoundError(f"AIE scores file not found at {aie_scores_path}. Cannot perform ablation.")

        # Load AIE scores (expected shape: [n_layers, n_heads])
        aie_scores = torch.load(aie_scores_path, map_location='cpu')
        n_layers, n_heads = aie_scores.shape

        # Ensure n_layers matches model config
        if n_layers != self.model.cfg.n_layers:
             print(f"Warning: AIE scores have {n_layers} layers, but model has {self.model.cfg.n_layers}. Using model's layer count for indexing.")
             # If mismatch, maybe trim or error? For now, proceed cautiously.

        # Find the indices of the top n_ablate_heads globally
        flat_scores = aie_scores.flatten()
        # Ensure k is not larger than the number of scores
        k = min(self.n_ablate_heads, flat_scores.numel())
        if k < self.n_ablate_heads:
            print(f"Warning: Requested ablation for {self.n_ablate_heads} heads, but only {k} scores available. Ablating top {k}.")

        top_k_indices_flat = torch.topk(flat_scores, k=k, largest=True).indices
        self.ablation_coords = [(idx.item() // n_heads, idx.item() % n_heads) for idx in top_k_indices_flat]
        print(f"Prepared ablation for {len(self.ablation_coords)} heads: {self.ablation_coords}")


    def _collate_metric_results(self, raw_results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Collates raw metric results keyed by hook names to friendly layer names."""
        final_results = {}
        # Assumes layer_map is populated correctly
        for hook_name, scores in raw_results.items():
            parts = hook_name.split('.')
            if len(parts) > 1 and parts[0] == 'blocks' and parts[1].isdigit():
                layer_idx_str = parts[1]
                layer_key = f'layer-{layer_idx_str}' # Construct the friendly name key
                final_results[layer_key] = scores
        return final_results

    def forward_with_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        metrics_to_compute: Dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ) -> Tuple[List[str], torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Run forward pass, compute specified metrics for each layer, and return predictions.

        Args:
            batch: Dictionary containing input_ids and optionally attention_mask.
            metrics_to_compute: Dictionary where keys are metric names (str) and
                                values are callable functions. Each function must accept
                                a single argument (activation tensor: [batch, seq, dim])
                                and return a tensor of metric scores (e.g., [batch] or [batch, n_windows]).
                                Use functools.partial to pass additional arguments to metric functions.

        Returns:
            Tuple containing:
                - decoded_outputs (List[str]): Decoded predicted tokens.
                - predicted_tokens (torch.Tensor): Predicted token IDs.
                - all_metric_results (Dict[str, Dict[str, torch.Tensor]]):
                    Nested dictionary containing results. Outer keys are metric names,
                    inner keys are friendly layer names (e.g., 'layer-0'), values are
                    metric score tensors for that layer, moved to CPU.
        """
        with torch.no_grad():
            # Get input tensors
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)

            # --- Hook Setup ---
            # Stores raw results before collation: {metric_name: {hook_name: results_tensor}}
            all_metric_results_raw = {name: {} for name in metrics_to_compute}
            all_hooks = []

            # --- Ablation Hook Setup (if enabled) ---
            if self.ablate_heads and self.ablation_coords:
                # Create a set of (layer, head) tuples for faster lookup inside the hook
                ablation_coords_set = set(self.ablation_coords)

                def ablation_hook(
                    activation: torch.Tensor, # Shape: [batch, seq_pos, n_heads, d_head]
                    hook: Any
                ):
                    layer_idx = hook.layer()
                    heads_to_ablate_in_layer = [
                        head_idx for l_idx, head_idx in ablation_coords_set if l_idx == layer_idx
                    ]
                    if heads_to_ablate_in_layer:
                        # Zero out the specified heads for this layer
                        # Note: This modifies the activation tensor in-place
                        activation[:, :, heads_to_ablate_in_layer, :] = 0.0

                # Add ablation hook for the 'z' activations (output of attention heads)
                for layer_idx in range(self.model.cfg.n_layers):
                     # Check if any heads in this layer need ablation before adding the hook
                    if any(l_idx == layer_idx for l_idx, _ in ablation_coords_set):
                        hook_point = f'blocks.{layer_idx}.attn.hook_z'
                        all_hooks.append((hook_point, ablation_hook))

            # --- Metric Hook Setup ---
            # Create hooks for each requested metric
            for metric_name, metric_fn in metrics_to_compute.items():

                # Define the hook function within this loop to capture metric_name and metric_fn
                def specific_metric_hook(
                    activation: torch.Tensor,
                    hook: Any,
                    _metric_name=metric_name, # Capture loop variables
                    _metric_fn=metric_fn
                ):
                    # Compute the metric using the provided function and cache results
                    try:
                        print(f"[Metric Hook] {hook.name}: activation shape = {activation.shape}, metric = {_metric_name}")
                        metric_result = _metric_fn(activation)
                        print(f"[Metric Hook] {hook.name}: metric_result shape = {metric_result.shape}")
                        # Store results keyed by the specific hook name (e.g., 'blocks.0.hook_resid_post')
                        all_metric_results_raw[_metric_name][hook.name] = metric_result.cpu()
                    except Exception as e:
                        print(f"Error computing metric '{_metric_name}' in hook {hook.name}: {e}")
                        traceback.print_exc()


                # Add this hook for all layers at the standard hook point (residual stream after MLP and Attn)
                for layer_idx in range(self.model.cfg.n_layers):
                    hook_point = f'blocks.{layer_idx}.hook_resid_post'
                    all_hooks.append((hook_point, specific_metric_hook))

            # --- Run Model ---
            logits = self.model.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                fwd_hooks=all_hooks
            )

            # --- Process Outputs ---
            # Get predictions (next token)
            predictions = torch.argmax(logits, dim=-1)
            predicted_tokens = predictions.cpu() # Move predictions to CPU

            # Decode predictions to text
            decoded_outputs = self.tokenizer.batch_decode(
                predicted_tokens,
                skip_special_tokens=True
            )

            # --- Collate Metric Results ---
            final_metric_results: Dict[str, Dict[str, torch.Tensor]] = {}
            for metric_name, raw_results in all_metric_results_raw.items():
                final_metric_results[metric_name] = self._collate_metric_results(raw_results)

        return decoded_outputs, predicted_tokens, final_metric_results
