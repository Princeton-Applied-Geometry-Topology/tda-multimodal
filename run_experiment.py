#!/usr/bin/env python3
"""
General script for running TDA multimodal experiments.
Usage: python run_experiment.py <model_name> <task_name> [options]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Default configuration
DEFAULT_CONFIG = {
    "hf_home": "/scratch/gpfs/kg4280/.cache/huggingface",
    "base_path": "/scratch/gpfs/kg4280",
    "project_path": "/scratch/gpfs/kg4280/ptonagt/tda-multimodal",
    "default_batch_size": 8,
    "default_seq_len": 256,
    "default_n_seqs": 500,
    "default_max_tokens": 512,
}

# Model configurations
MODEL_CONFIGS = {
    "qwen2-vl-2b": {
        "model_path": "Qwen/Qwen2-VL-2B-Instruct",
        "snapshot": "895c3a49bc3fa70a340399125c650a463535e71c",
        "layers": list(range(32)),  # 0-31
    },
    "qwen-7b": {
        "model_path": "Qwen/Qwen-7B-Instruct",
        "snapshot": "default",
        "layers": list(range(32)),  # Adjust based on actual model
    },
    "llama-8b": {
        "model_path": "meta-llama/Llama-2-8b-chat-hf",
        "snapshot": "default",
        "layers": list(range(32)),  # Adjust based on actual model
    },
    "pythia-6.9b": {
        "model_path": "EleutherAI/pythia-6.9b",
        "snapshot": "default",
        "layers": list(range(28)),  # Adjust based on actual model
    }
}

# Task configurations
TASK_CONFIGS = {
    "conme-replace-obj": {
        "dataset_name": "conme",
        "task_name": "conme_replace_obj",
        "data_root": "/scratch/gpfs/nb0564/tda-multimodal/datasets/ConMe",
        "csv_path": "replace-obj.csv",
        "use_images": True,
    },
    "conme-replace-rel": {
        "dataset_name": "conme",
        "task_name": "conme_replace_rel",
        "data_root": "/scratch/gpfs/nb0564/tda-multimodal/datasets/ConMe",
        "csv_path": "replace-rel.csv",
        "use_images": True,
    },
    "conme-replace-att": {
        "dataset_name": "conme",
        "task_name": "conme_replace_att",
        "data_root": "/scratch/gpfs/nb0564/tda-multimodal/datasets/ConMe",
        "csv_path": "replace-att.csv",
        "use_images": True,
    },
    "conme-replace-obj-human": {
        "dataset_name": "conme",
        "task_name": "conme_replace_obj_human",
        "data_root": "/scratch/gpfs/nb0564/tda-multimodal/datasets/ConMe",
        "csv_path": "replace-obj-human.csv",
        "use_images": True,
    },
    "conme-replace-rel-human": {
        "dataset_name": "conme",
        "task_name": "conme_replace_rel_human",
        "data_root": "/scratch/gpfs/nb0564/tda-multimodal/datasets/ConMe",
        "csv_path": "replace-rel-human.csv",
        "use_images": True,
    },
    "conme-replace-att-human": {
        "dataset_name": "conme",
        "task_name": "conme_replace_att_human",
        "data_root": "/scratch/gpfs/nb0564/tda-multimodal/datasets/ConMe",
        "csv_path": "replace-att-human.csv",
        "use_images": True,
    }
}

def get_model_snapshot_path(model_name: str) -> str:
    """Get the full snapshot path for a model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = MODEL_CONFIGS[model_name]
    if model_config["snapshot"] == "default":
        # For models without specific snapshots, use the base path
        return model_config["model_path"]
    else:
        # For models with specific snapshots
        return f"{DEFAULT_CONFIG['hf_home']}/hub/models--{model_config['model_path'].replace('/', '--')}/snapshots/{model_config['snapshot']}"

def build_command(model_name: str, task_name: str, **kwargs) -> List[str]:
    """Build the command list for subprocess."""
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}")
    
    model_config = MODEL_CONFIGS[model_name]
    task_config = TASK_CONFIGS[task_name]
    
    # Get the full model path
    model_snapshot_path = get_model_snapshot_path(model_name)
    
    # Base command
    cmd = [
        "python", f"{DEFAULT_CONFIG['project_path']}/run.py",
        f"model={model_name}",
        f"dataset={task_name}",
    ]
    
    # Model-specific parameters
    if model_config["snapshot"] != "default":
        cmd.extend([
            # Set weights_path to local model directory
            f"model.weights_path={model_config['model_path']}",
            # Keep original model name for processor and tokenizer, but ensure local files only
            f"model.processor.pretrained_model_name_or_path={model_config['model_path']}",
            f"model.tokenizer.pretrained_model_name_or_path={model_config['model_path']}",
            "++model.processor.local_files_only=true",
            "++model.tokenizer.local_files_only=true",
        ])
    
    # Metrics configuration
    layers_str = ",".join([f"layer-{i}" for i in model_config["layers"]])
    cmd.extend([
        "+metrics.save_activations.enabled=true",
        f"+metrics.save_activations.layers=[{layers_str}]",
    ])
    
    # Hydra configuration
    cmd.extend([
        "hydra.run.dir=.",
    ])
    
    # Additional overrides from kwargs
    for key, value in kwargs.items():
        if value is not None:
            cmd.append(f"{key}={value}")
    
    return cmd

def run_experiment(model_name: str, task_name: str, **kwargs):
    """Run the experiment with the given parameters."""
    
    # Set environment variables
    env = os.environ.copy()
    env["HF_HOME"] = DEFAULT_CONFIG["hf_home"]
    
    # Build command
    cmd = build_command(model_name, task_name, **kwargs)
    
    print(f"Running experiment with:")
    print(f"  Model: {model_name}")
    print(f"  Task: {task_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  HF_HOME: {env['HF_HOME']}")
    print("-" * 80)
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            env=env,
            cwd=DEFAULT_CONFIG["project_path"],
            check=True,
            text=True
        )
        print("Experiment completed successfully!")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return e
    except Exception as e:
        print(f"Unexpected error: {e}")
        return e

def main():
    parser = argparse.ArgumentParser(
        description="Run TDA multimodal experiments with automatic configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py qwen2-vl-2b conme-replace-obj
  python run_experiment.py qwen-7b conme-replace-rel
  python run_experiment.py llama-8b conme-replace-att --batch_size 16
        """
    )
    
    parser.add_argument(
        "model_name",
        choices=list(MODEL_CONFIGS.keys()),
        help="Name of the model to use"
    )
    
    parser.add_argument(
        "task_name",
        choices=list(TASK_CONFIGS.keys()),
        help="Name of the task to run"
    )
    
    # Optional parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for the experiment"
    )
    
    parser.add_argument(
        "--seq_len",
        type=int,
        help="Sequence length"
    )
    
    parser.add_argument(
        "--n_seqs",
        type=int,
        help="Number of sequences"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum tokens"
    )
    
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Enable shuffling"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the command that would be run without executing it"
    )
    
    args = parser.parse_args()
    
    # Build kwargs for additional parameters
    kwargs = {}
    if args.batch_size:
        kwargs["dataset.batch_size"] = args.batch_size
    if args.seq_len:
        kwargs["dataset.seq_len"] = args.seq_len
    if args.n_seqs:
        kwargs["dataset.n_seqs"] = args.n_seqs
    if args.max_tokens:
        kwargs["dataset.max_tokens"] = args.max_tokens
    if args.shuffle:
        kwargs["dataset.shuffle"] = "true"
    
    if args.dry_run:
        # Just show the command
        cmd = build_command(args.model_name, args.task_name, **kwargs)
        print("Command that would be run:")
        print(" ".join(cmd))
        print(f"\nEnvironment: HF_HOME={DEFAULT_CONFIG['hf_home']}")
        return
    
    # Run the experiment
    run_experiment(args.model_name, args.task_name, **kwargs)

if __name__ == "__main__":
    main()
