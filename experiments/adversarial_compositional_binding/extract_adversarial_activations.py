# extract_adversarial_activations.py
# Extracts activations for adversarial image-text pairs
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# --- Model and Tokenizer Setup ---
# Get project root (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_NAME = os.path.join(PROJECT_ROOT, "qwen-vl-chat-local")
DATA_DIR = os.path.join(PROJECT_ROOT, "data/physics_experiment_6x6")
METADATA_PATH = os.path.join(DATA_DIR, "adversarial_metadata.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "adversarial_activations.pt")

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    local_files_only=True
)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    device_map="auto",
    local_files_only=True
).eval()

# --- Hooking Setup ---
layer_activations = {} 

def get_hook(layer_name):
    """Returns a hook function that saves the output of a layer."""
    def hook(module, input, output):
        layer_activations[layer_name] = output[0].detach().cpu()
    return hook

# Register hooks on all decoder layers
num_layers = model.config.num_hidden_layers
hook_handles = []
for i in range(num_layers):
    layer_name = f"layer_{i}"
    try:
        layer = model.transformer.h[i]
        handle = layer.register_forward_hook(get_hook(layer_name))
        hook_handles.append(handle)
    except AttributeError:
        print(f"Warning: Could not find layer 'model.transformer.h[{i}]'. Check model architecture.")

print(f"Registered {len(hook_handles)} hooks.")

# --- Activation Extraction (with incremental saving) ---
SAVE_INTERVAL = 50  # Save every N samples to avoid memory issues
TEMP_OUTPUT_PATH = OUTPUT_PATH + ".tmp"

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

print(f"Processing {len(metadata)} adversarial samples...")
print(f"Will save incrementally every {SAVE_INTERVAL} samples to avoid memory issues.")

# Load existing results if resuming
all_results = {}
if os.path.exists(TEMP_OUTPUT_PATH):
    print(f"Found existing checkpoint, resuming...")
    try:
        all_results = torch.load(TEMP_OUTPUT_PATH)
        print(f"Loaded {len(all_results)} existing samples.")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}. Starting fresh.")

processed_count = len(all_results)
skipped_count = 0

for idx, item in enumerate(tqdm(metadata, desc="Extracting activations", initial=processed_count)):
    # Skip if already processed
    if item["id"] in all_results:
        continue
    
    layer_activations.clear()
    
    # Clear GPU cache periodically
    if idx % 20 == 0:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 1. Prepare model input (image + text)
    prompt = item["prompt"]
    image_path = item["image_path"]
    
    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': prompt},
    ])
    
    try:
        inputs = tokenizer(query, return_tensors='pt').to(model.device)
    except Exception as e:
        print(f"Skipping item {item['id']} due to tokenization error: {e}")
        skipped_count += 1
        continue
        
    # 2. Find the index of the last text token
    text_only_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    num_text_tokens = len(text_only_ids)
    
    full_ids_list = inputs.input_ids[0].tolist()
    text_start_idx = -1
    for i in range(len(full_ids_list) - num_text_tokens + 1):
        if full_ids_list[i:i+num_text_tokens] == text_only_ids:
            text_start_idx = i
            break
            
    if text_start_idx == -1:
        print(f"Warning: Could not reliably find text tokens for {item['id']}. Using -2 as index.")
        last_token_idx = -2
    else:
        last_token_idx = text_start_idx + num_text_tokens - 1

    # 3. Run the model
    with torch.no_grad():
        model(**inputs) 

    # 4. Process and save the activations
    if not layer_activations:
        print(f"Warning: No activations captured for {item['id']}. Hooks might be misconfigured.")
        skipped_count += 1
        continue
        
    sample_activations = {}
    current_last_token_idx = last_token_idx
    for layer_name, activation_tensor in layer_activations.items():
        if current_last_token_idx >= activation_tensor.shape[1]:
             current_last_token_idx = -1
             
        last_token_vec = activation_tensor[0, current_last_token_idx, :].clone().cpu()
        sample_activations[layer_name] = last_token_vec

    all_results[item["id"]] = {
        "metadata": item,
        "activations": sample_activations
    }
    processed_count += 1
    
    # Incremental save to avoid memory issues
    if processed_count % SAVE_INTERVAL == 0:
        print(f"\nCheckpoint: Saving {processed_count} samples...")
        torch.save(all_results, TEMP_OUTPUT_PATH)
        # Clear memory
        del sample_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- Cleanup and Final Save ---
for handle in hook_handles:
    handle.remove()

print(f"\nExtracted activations for {processed_count} samples (skipped {skipped_count}).")
if len(all_results) > 0:
    print(f"Saving final results to {OUTPUT_PATH}...")
    torch.save(all_results, OUTPUT_PATH)
    # Remove temp file if final save succeeds
    if os.path.exists(TEMP_OUTPUT_PATH):
        os.remove(TEMP_OUTPUT_PATH)
    print("Done.")
else:
    print("No results to save.")

