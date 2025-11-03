# extract_activations.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import os

# --- Model and Tokenizer Setup ---
MODEL_NAME = "./qwen-vl-chat-local"
DATA_DIR = "data/physics_experiment_6x6"  # <-- FIX 1: Point to the new 6x6 data
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "all_activations.pt")

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    local_files_only=True  # <-- FIX 2: Stop internet connection
)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    device_map="auto",
    local_files_only=True  # <-- FIX 2: Stop internet connection
).eval()

# --- Hooking Setup ---
# This dictionary will store activations from our hooks
layer_activations = {} 

def get_hook(layer_name):
    """Returns a hook function that saves the output of a layer."""
    def hook(module, input, output):
        # output[0] contains the hidden states
        # We grab them, put them on CPU to save VRAM, and detach
        layer_activations[layer_name] = output[0].detach().cpu()
    return hook

# Register hooks on all decoder layers
# For Qwen, it's model.transformer.h.[i]
num_layers = model.config.num_hidden_layers
hook_handles = []
for i in range(num_layers):
    layer_name = f"layer_{i}"
    try:
        layer = model.transformer.h[i] # Qwen-specific module name
        handle = layer.register_forward_hook(get_hook(layer_name))
        hook_handles.append(handle)
    except AttributeError:
        print(f"Warning: Could not find layer 'model.transformer.h[{i}]'. Check model architecture.")

print(f"Registered {len(hook_handles)} hooks.")

# --- Activation Extraction ---
all_results = {}
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

for item in tqdm(metadata, desc="Extracting activations"):
    
    # --- FIX: Clear the global dict, don't re-assign it ---
    layer_activations.clear()
    
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
        continue
        
    # 2. Find the index of the last text token
    # We want the activation *before* the model predicts the next token
    # This corresponds to the *last token of the input prompt*
    
    # Qwen-VL tokenization is complex. A robust way to find the last *text* token:
    # Tokenize text-only, find its length (excluding special tokens)
    text_only_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    num_text_tokens = len(text_only_ids)

    # In the full 'inputs', the text tokens are at the end,
    # *before* the final special tokens (like <|endoftext|>)
    # The last_token_idx should be the index of the final text token.
    
    # Let's find the start of text tokens in the full input
    full_ids_list = inputs.input_ids[0].tolist()
    text_start_idx = -1
    for i in range(len(full_ids_list) - num_text_tokens + 1):
        if full_ids_list[i:i+num_text_tokens] == text_only_ids:
            text_start_idx = i
            break
            
    if text_start_idx == -1:
        print(f"Warning: Could not reliably find text tokens for {item['id']}. Using -2 as index.")
        last_token_idx = -2 # Fallback: 2nd to last token
    else:
        last_token_idx = text_start_idx + num_text_tokens - 1

    # 3. Run the model
    with torch.no_grad():
        model(**inputs) 

    # 4. Process and save the activations
    if not layer_activations:
        print(f"Warning: No activations captured for {item['id']}. Hooks might be misconfigured.")
        continue
        
    sample_activations = {}
    for layer_name, activation_tensor in layer_activations.items():
        # activation_tensor shape is [1, seq_len, hidden_dim]
        # We just want the vector for our last token
        if last_token_idx >= activation_tensor.shape[1]:
             print(f"Warning: last_token_idx {last_token_idx} out of bounds for layer {layer_name} shape {activation_tensor.shape}. Using -1.")
             last_token_idx = -1 # Fallback
             
        last_token_vec = activation_tensor[0, last_token_idx, :].clone()
        sample_activations[layer_name] = last_token_vec

    all_results[item["id"]] = {
        "metadata": item,
        "activations": sample_activations
    }

# --- Cleanup and Save ---
for handle in hook_handles:
    handle.remove()

print(f"\nExtracted activations for {len(all_results)} samples.")
if len(all_results) > 0:
    print(f"Saving to {OUTPUT_PATH}...")
    torch.save(all_results, OUTPUT_PATH)
    print("Done.")
else:
    print("No results to save.")