# download_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen-VL-Chat"
SAVE_DIR = "./qwen-vl-chat-local"

print(f"Downloading tokenizer to {SAVE_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Downloading model to {SAVE_DIR}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.save_pretrained(SAVE_DIR)

print("Download complete.")