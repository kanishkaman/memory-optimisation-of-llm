import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

accelerator = Accelerator()

# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.1-8B" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto",low_cpu_mem_usage=True)

model = accelerator.prepare(model)

# Function to extract KV cache from the model
def extract_kv_cache(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)  # Ensure KV caching is enabled

    # Extract KV cache properly from past_key_values
    kv_cache = list(outputs.past_key_values)  # Convert tuple to list for easier indexing

    return kv_cache, outputs

# SLERP function for merging KV caches
def slerp(v0, v1, t):
    v0 = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    dot = (v0 * v1).sum(dim=-1, keepdim=True)

    omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
    so = torch.sin(omega)
    
    return (torch.sin((1.0 - t) * omega) / so) * v0 + (torch.sin(t * omega) / so) * v1

# Token retention strategy
def token_retention(k1, k2, threshold=0.1):
    """Identify tokens that should not be merged due to large angular differences."""
    similarity = torch.nn.functional.cosine_similarity(k1, k2, dim=-1)
    mask = similarity < threshold  # Tokens with low similarity should be retained
    return mask

# Merge KV Cache with SLERP and Token Retention
def merge_kv_cache(kv_cache, start_layer=12, t=0.6, retention_threshold=0.1):
    merged_kv_cache = kv_cache[:start_layer]  # Keep shallow layers unchanged
    
    for i in range(start_layer, len(kv_cache)-1, 2):
        k1, v1 = kv_cache[i][:2]  # Extract first two elements (keys, values)
        k2, v2 = kv_cache[i+1][:2]

        # Identify tokens that should not be merged
        retention_mask = token_retention(k1, k2, retention_threshold)

        # Merge KV states using SLERP
        k_merged = slerp(k1, k2, t)
        v_merged = slerp(v1, v2, t)

        # Restore unmergeable tokens
        k_merged[retention_mask] = k1[retention_mask]
        v_merged[retention_mask] = v1[retention_mask]

        merged_kv_cache.append((k_merged, v_merged))

    return merged_kv_cache

# Run MiniCache on an input
input_text = "Once upon a time in a distant land,"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

kv_cache, outputs = extract_kv_cache(model, input_ids)
compressed_kv_cache = merge_kv_cache(kv_cache)

# Memory usage before and after compression
original_size = sum(k.numel() for k, v in kv_cache)
compressed_size = sum(k.numel() for k, v in compressed_kv_cache)

print(f"Original KV Cache Size: {original_size}")
print(f"Compressed KV Cache Size: {compressed_size}")
print(f"Compression Ratio: {original_size / compressed_size:.2f}x")
