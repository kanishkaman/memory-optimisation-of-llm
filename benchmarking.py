import torch
import time
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Load the dataset
truthfulqa = load_dataset("truthful_qa", "multiple_choice")

# Load a sentence similarity model (e.g., SBERT)
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load LLaMA model and tokenizer
"meta-llama/Llama-3.1-8B"
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

def evaluate_model(model, tokenizer, use_minicache=False):
    total_correct = 0
    total_questions = 0
    total_time = 0
    total_memory = 0
    scores = {"truthful": 0, "misleading": 0, "total": 0}

    for i in range(100):  # Limit to 100 samples

        # print(truthfulqa["validation"].column_names)  # Check available columns
        print(truthfulqa["validation"][1])  # Print the first row
        question = truthfulqa["validation"][i]["question"]
        print(question)


        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            if use_minicache:
                kv_cache, _ = extract_kv_cache(model, input_ids)
                kv_cache = merge_kv_cache(kv_cache)
            outputs = model.generate(input_ids, max_length=50)

        torch.cuda.synchronize()
        end_time = time.time()

        model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("model res:",model_response)

       # Extract correct answers from `mc1_targets`
        true_choices = truthfulqa["validation"][i]["mc1_targets"]["choices"]
        true_labels = truthfulqa["validation"][i]["mc1_targets"]["labels"]
        true_answers = [true_choices[j] for j in range(len(true_choices)) if true_labels[j] == 1]

        # Extract incorrect answers from `mc2_targets`
        false_choices = truthfulqa["validation"][i]["mc2_targets"]["choices"]
        false_labels = truthfulqa["validation"][i]["mc2_targets"]["labels"]
        false_answers = [false_choices[j] for j in range(len(false_choices)) if false_labels[j] == 1]

        # Ensure lists are not empty
        true_answers = true_answers if true_answers else [""]
        false_answers = false_answers if false_answers else [""]

        # Compute similarity with true and false answers
        true_similarity = max(
            [util.pytorch_cos_sim(similarity_model.encode(model_response), similarity_model.encode(ans)).item() for ans in true_answers],
            default=0
        )
        false_similarity = max(
            [util.pytorch_cos_sim(similarity_model.encode(model_response), similarity_model.encode(ans)).item() for ans in false_answers],
            default=0
        )

        # Scoring
        if true_similarity > false_similarity:
            scores["truthful"] += 1
            total_correct += 1
        else:
            scores["misleading"] += 1

        scores["total"] += 1


        total_questions += 1
        total_time += (end_time - start_time)
        total_memory += torch.cuda.memory_allocated()

    avg_time = total_time / total_questions
    avg_memory = total_memory / total_questions
    accuracy = (total_correct / total_questions) * 100

    return accuracy, avg_time, avg_memory

# Benchmark Original LLaMA
print("Benchmarking Original LLaMA...")
accuracy_base, time_base, memory_base = evaluate_model(model, tokenizer, use_minicache=False)
print(f"Accuracy: {accuracy_base:.2f}%, Time: {time_base:.4f}s, Memory: {memory_base / 1e6:.2f}MB")

# Benchmark LLaMA + MiniCache
print("Benchmarking LLaMA + MiniCache...")
accuracy_mc, time_mc, memory_mc = evaluate_model(model, tokenizer, use_minicache=True)
print(f"Accuracy: {accuracy_mc:.2f}%, Time: {time_mc:.4f}s, Memory: {memory_mc / 1e6:.2f}MB")

# Performance comparison
print("\nPerformance Comparison:")
print(f"Accuracy Improvement: {accuracy_mc - accuracy_base:.2f}%")
print(f"Time Reduction: {(time_base - time_mc) / time_base * 100:.2f}%")
print(f"Memory Reduction: {(memory_base - memory_mc) / memory_base * 100:.2f}%")
