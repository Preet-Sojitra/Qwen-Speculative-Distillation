import time
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def main():
    # Model configuration
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print("Loading openai_humaneval dataset...")
    # HumanEval validation set has 164 coding problems
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    # We'll evaluate on a subset early on just for quick testing
    NUM_SAMPLES = 20
    dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

    results = []
    total_tokens_generated = 0
    total_wall_clock_time = 0.0

    print("Starting autoregressive generation evaluation...")
    for item in tqdm(dataset):
        prompt = item["prompt"]
        task_id = item["task_id"]

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate (Standard Autoregressive)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # Limit new tokens for evaluation speed
                do_sample=False,     # Greedy decoding for baseline
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()

        wall_time = end_time - start_time
        # Count only newly generated tokens
        generated_tokens = outputs.shape[1] - input_length

        total_wall_clock_time += wall_time
        total_tokens_generated += generated_tokens

        # Decode generated text
        completion = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        results.append({
            "task_id": task_id,
            "completion": completion,
            "tokens_generated": generated_tokens,
            "wall_time": wall_time,
            "tps": generated_tokens / wall_time if wall_time > 0 else 0
        })

    # Summary Metrics
    overall_tps = total_tokens_generated / total_wall_clock_time if total_wall_clock_time > 0 else 0

    print("\n--- Evaluation Summary ---")
    print(f"Total Prompts Evaluated: {NUM_SAMPLES}")
    print(f"Total Tokens Generated: {total_tokens_generated}")
    print(f"Total Wall Clock Time: {total_wall_clock_time:.2f}s")
    print(f"Average Tokens Per Second (TPS): {overall_tps:.2f}")

    # Save results to JSON
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "autoregressive_baseline.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model_id,
            "metrics": {
                "tps": overall_tps,
                "total_time": total_wall_clock_time,
                "total_tokens": total_tokens_generated
            },
            "tasks": results
        }, f, indent=4)
        
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
