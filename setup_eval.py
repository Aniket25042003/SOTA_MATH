import os
import random
from datasets import load_dataset
from langsmith import Client

# Initialize LangSmith client
# Requires LANGCHAIN_API_KEY environment variable to be set
client = Client()

DATASET_NAME = "Math_Agent_Eval_Top_500"

def get_gsm8k_samples(n=166):
    print("Loading GSM8K...")
    # GSM8K has 'main' config
    ds = load_dataset("gsm8k", "main", split="test")
    # Shuffle and pick n
    ds = ds.shuffle(seed=42).select(range(n))
    
    samples = []
    for item in ds:
        # Extract the final numeric answer which is after '#### '
        ans = item['answer'].split('#### ')[-1].strip()
        samples.append({
            "question": item['question'],
            "ground_truth": ans,
            "source": "gsm8k"
        })
    return samples

def get_math_samples(n=166):
    print("Loading MATH benchmark...")
    ds = load_dataset("hendrycks/competition_math", split="test")
    ds = ds.shuffle(seed=42).select(range(n))
    
    samples = []
    for item in ds:
        # The 'solution' contains the full reasoning and boxed answer
        # extracting boxed answer can be tricky, but we can store the whole solution for the Judge LLM
        samples.append({
            "question": item['problem'],
            "ground_truth": item['solution'],
            "source": "math",
            "level": item['level'],
            "type": item['type']
        })
    return samples

def get_aime_samples(n=166):
    print("Loading AIME/AMC (via NuminaMath-CoT)...")
    # NuminaMath-CoT contains AIME/AMC data in the train split under different sources
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    
    # Filter for AIME
    aime_ds = ds.filter(lambda x: x['source'] == 'aops_forum') # AoPS contains AIME/AMC
    aime_ds = aime_ds.shuffle(seed=42).select(range(n))
    
    samples = []
    for item in ds:
        samples.append({
            "question": item['problem'],
            "ground_truth": item['solution'],
            "source": "aime_amc"
        })
        if len(samples) >= n:
            break
    return samples

def create_langsmith_dataset():
    if not os.environ.get("LANGCHAIN_API_KEY"):
        raise ValueError("Please set the LANGCHAIN_API_KEY environment variable.")
        
    print(f"Creating LangSmith dataset: {DATASET_NAME}")
    
    # Check if dataset already exists
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        print("Dataset already exists, deleting old one...")
        client.delete_dataset(dataset_id=datasets[0].id)
        
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="A balanced 500-question evaluation dataset containing GSM8K, MATH, and AIME to test agentic reasoning and tool-calling."
    )
    
    # Gather samples
    gsm8k_samples = get_gsm8k_samples(166)
    math_samples = get_math_samples(166)
    aime_samples = get_aime_samples(168) # 168 to make exactly 500
    
    all_samples = gsm8k_samples + math_samples + aime_samples
    random.seed(42)
    random.shuffle(all_samples)
    
    print(f"Uploading {len(all_samples)} examples to LangSmith...")
    
    inputs = [{"question": s["question"]} for s in all_samples]
    outputs = [{"ground_truth": s["ground_truth"], "source": s["source"]} for s in all_samples]
    
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id
    )
    
    print("Dataset Upload Complete! Ready for evaluation.")

if __name__ == "__main__":
    create_langsmith_dataset()
