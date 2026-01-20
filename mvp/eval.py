import gc

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= CONFIGURATION =================
BASE_MODEL_ID = "google/gemma-2-2b"
ADAPTER_MODELS = {
    "Base Model": None,
    "Subset Adapter": "gemma2-2b-alpaca-subset-lora",
    "Full Alpaca Adapter": "gemma2-2b-alpaca-cleaned-lora",
}

# Limit samples per dataset to keep evaluation speed reasonable
# Set to None to evaluate the full test splits (can take hours)
NUM_SAMPLES = 300
MAX_NEW_TOKENS = 256


# ================= DATASET LOADERS =================
def load_eval_datasets(limit=None):
    """
    Loads and normalizes 5 instruction datasets into a standard format:
    [{'prompt': str, 'reference': str, 'source': str}, ...]
    """
    eval_data = {}
    print("Loading Evaluation Datasets...")

    # 1. Dolly Eval (databricks-dolly-15k)
    # No official test split, so we take a random slice from the end
    print(" - Loading Dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.select(range(len(ds) - 1000, len(ds)))  # Take last 1000 as pseudo-test
    if limit:
        ds = ds.select(range(limit))

    eval_data["DollyEval"] = [
        {
            "prompt": f"{row['instruction']}\n{row['context']}".strip(),
            "reference": row["response"],
        }
        for row in ds
    ]

    # 2. Self-Instruct (Using a HF mirror of the eval set)
    print(" - Loading Self-Instruct...")
    # Using the 'eval' set often found in cleaned versions or the user-oriented set
    ds = load_dataset("HuggingFaceH4/instruction-dataset", split="test")
    if limit:
        ds = ds.select(range(limit))

    eval_data["SelfInst"] = [
        {"prompt": row["prompt"], "reference": row["completion"]} for row in ds
    ]

    # 3. Vicuna Eval (Using GPT-4 answers as reference for Rouge)
    print(" - Loading Vicuna Eval...")
    ds = load_dataset("zhengxuanzenwu/vicuna-eval-with-gpt4", split="test")
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    eval_data["VicunaEval"] = [
        {
            "prompt": row["instruction"],
            "reference": row["response"],  # GPT-4 response as gold standard
        }
        for row in ds
    ]

    # # 4. Super-NaturalInstructions (S-NI)
    # print(" - Loading Super-NaturalInstructions (S-NI)...")
    # # Using a core subset to avoid downloading 100GB
    # ds = load_dataset(
    #     "mrm8488/unnatural-instructions-core", split="train"
    # )  # Proxy for diverse instructions
    # if limit:
    #     ds = ds.select(range(limit))

    # eval_data["S-NI"] = [
    #     {"prompt": row["instruction_with_input"], "reference": row["output"]}
    #     for row in ds
    # ]

    # # 5. Unnatural Instructions (UnNI)
    # print(" - Loading Unnatural Instructions...")
    # # We reuse the core dataset above but filter for the 'unnatural' flavor or load a specific subset
    # # Since 'unnatural-instructions-core' is often used for both, we will load a different subset
    # # or strictly the 'sequences' if available. Here we load a specific unnatural proxy.
    # try:
    #     ds = load_dataset(
    #         "fka/awesome-chatgpt-prompts", split="train"
    #     )  # High quality "unnatural" creative prompts
    #     if limit:
    #         ds = ds.select(range(limit))
    #     eval_data["UnNI"] = [
    #         {"prompt": row["prompt"], "reference": "N/A - Creative"} for row in ds
    #     ]
    #     # NOTE: UnNI usually doesn't have fixed answers (creative).
    #     # For Rouge, we need ground truth. Reverting to a dataset with answers:
    #     ds = load_dataset("truthful_qa", "generation", split="validation")
    #     if limit:
    #         ds = ds.select(range(limit))
    #     eval_data["UnNI (Proxy:TQA)"] = [
    #         {"prompt": row["question"], "reference": row["best_answer"]} for row in ds
    #     ]
    # except:
    #     print(" ! Could not load UnNI proxy, skipping.")

    return eval_data


# ================= EVALUATION LOGIC =================


def generate_responses(model, tokenizer, dataset, batch_size=4):
    """Generates responses for a dataset."""
    model.eval()
    predictions = []
    references = []

    # Simple batching
    prompts = [x["prompt"] for x in dataset]
    refs = [x["reference"] for x in dataset]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        # Apply Chat Template (Simple user/assistant format)
        formatted_prompts = []
        for p in batch_prompts:
            msgs = [{"role": "user", "content": p}]
            formatted_prompts.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Greedy decoding for deterministic eval
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and strip the prompt
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for j, text in enumerate(decoded):
            # Heuristic to strip input prompt if it leaks into output
            # (Gemma chat template usually handles this, but we ensure cleanliness)
            response = text.split("model")[-1] if "model" in text else text
            predictions.append(response.strip())

    return predictions, refs


def main():
    # 1. Prepare Datasets
    datasets_map = load_eval_datasets(limit=NUM_SAMPLES)
    rouge = evaluate.load("rouge")

    final_results = []

    # 2. Iterate Models
    for model_name, adapter_path in ADAPTER_MODELS.items():
        print(f"\n\n{'=' * 20} Evaluating: {model_name} {'=' * 20}")

        # Load Model
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        tokenizer.padding_side = "left"  # Key for batch generation

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
        )

        if adapter_path:
            print(f"Loading Adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        model_scores = {"Model": model_name}

        # 3. Evaluate on each Dataset
        for ds_name, data in datasets_map.items():
            print(f"Running {ds_name}...")
            preds, refs = generate_responses(model, tokenizer, data)

            # Compute ROUGE
            # rougeLsum is often preferred for summary/chat tasks over raw rougeL
            scores = rouge.compute(
                predictions=preds, references=refs, use_aggregator=True
            )
            model_scores[ds_name] = round(scores["rougeL"] * 100, 2)

            # Print a quick sample to verify it's not outputting garbage
            print(f"Sample [{ds_name}]:")
            print(f"Prompt: {data[0]['prompt'][:50]}...")
            print(f"Pred:   {preds[0][:50]}...")
            print(f"Ref:    {refs[0][:50]}...")

        final_results.append(model_scores)

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ================= RESULTS =================
    print("\n\nFINAL ROUGE-L SCORES")
    df = pd.DataFrame(final_results)
    print(df.to_markdown(index=False))
    df.to_csv("rouge_benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
