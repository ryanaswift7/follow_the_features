import gc
from src.prune import convert_alpaca_to_messages, get_formatting_function
import torch
import numpy as np
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset, get_dataset_config_names
from huggingface_hub import create_repo
import multiprocessing as mp
from tqdm.auto import tqdm


def format_grape_prompt(tokenizer, instruction):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question: {instruction}\nAnswer: "


def select_grape_responses(model, tokenizer, candidates, budget, batch_size=1):
    """Select one highest normalized-probability response per instruction.

    ``candidates`` is a sequence of dictionaries with ``instruction`` and a
    ``responses`` list.  This is the response-level GRAPE operation from the
    paper, kept independent of Hub persistence and teacher generation.
    """
    if budget > len(candidates):
        raise ValueError(
            "GRAPE cannot satisfy the requested budget: the budget exceeds "
            f"the number of unique instructions ({budget} > {len(candidates)}). "
            "GRAPE selects at most one response per instruction."
        )
    if budget < 0:
        raise ValueError("GRAPE budget must be non-negative")

    device = next(model.parameters()).device
    scored = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            prompts = [
                format_grape_prompt(tokenizer, item["instruction"])
                for item in batch
            ]
            for item, prompt in zip(batch, prompts):
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                response_scores = []
                for response in item["responses"]:
                    full = tokenizer(
                        prompt + response,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    inputs = {key: value.to(device) for key, value in full.items()}
                    logits = model(**inputs).logits[0, :-1]
                    labels = inputs["input_ids"][0, 1:]
                    logprobs = torch.log_softmax(logits, dim=-1)
                    token_logprobs = torch.gather(
                        logprobs, -1, labels.unsqueeze(-1)
                    ).squeeze(-1)
                    response_logprobs = token_logprobs[len(prompt_ids) - 1 :]
                    response_scores.append(
                        response_logprobs.mean().item()
                        if response_logprobs.numel()
                        else float("-inf")
                    )
                best = int(np.argmax(response_scores))
                scored.append({**item, "score": response_scores[best], "best": best})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return [
        {
            "messages": [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["responses"][item["best"]]},
            ],
            "grape_score": item["score"],
            "generator": (
                item.get("sources", [])[item["best"]]
                if item.get("sources")
                else None
            ),
        }
        for item in scored[:budget]
    ]

class GrapePipeline:
    def __init__(
            self, 
            base_model_id,
            base_tokenizer_id,
            hf_repo_id, 
            teacher_ids,
            dataset_id,
            prompt_column, 
            split, 
            sample_limit=None,
            gpu_utilization=0.95,
            max_tokens=512,
            max_model_len=2048):
        
        # Models & Hub Config
        self.base_model_id = base_model_id
        self.base_tokenizer_id = base_tokenizer_id
        self.hf_repo_id = hf_repo_id
        self.teacher_ids = teacher_ids
        
        # Dataset Config
        self.dataset_id = dataset_id
        self.prompt_column = prompt_column
        self.split = split
        self.sample_limit = sample_limit
        
        # vLLM Config
        self.gpu_utilization = gpu_utilization
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len

        # Initialize Hub Repo
        create_repo(hf_repo_id, repo_type="dataset", exist_ok=True)

    def _clean_vram(self):
        """Internal helper to reset GPU state."""
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _get_instructions(self, existing_configs):
        """Loads dataset from Hugging Face and extracts the original responses as a baseline candidate."""
        if not self.dataset_id:
            raise ValueError("A valid 'dataset_id' must be provided.")

        print(f"📥 Loading dataset '{self.dataset_id}' (split: {self.split})...")
        ds = load_dataset(self.dataset_id, split=self.split)
        # formatter_func = get_formatting_function(self.dataset_id)
        # ds = ds.map(
        #     lambda examples: formatter_func(tokenizer, examples, is_inference=False),
        #     batched=True,
        #     batch_size=1000,  # batch_size,
        #     num_proc=12,
        # )
        
        if self.sample_limit and self.sample_limit < len(ds):
            print(f"🎲 Subsampling {self.sample_limit} records from the dataset...")
            ds = ds.shuffle(seed=42).select(range(self.sample_limit))

        if "alpaca" in self.dataset_id.lower():
            ds = ds.map(
                convert_alpaca_to_messages,
                batched=True,
                batch_size=1000,
                num_proc=12,
                remove_columns=["instruction", "input", "output"],
            )
            
        raw_prompts = ds[self.prompt_column]
        instructions = []
        original_data = []
        
        if len(raw_prompts) > 0 and isinstance(raw_prompts[0], list) and isinstance(raw_prompts[0][0], dict):
            print(f"💬 Detected chat format in '{self.prompt_column}'. Extracting users & original assistants...")
            for msgs in raw_prompts:
                user_content = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                if not user_content and len(msgs) > 0:
                    user_content = msgs[0].get("content", "")
                instructions.append(user_content)
                
                assistant_content = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
                if assistant_content:
                    original_data.append({
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    })
        else:
            print(f"✅ Extracted standard prompts from column '{self.prompt_column}'.")
            instructions = raw_prompts
            
            response_col = "response" if "response" in ds.column_names else ("output" if "output" in ds.column_names else None)
            if response_col:
                print(f"📝 Found original responses in column '{response_col}'.")
                for inst, resp in zip(instructions, ds[response_col]):
                    original_data.append({
                        "messages": [
                            {"role": "user", "content": inst},
                            {"role": "assistant", "content": resp}
                        ]
                    })

        if original_data:
            if "teacher_original" not in existing_configs:
                print(f"💾 Saving original responses as baseline candidate: 'teacher_original' ({len(original_data)} rows)...")
                Dataset.from_list(original_data).push_to_hub(self.hf_repo_id, config_name="teacher_original")
            else:
                print("⏭️ Skipping baseline extraction: 'teacher_original' already on Hub.")
        
        return instructions

    def _run_teacher_phase(self, teacher_id, instructions):
        """Loads a teacher, formats with its chat template, generates data, and pushes to hub."""
        teacher_name = teacher_id.split("/")[-1].replace(".", "-")
        print(f"🚀 [Teacher Phase] Loading: {teacher_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(teacher_id)
        llm = LLM(model=teacher_id, gpu_memory_utilization=self.gpu_utilization)
        params = SamplingParams(temperature=0.7, max_tokens=self.max_tokens)
        
        formatted_prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": inst}], 
                                          tokenize=False, 
                                          add_generation_prompt=True)
            for inst in instructions
        ]
        
        outputs = llm.generate(formatted_prompts, params)
        
        data = [{
            "messages": [
                {"role": "user", "content": inst},
                {"role": "assistant", "content": out.outputs[0].text}
            ]
        } for inst, out in zip(instructions, outputs)]
        
        del llm
        del tokenizer
        self._clean_vram()
        
        ds = Dataset.from_list(data)
        ds.push_to_hub(self.hf_repo_id, config_name=f"teacher_{teacher_name}")
        print(f"✅ Subset 'teacher_{teacher_name}' pushed.")

    def _compile_candidates(self):
        """Merges ALL subsets starting with 'teacher_' into a single candidate pool."""
        print("📦 [Compilation Phase] Merging candidate outputs...")
        dfs = []
        
        def get_role_content(msgs, role):
            if isinstance(msgs, np.ndarray):
                msgs = msgs.tolist()
            return next((m["content"] for m in msgs if m.get("role") == role), "")

        all_configs = get_dataset_config_names(self.hf_repo_id)
        teacher_configs = [c for c in all_configs if c.startswith("teacher_")]

        for config_name in teacher_configs:
            name = config_name.replace("teacher_", "")
            print(f"   ➕ Adding candidate source: {name}")
            df = load_dataset(self.hf_repo_id, config_name, split="train").to_pandas()
            
            df["instruction"] = df["messages"].apply(lambda x: get_role_content(x, "user"))
            df[f"res_{name}"] = df["messages"].apply(lambda x: get_role_content(x, "assistant"))
            dfs.append(df[["instruction", f"res_{name}"]])
        
        combined_df = dfs[0]
        for next_df in dfs[1:]:
            combined_df = combined_df.merge(next_df, on="instruction")
        
        res_cols = [c for c in combined_df.columns if c.startswith("res_")]
        combined_df["all_responses"] = combined_df[res_cols].values.tolist()
        
        sources_list = [c.replace("res_", "") for c in res_cols]
        combined_df["sources"] = [sources_list for _ in range(len(combined_df))]
        
        ds = Dataset.from_pandas(combined_df[["instruction", "all_responses", "sources"]])
        ds.push_to_hub(self.hf_repo_id, config_name="all_candidates")
        print(f"✅ Subset 'all_candidates' pushed with {len(res_cols)} candidate options per prompt.")

    def _run_selection_phase(self):
        """Loads base model natively, scores candidates via GRAPE, and pushes final dataset."""
        print(f"⚖️ [Selection Phase] Loading Base Model: {self.base_model_id}")
        
        candidates_ds = load_dataset(self.hf_repo_id, "all_candidates", split="train")
        tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_id)
        
        # 1. Load the model using standard PyTorch (no vLLM)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, 
            device_map="auto", 
            dtype=torch.bfloat16,
        )
        model.eval()
        
        all_scores = []

        print("🚀 Scoring candidates natively...")
        # 2. Score candidates using a single forward pass per sequence
        with torch.no_grad():
            for idx, item in tqdm(enumerate(candidates_ds), total=len(candidates_ds)):
                messages = [{"role": "user", "content": item['instruction']}]
                prompt_header = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Get the length of the prompt to ignore its tokens during scoring
                prompt_len = tokenizer(prompt_header, return_tensors="pt").input_ids.shape[1]
                
                for res in item["all_responses"]:
                    full_text = prompt_header + str(res)
                    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
                    
                    # Forward pass
                    outputs = model(**inputs)
                    
                    # Shift logits and labels for next-token prediction
                    logits = outputs.logits[0, :-1, :]
                    labels = inputs.input_ids[0, 1:]
                    
                    # Calculate log probabilities
                    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                    token_logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                    
                    # Isolate the response tokens (skip the prompt tokens)
                    response_logprobs = token_logprobs[prompt_len - 1:]
                    
                    score = response_logprobs.mean().item() if len(response_logprobs) > 0 else -100.0
                    all_scores.append(score)

        # 3. Clean up VRAM safely and naturally
        del model
        del tokenizer
        self._clean_vram()

        # 4. Strict GRAPE Selection Logic (No Filtering)
        final_rows = []
        ptr = 0
        win_counts = {}
        
        for idx, item in enumerate(candidates_ds):
            num_cands = len(item["all_responses"])
            subset_scores = all_scores[ptr : ptr + num_cands]
            best_idx = np.argmax(subset_scores)
            best_score = float(subset_scores[best_idx])
            
            winning_source = item["sources"][best_idx]
            
            final_rows.append({
                "messages": [
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["all_responses"][best_idx]}
                ],
                "grape_score": best_score,
                "generator": winning_source
            })
            
            win_counts[winning_source] = win_counts.get(winning_source, 0) + 1
            ptr += num_cands
            
        print("\n🏆 --- Winning Candidate Distribution ---")
        total_kept = len(final_rows)
        for source, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_kept) * 100
            print(f"  • {source}: {pct:.1f}% ({count} samples)")
        print(f"----------------------------------------\n")
        
        Dataset.from_list(final_rows).push_to_hub(self.hf_repo_id, config_name="grape_selected")
        print(f"🎉 Final curated dataset pushed to {self.hf_repo_id} subset 'grape_selected'")

    def run(self, force_restart=False):
        """Executes the end-to-end GRAPE pipeline with resume capabilities."""
        print("🚀 Starting Strict GRAPE Pipeline...")
        
        if force_restart:
            print("⚠️ Force restart enabled. Overwriting any existing progress on the Hub.")
            existing_configs = []
        else:
            try:
                existing_configs = get_dataset_config_names(self.hf_repo_id)
            except Exception:
                existing_configs = [] 
            
        if "grape_selected" in existing_configs:
            print("🎉 Pipeline already complete! 'grape_selected' found on Hub.")
            return

        # Fetch instructions and original data directly
        instructions = self._get_instructions(existing_configs)

        ctx = mp.get_context("spawn")

        for teacher_id in self.teacher_ids:
            teacher_name = teacher_id.split("/")[-1].replace(".", "-")
            config_name = f"teacher_{teacher_name}"
            
            if config_name in existing_configs:
                print(f"⏭️ Skipping {teacher_id}: '{config_name}' already exists on Hub.")
            else:
                print(f"🔥 Starting isolated process for: {teacher_id}")
                
                # Create an isolated subprocess for this specific teacher
                p = ctx.Process(
                    target=self._run_teacher_phase, 
                    args=(teacher_id, instructions)
                )
                
                p.start() # Starts the new process (and initializes CUDA safely inside it)
                p.join()  # Halts the main script until this teacher completely finishes
                
                # Catch any crashes (like Out of Memory errors) inside the subprocess
                if p.exitcode != 0:
                    print(f"❌ Error: Process for {teacher_id} crashed with exit code {p.exitcode}")
                    
        if "all_candidates" in existing_configs:
            print("⏭️ Skipping Compilation: 'all_candidates' already exists on Hub.")
        else:
            self._compile_candidates()
        
        self._run_selection_phase()
        print("✅ Pipeline execution complete!")
