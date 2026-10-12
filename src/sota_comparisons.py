from argparse import ArgumentParser
from pathlib import Path
from typing import List, Literal
from src.prune import get_formatting_function
from src.sft import PEFT_CONFIG, generate_sft_trainer_args
import torch
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl.trainer.sft_trainer import SFTTrainer


from src.schemas.suite import SuiteConfig
from src.eval import eval_and_save
from src.utils.enums import ModelType
from src.utils.ensure_pad_token import ensure_pad_token
from src.sota_methods.grape import GrapePipeline
from src.sota_methods.s2l import select_s2l
from src.utils.load_model_and_tokenizer import load_model_and_tokenizer


def save_sota_df(sota_df: pd.DataFrame, output_dir: Path, filename: str) -> pd.DataFrame | None:
    """
    Saves a DataFrame to CSV, Markdown, and Parquet formats.
    """
    # 1. Input Validation
    if sota_df is None or sota_df.empty:
        print("[WARNING] Input DataFrame is empty or None. Aborting save.")
        return None

    # 2. Ensure output directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"[ERROR] Permission denied to create directory: {output_dir}")
        raise

    clean_filename = Path(filename).stem
    
    # 3. Define export formats and methods
    file_formats = {
        "CSV": (output_dir / f"{clean_filename}.csv", lambda p: sota_df.to_csv(p, index=False)),
        "Markdown": (output_dir / f"{clean_filename}.md", lambda p: sota_df.to_markdown(p, index=False)),
        "Parquet": (output_dir / f"{clean_filename}.parquet", lambda p: sota_df.to_parquet(p, index=False))
    }

    # 4. Save to all formats
    for fmt, (path, save_func) in file_formats.items():
        try:
            save_func(path)
            print(f"[INFO] Saved {fmt} to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save {fmt} file to {path}. Error: {e}")

    print("\nProcessed DataFrame:")
    print(sota_df)

    return sota_df

def olmo3_comp(suite: SuiteConfig):
    rows = []
    filename = "olmo3_comp_pt"
    sota_dir = suite.dirs.ensure_dir(suite.dirs.main_results / "sota")
    OLMO3_COMP = "allenai/Olmo-3-7B-Instruct-SFT"

    # for model_name in LLAMA2_SOTA_MODELS:
    # tokenizer = model_name if "detia" not in model_name else TOKENIZER
    model = AutoModelForCausalLM.from_pretrained(OLMO3_COMP, device_map="auto", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(OLMO3_COMP, fix_mistral_regex=True)
    model, tokenizer = ensure_pad_token(model=model, tokenizer=tokenizer)

    sota_gen_cache_dir = suite.dirs.ensure_dir(
        sota_dir
        / OLMO3_COMP
        / suite.dirs.get_generation_cache_subdir(
            max_new_tokens=suite.eval.cfg.lm_eval.max_new_tokens)
    )

    sota_results, _ = eval_and_save(
        model=model,
        tokenizer=tokenizer,
        suite=suite,
        model_type=ModelType.SOTA,
        dataset_size=None,
        topk=None,
        save_results_and_config=False,
        gen_cache_dir=sota_gen_cache_dir,
    )
    sota_results.update({"model_name": Path(OLMO3_COMP).name})
    rows.append(sota_results)

    sota_df = pd.DataFrame(rows)
    # save_sota_df(sota_df=sota_df, output_dir=sota_results_dir, filename="662_modern_comps_redux")

    lm_df = pd.json_normalize(sota_df['lm_eval_results'])

    # 2. Flatten the 'reference_results' column (this handles the deep nesting!)
    ref_df = pd.json_normalize(sota_df['reference_results'])

    # 3. Grab your standard column(s)
    model_df = sota_df[['model_name']]

    # 4. Concatenate them all together side-by-side
    final_df = pd.concat([model_df, lm_df, ref_df], axis=1)
    save_sota_df(sota_df=final_df, output_dir=sota_dir, filename=filename)
    # sota_df = pd.json_normalize(sota_df['0'])
    # print(sota_df)

    # sota_df.to_csv(sota_dir / f"{filename}.csv")
    # sota_df.to_markdown(sota_dir / f"{filename}.md")
    # sota_df.to_parquet(sota_dir / f"{filename}.parquet")
    # print(sota_df)


def llama2_comps(suite: SuiteConfig):
    rows = []
    # LLAMA2_BASE_MODEL = "meta-llama/Llama-2-13b-hf"
    TOKENIZER = "meta-llama/Llama-2-13b-chat-hf"
    LLAMA2_SOTA_MODELS = [
        "WizardLMTeam/WizardLM-13B-V1.2",
        "lmsys/vicuna-13b-v1.5",
        "hkust-nlp/deita-llama2-13b-v1.0-sft",
        "meta-llama/Llama-2-13b-chat-hf",
        "allenai/tulu-2-13b",
    ]
    RESULTS_FILENAME = "llama2_comps"

    sota_dir = suite.dirs.ensure_dir(suite.dirs.main_results / "sota")

    for model_name in LLAMA2_SOTA_MODELS:
        # tokenizer = model_name if "detia" not in model_name else TOKENIZER
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, fix_mistral_regex=True)
        model, tokenizer = ensure_pad_token(model=model, tokenizer=tokenizer)

        sota_gen_cache_dir = suite.dirs.ensure_dir(
            sota_dir
            / model_name
            / suite.dirs.get_generation_cache_subdir(
                max_new_tokens=suite.eval.cfg.lm_eval.max_new_tokens)
        )

        sota_results, _ = eval_and_save(
            model=model,
            tokenizer=tokenizer,
            suite=suite,
            model_type=ModelType.SOTA,
            dataset_size=None,
            topk=None,
            save_results_and_config=False,
            gen_cache_dir=sota_gen_cache_dir,
        )
        sota_results.update({"model_name": Path(model_name).name})
        rows.append(json.dumps(sota_results))

    sota_df = pd.DataFrame(rows)

    sota_df = pd.json_normalize(sota_df['0'])
    print(sota_df)

    sota_df.to_csv(sota_dir / f"{RESULTS_FILENAME}.csv")
    sota_df.to_markdown(sota_dir / f"{RESULTS_FILENAME}.md")
    sota_df.to_parquet(sota_dir / f"{RESULTS_FILENAME}.parquet")
    print(sota_df)

"""
TODO: implement more modern comps
    - Methods: GRAPE and S2L
    - Datasets: Tulu SFT mixture
    - Base Models: Llama 3.1 8B, Qwen 3 8B, Gemma 3 12B PT
    - S2L Proxy Models: Llama 3.2 1B, Qwen 3 0.6B, Gemma 3 270M PT
    - STEPS:
        1) Generate datasets
        2) Run SFT 
        3) Eval
"""
def modern_comps_olmo3(suite: SuiteConfig):
    # base_dataset_id = "allenai/tulu-3-sft-mixture"
    base_dataset_id = "HuggingFaceH4/ultrachat_200k"
    # base_models = ["meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B", "google/gemma-3-12b-pt"]
    grape_teacher_models = ["Qwen/Qwen3-14B", "google/gemma-3-12b-it"]  # ["meta-llama/Llama-3.3-70B-Instruct", 
    # s2l_proxy_models = ["meta-llama/Llama-3.2-1B", "Qwen/Qwen3-0.6B", "google/gemma-3-270m-pt"]
    base_models = ["allenai/Olmo-3-1025-7B"]
    base_tokenizers = ["allenai/Olmo-3-7B-Instruct-SFT"]
    s2l_proxy_models = ["allenai/OLMo-2-0425-1B"]  # different gen but much smaller than 7B which is the smallest olmo2 option
    proxy_tokenizers = ["allenai/OLMo-2-0425-1B-SFT"]

    run_modern_comps(suite, base_dataset_id, grape_teacher_models, base_models, base_tokenizers, s2l_proxy_models, proxy_tokenizers)


def modern_comps_llama2(suite: SuiteConfig):
    # base_dataset_id = "allenai/tulu-3-sft-mixture"
    base_dataset_id = "yahma/alpaca-cleaned"
    # base_models = ["meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B", "google/gemma-3-12b-pt"]
    grape_teacher_models = ["Qwen/Qwen3-32B", "google/gemma-3-27b-it"]  # ["meta-llama/Llama-3.3-70B-Instruct", 
    # s2l_proxy_models = ["meta-llama/Llama-3.2-1B", "Qwen/Qwen3-0.6B", "google/gemma-3-270m-pt"]
    base_models = ["meta-llama/Llama-2-13b-hf"]
    base_tokenizers = ["meta-llama/Llama-2-13b-chat-hf"]
    s2l_proxy_models = ["meta-llama/Llama-3.2-1B"]  # different gen but much smaller than 7B which is the smallest llama2 option
    proxy_tokenizers = ["meta-llama/Llama-3.2-1B-Instruct"]

    run_modern_comps(suite, base_dataset_id, grape_teacher_models, base_models, base_tokenizers, s2l_proxy_models, proxy_tokenizers)


def run_modern_comps(suite, base_dataset_id, grape_teacher_models, base_models, base_tokenizers, s2l_proxy_models, proxy_tokenizers):

    # def get_repos(base_model_id: str) -> List[str]:
    #     base_repos = ["ryanaswift7/fdp-662-grape-alpaca-sft", "ryanaswift7/fdp-662-s2l-alpaca-sft"]
    #     model_class = Path(base_model_id).name
    #     return [f"{base_repos[0]}-{model_class}", base_repos[1]]
    

    def get_repos(base_model_id: str) -> List[str]:
        base_repos = ["ryanaswift7/fdp-662-grape-ultrachat-sft", "ryanaswift7/fdp-662-s2l-ultrachat-sft"]
        model_class = Path(base_model_id).name
        return [f"{base_repos[0]}-{model_class}", base_repos[1]]

    """
    NOTE:   There is a separate repo for each grape base model,
            but a single repo with different subsets for each S2L proxy model.
            This is why s2l can be excluded from the loop
    """
    for base_model, base_tokenizer in zip(base_models, base_tokenizers):
        # use final 'grape_selected' subset for training
        repos = get_repos(base_model_id=base_model)
        grape = GrapePipeline(
            base_model_id=base_model,
            base_tokenizer_id=base_tokenizer,
            hf_repo_id=repos[0],
            teacher_ids=grape_teacher_models,
            dataset_id=base_dataset_id,
            prompt_column="messages",
            split="train_sft",  # for ultrachat
            # sample_limit=10,
        )
        grape.run()

    s2l = S2LConsensusEngine(
        suite=suite,
        dataset_name=base_dataset_id, 
        hub_repo_id=repos[1],
        proxy_models=s2l_proxy_models,
        proxy_tokenizers=proxy_tokenizers,
        max_length=suite.sft.cfg.max_seq_length,
        ratio=0.193,  # about 10k/51760 samples of alpaca-cleaned
        batch_size=8,
        split="train_sft",
        # sample_limit=10,
    )

    s2l.run()

#     # TODO: implement SFT and evals

    rows = []
    # Define standard output directory once
    sota_results_dir = suite.dirs._ensure_dir(suite.dirs.results / "sota")
    sota_models_dir = suite.dirs._ensure_dir(suite.dirs.models / "sota")

    for base_model, base_tokenizer in zip(base_models, base_tokenizers):
        repos = get_repos(base_model_id=base_model)
        data_limit_int = 10_000 if "olmo" not in base_model.lower() else 40_000
        data_limit_str = "10k" if "olmo" not in base_model.lower() else "40k"
        
        # Configure the two different training routines to loop through
        setups = [
            {
                "name": "GRAPE",
                "repo": repos[0],
                "dataset_subset": "grape_selected",
                "sort_by": "grape_score",
                "limit": data_limit_int,
                "model_name": f"GRAPE-{data_limit_str}-{Path(base_model).name}",
                "results_file": "grape_results.json"
            },
            {
                "name": "S2L",
                "repo": repos[1],
                "dataset_subset": "OLMo_2_0425_1B", # "Llama_3.2_1B",
                "sort_by": None,
                "limit": None,
                "model_name": f"S2L-{data_limit_str}-{Path(base_model).name}",
                "results_file": "s2l_results.json"
            }
        ]

        for setup in setups:
            model_dir = suite.dirs._ensure_dir(sota_models_dir / setup["model_name"])
            results_path = model_dir / setup["results_file"]

            # ---------------------------------------------------------
            # CHECK 1: Are results already generated? (Skip EVERYTHING)
            # ---------------------------------------------------------
            if results_path.exists():
                print(f"[INFO] Results found for {setup['model_name']}. Skipping SFT and Eval.")
                try:
                    with open(results_path, "r") as f:
                        sota_results = json.load(f)
                    rows.append(sota_results)
                except Exception as e:
                    print(f"[ERROR] Failed to load existing results for {setup['model_name']}: {e}")
                continue

            # ---------------------------------------------------------
            # CHECK 2: Is the model already trained? (Skip SFT)
            # ---------------------------------------------------------
            if (model_dir / "config.json").exists():
                print(f"[INFO] Merged model found for {setup['model_name']}. Skipping SFT.")
                
                # Load the already trained model from the local directory for evaluation
                model, tokenizer = load_model_and_tokenizer(
                    model_id=str(model_dir), 
                    tokenizer_id=str(model_dir)
                )
            
            # ---------------------------------------------------------
            # RUN SFT: If no model exists, train and save
            # ---------------------------------------------------------
            else:
                print(f"[INFO] Training {setup['model_name']}...")
                
                # 1. Load Base Model & Tokenizer
                model, tokenizer = load_model_and_tokenizer(
                    model_id=base_model, 
                    tokenizer_id=base_tokenizer
                )
                
                # 2. Load and Prepare Dataset
                dataset = load_dataset(setup["repo"], setup["dataset_subset"], split="train")
                

                if setup["sort_by"]:
                    dataset = dataset.sort(setup["sort_by"], reverse=True)
                if setup["limit"]:
                    dataset = dataset.select(range(setup["limit"]))

                if "s2l" in setup["repo"].lower():
                    formatter_func = get_formatting_function(setup["repo"])

                    dataset = dataset.map(
                        lambda examples: formatter_func(tokenizer, examples, is_inference=False),
                        batched=True,
                        batch_size=1000,  # batch_size,
                        num_proc=12,
                    )

                # 3. SFT Training
                trainer = SFTTrainer(
                    model=model,
                    args=generate_sft_trainer_args(
                        suite=suite,
                        output_dir=None,
                        num_training_examples=len(dataset),
                    ),
                    train_dataset=dataset,
                    processing_class=tokenizer,
                    peft_config=PEFT_CONFIG,
                )
                trainer.train()

                # 4. Save Merged Model
                merged_model = trainer.model.merge_and_unload()
                merged_model.save_pretrained(str(model_dir))
                tokenizer.save_pretrained(str(model_dir))
                
                # Explicitly point our model variable to the merged model so eval 
                # uses the exact same artifact whether trained fresh or loaded from disk.
                model = merged_model

            # ---------------------------------------------------------
            # RUN EVALUATION
            # ---------------------------------------------------------
            print(f"[INFO] Evaluating {setup['model_name']}...")
            
            gen_cache_dir = suite.dirs._ensure_dir(
                model_dir / suite.dirs.get_generation_cache_subdir(
                    max_new_tokens=suite.eval.cfg.lm_eval.max_new_tokens
                )
            )

            sota_results, _ = eval_and_save(
                model=model,
                tokenizer=tokenizer,
                suite=suite,
                model_type=ModelType.SOTA,
                dataset_size=None,
                topk=None,
                save_results_and_config=False,
                gen_cache_dir=gen_cache_dir,
            )
            
            # Save and Record Results
            sota_results["model_name"] = setup["model_name"]
            
            with open(results_path, "w") as f:
                json.dump(sota_results, f, indent=4)
            
            rows.append(sota_results)

    # Generate final SOTA DataFrame
    sota_df = pd.DataFrame(rows)
    # save_sota_df(sota_df=sota_df, output_dir=sota_results_dir, filename="662_modern_comps_redux")

    lm_df = pd.json_normalize(sota_df['lm_eval_results'])

    # 2. Flatten the 'reference_results' column (this handles the deep nesting!)
    ref_df = pd.json_normalize(sota_df['reference_results'])

    # 3. Grab your standard column(s)
    model_df = sota_df[['model_name']]

    # 4. Concatenate them all together side-by-side
    final_df = pd.concat([model_df, lm_df, ref_df], axis=1)
    save_sota_df(sota_df=final_df, output_dir=sota_results_dir, filename="662_modern_comps")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the suite configuration file",
    )
    return parser.parse_args()

def main():
    # Parse args and load config inside main()
    args = parse_args()
    suite = SuiteConfig.from_yaml(args.config)
    
    # llama2_comps(suite=suite)
    # olmo3_comp(suite=suite)
    # modern_comps_llama2(suite=suite)
    modern_comps_olmo3(suite=suite)

if __name__ == "__main__":
    main()
