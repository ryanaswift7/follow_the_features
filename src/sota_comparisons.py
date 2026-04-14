from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
from transformers import AutoModel, AutoTokenizer

from src.schemas.suite import SuiteConfig
from src.eval import eval_and_save
from src.utils.enums import ModelType
from src.utils.sota.grape import GrapePipeline
from src.utils.sota.s2l import S2LConsensusEngine


def llama2_comps(suite: SuiteConfig):
    rows = []
    # LLAMA2_BASE_MODEL = "meta-llama/Llama-2-13b-hf"
    # TOKENIZER
    LLAMA2_SOTA_MODELS = [
        "WizardLMTeam/WizardLM-13B-V1.2",
        "lmsys/vicuna-13b-v1.5",
        "hkust-nlp/deita-llama2-13b-v1.0-sft",
        "meta-llama/Llama-2-13b-chat-hf",
        "allenai/tulu-2-13b",
    ]


    for model_name in LLAMA2_SOTA_MODELS:
        model = AutoModel.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sota_results, sota_cfg = eval_and_save(
            model=model,
            tokenizer=tokenizer,
            suite=suite,
            model_type=ModelType.REFERENCE,
            dataset_size=None,
            topk=None,
            save_results_and_config=False,
        )
        sota_results.update({"model_name": Path(model_name).name})
        rows.append(sota_results)

    sota_df = pd.DataFrame(rows)
    sota_df.to_csv(suite.dirs.results / "sota" / "llama2_comps.csv")
    sota_df.to_parquet(suite.dirs.results / "sota" / "llama2_comps.parquet")
    sota_df.to_markdown(suite.dirs.results / "sota" / "llama2_comps.md")
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
def modern_comps(suite: SuiteConfig):
    base_dataset_id = "allenai/tulu-3-sft-mixture"
    base_models = ["meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B", "google/gemma-3-12b-pt"]
    grape_teacher_models = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-32B", "google/gemma-3-27b-it"]
    s2l_proxy_models = ["meta-llama/Llama-3.2-1B", "Qwen/Qwen3-0.6B", "google/gemma-3-270m-pt"]

    def get_repos(base_model_id: str) -> List[str]:
        base_repos = ["ryanaswift7/fdp-grape-tulu-sft", "ryanaswift7/fdp-s2l-tulu-sft"]
        model_class = Path(base_model_id).name
        return [f"{base_repos[0]}-{model_class}", base_repos[1]]

    """
    NOTE:   There is a separate repo for each grape base model,
            but a single repo with different subsets for each S2L proxy model
    """
    for base in base_models:
        # use final 'grape_selected' subset
        repos = get_repos(base_model_id=base)
        grape = GrapePipeline(
            base_model_id=base,
            hf_repo_id=repos[0],
            teacher_ids=grape_teacher_models,
            dataset_id=base_dataset_id,
            prompt_column="messages",
            split="train",
            sample_limit=10,
        )
        grape.run()

    s2l = S2LConsensusEngine(
        dataset_name=base_dataset_id, 
        hub_repo_id=repos[1],
        proxies=s2l_proxy_models,
        ratio=0.1,
        batch_size=8,
        sample_limit=50
    )

    s2l.run()

    # TODO: implement SFT and evals





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

def main(suite: SuiteConfig):
    llama2_comps(suite=suite)
    # modern_comps(suite=suite)

if __name__ == "__main__":
    args = parse_args()
    suite = SuiteConfig.from_yaml(args.config)
    main(suite=suite)
