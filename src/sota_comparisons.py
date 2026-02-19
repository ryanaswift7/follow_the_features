from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from transformers import AutoModel, AutoTokenizer

from schemas.suite import SuiteConfig
from src.eval import eval_and_save
from utils.enums import ModelType


def main(suite: SuiteConfig):
    rows = []
    BASE_MODEL = "meta-llama/Llama-2-13b"
    SOTA_MODELS = [
        "WizardLMTeam/WizardLM-13B-V1.2",
        "lmsys/vicuna-13b-v1.5",
        "hkust-nlp/deita-llama2-13b-v1.0-sft",
        "meta-llama/Llama-2-13b-chat-hf",
        "allenai/tulu-2-13b",
    ]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    for model_name in SOTA_MODELS:
        model = AutoModel.from_pretrained(model_name, device_map="auto")
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
    sota_df.to_csv(suite.dirs.results / "sota" / "sota_comparisons.csv")
    sota_df.to_parquet(suite.dirs.results / "sota" / "sota_comparisons.parquet")
    print(sota_df)


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


if __name__ == "__main__":
    args = parse_args()
    suite = SuiteConfig.from_yaml(args.config)
    main(suite=suite)
