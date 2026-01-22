from typing import List

from pydantic import BaseModel


class PruneConfig(BaseModel):
    model_id: str
    sae_release: str
    sae_id: str
    dataset_id: str
    batch_size: int
    max_length: int
    num_samples: int
    top_k_features: List[int]
    prune_thresholds: List[float]


class SFTConfig(BaseModel):
    max_seq_length: int
    learning_rate: float
    max_original_samples: int
    batch_size: int


class EvalConfig(BaseModel):
    num_samples: int
    max_new_tokens: int
    batch_size: int


class ExperimentConfig(BaseModel):
    suite_title: str
    tokenizer_id: str
    base_model_id: str
    prune: PruneConfig
    sft: SFTConfig
    eval: EvalConfig
    output_dir: str
    seed: int

    @staticmethod
    def from_yaml(path: str) -> "ExperimentConfig":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return ExperimentConfig(**data)
