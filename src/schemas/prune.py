from typing import List, Literal

from pydantic import BaseModel


class PruneConfig(BaseModel):
    prune_model_id: str
    sae_release: str
    sae_id: str
    dataset_id: str
    num_samples: int
    seq_agg_method: Literal["sum", "mean", "max", "last_token"]
    feat_agg_method: Literal["sum", "mean"]
    divide_by_entropy: bool


class PruneParams(BaseModel):
    batch_size: int
    dataset_sizes: List[int]
    topk_features: List[int]


class PruneSetup(BaseModel):
    cfg: PruneConfig
    params: PruneParams
