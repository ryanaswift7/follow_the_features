from typing import List, Literal, Optional

from pydantic import BaseModel, Field, computed_field


class PruneConfig(BaseModel):
    prune_model_id: str
    sae_release: str
    sae_id: str
    dataset_id: str
    num_samples: int
    seq_agg_method: Literal["sum", "mean", "max", "last_token"]
    feat_agg_method: Literal["sum", "mean"]
    divide_by_entropy: bool
    diversity_param: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @computed_field
    @property
    def diversity_tag(self) -> str:
        if self.diversity_param is None or self.diversity_param <= 0.0:
            return "divoff"
        pct = int(round(self.diversity_param * 100))
        return f"div{pct}"


class PruneParams(BaseModel):
    batch_size: int
    dataset_sizes: List[int]
    topk_features: List[int]


class PruneSetup(BaseModel):
    cfg: PruneConfig
    params: PruneParams
