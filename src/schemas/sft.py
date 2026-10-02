from typing import Literal

from pydantic import BaseModel


SchedulerType = Literal[
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
    "reduce_lr_on_plateau",
    "cosine_with_min_lr",
    "cosine_warmup_with_min_lr",
    "warmup_stable_decay",
    "greedy",
]


class SFTConfig(BaseModel):
    max_seq_length: int
    learning_rate: float
    lr_scheduler_type: SchedulerType
    tokenizer_id: str
    base_model_id: str
    full_ft: bool


class SFTParams(BaseModel):
    batch_size: int


class SFTSetup(BaseModel):
    cfg: SFTConfig
    params: SFTParams
