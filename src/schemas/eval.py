from typing import List, Optional

from pydantic import BaseModel


class EvalConfig(BaseModel):
    num_samples: int
    max_new_tokens: int
    ollama_judge: str
    num_ref_models: int
    tasks: Optional[List[str]]
    task_limit: Optional[int]


class EvalParams(BaseModel):
    batch_size: int


class EvalSetup(BaseModel):
    cfg: EvalConfig
    params: EvalParams
