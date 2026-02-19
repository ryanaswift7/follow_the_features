from pydantic import BaseModel


class SFTConfig(BaseModel):
    max_seq_length: int
    learning_rate: float
    max_original_samples: int
    tokenizer_id: str
    base_model_id: str


class SFTParams(BaseModel):
    batch_size: int


class SFTSetup(BaseModel):
    cfg: SFTConfig
    params: SFTParams
