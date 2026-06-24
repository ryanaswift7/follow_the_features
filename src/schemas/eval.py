from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LMEvalConfig(BaseModel):
    run: bool = True
    max_new_tokens: int = 256
    tasks: Optional[List[str]] = None
    task_limit: Optional[int] = None


class AlpacaEvalConfig(BaseModel):
    run: bool = True
    backend: Literal["api", "vllm"] = "api"
    model_name: Optional[str] = None
    dataset: Literal["alpaca_eval", "alpaca_eval_gpt4_baseline"] = (
        "alpaca_eval_gpt4_baseline"
    )
    max_instances: Optional[int] = None
    max_new_tokens: int = 256
    judge_max_tokens: int = 100
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 1
    local_server_host: str = "127.0.0.1"
    local_server_port: Optional[int] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    client_kwargs: Dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    num_ref_models: int
    lm_eval: LMEvalConfig = Field(default_factory=LMEvalConfig)
    alpacaeval: AlpacaEvalConfig = Field(default_factory=AlpacaEvalConfig)


class EvalParams(BaseModel):
    batch_size: int


class EvalSetup(BaseModel):
    cfg: EvalConfig
    params: EvalParams
