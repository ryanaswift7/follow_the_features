from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LMEvalConfig(BaseModel):
    run: bool = True
    max_new_tokens: int = 256
    tasks: Optional[List[str]] = None
    task_limit: Optional[int] = None


class AlpacaEvalConfig(BaseModel):
    run: bool = True
    candidate_backend: Literal["huggingface", "vllm"] = "huggingface"
    judge_backend: Literal["api", "vllm"] = "api"
    model_name: Optional[str] = None
    candidate_dtype: str = "bfloat16"
    judge_dtype: str = "auto"
    dataset: Literal["alpaca_eval", "alpaca_eval_gpt4_baseline"] = (
        "alpaca_eval_gpt4_baseline"
    )
    max_instances: Optional[int] = None
    max_new_tokens: int = 256
    judge_max_tokens: int = 512
    judge_enable_thinking: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    candidate_model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    judge_model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_candidate_backend(self) -> Literal["huggingface", "vllm"]:
        return self.candidate_backend

    @property
    def effective_judge_backend(self) -> Literal["api", "vllm"]:
        return self.judge_backend

    @property
    def effective_judge_model_kwargs(self) -> Dict[str, Any]:
        return self.judge_model_kwargs or self.model_kwargs


class EvalConfig(BaseModel):
    lm_eval: LMEvalConfig = Field(default_factory=LMEvalConfig)
    alpacaeval: AlpacaEvalConfig = Field(default_factory=AlpacaEvalConfig)


class EvalParams(BaseModel):
    batch_size: int


class EvalSetup(BaseModel):
    cfg: EvalConfig
    params: EvalParams
