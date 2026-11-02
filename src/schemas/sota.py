from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class SOTAGrapeConfig(BaseModel):
    teacher_ids: list[str] = Field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.7
    batch_size: int = 1


class SOTAS2LConfig(BaseModel):
    proxy_model_id: str
    proxy_tokenizer_id: str
    trajectory_checkpoints: list[int] = Field(min_length=1)
    n_clusters: int = Field(default=100, gt=0)


class SOTAConfig(BaseModel):
    dataset_id: str
    split: str = "train"
    output_dir: Path | None = None
    seeds: list[int] | None = None
    grape: SOTAGrapeConfig
    s2l: SOTAS2LConfig

    @model_validator(mode="after")
    def validate_checkpoints(self):
        if self.s2l.trajectory_checkpoints != sorted(set(self.s2l.trajectory_checkpoints)):
            raise ValueError("s2l.trajectory_checkpoints must be sorted and unique")
        if self.seeds is not None and (
            not self.seeds or len(self.seeds) != len(set(self.seeds))
        ):
            raise ValueError("sota.seeds must be non-empty and unique when provided")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SOTAConfig":
        import yaml

        with open(path, "r") as handle:
            return cls(**yaml.safe_load(handle))
