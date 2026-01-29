import time
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, computed_field

from src.utils.hashing import config_hash, string_hash


class PruneConfig(BaseModel):
    model_id: str
    sae_release: str
    sae_id: str
    dataset_id: str
    num_samples: int
    seq_agg_method: Literal["sum", "mean", "max"]
    feat_agg_method: Literal["sum", "mean"]


class PruneParams(BaseModel):
    batch_size: int
    dataset_sizes: List[int]
    topk_features: List[int]


class PruneSetup(BaseModel):
    cfg: PruneConfig
    params: PruneParams


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


class EvalConfig(BaseModel):
    num_samples: int
    max_new_tokens: int


class EvalParams(BaseModel):
    batch_size: int


class EvalSetup(BaseModel):
    cfg: EvalConfig
    params: EvalParams


class SuiteIDs(BaseModel):
    prune: str
    sft: str
    eval: str
    experiment: str


class SuiteDirs(BaseModel):
    suite_root: Path
    ids: SuiteIDs
    timestamp: str

    base_model_id: str
    prune_cfg: PruneConfig
    sft_cfg: SFTConfig

    def _ensure_dir(self, path: Path) -> Path:
        """Creates the directory if it doesn't exist and returns the path."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def models(self) -> Path:
        return self._ensure_dir(self.suite_root / "models")

    @computed_field
    @property
    def datasets(self) -> Path:
        return self._ensure_dir(self.suite_root / "datasets" / self.ids.prune)

    @computed_field
    @property
    def results(self) -> Path:
        return self._ensure_dir(
            self.suite_root / "results" / self.ids.experiment / self.timestamp
        )

    @computed_field
    @property
    def base_model_eval(self) -> Path:
        name = Path(self.sft_cfg.base_model_id).name
        return self._ensure_dir(self.models / "base_model" / name / self.ids.eval)

    @computed_field
    @property
    def supervisor_model_eval(self) -> Path:
        name = Path(self.prune_cfg.model_id).name
        return self._ensure_dir(self.models / "supervisor_model" / name / self.ids.eval)

    def get_dataset_path(self, dataset_size: int, topk: int) -> Path:
        """
        Returns path to the dataset file/folder.
        Note: We do NOT auto-create this specific path, as it is an artifact
        that will be written by the pruning script.
        """
        name = Path(self.prune_cfg.dataset_id).name
        return self.datasets / f"{name}_ds{dataset_size}_topk{topk}.hf"

    def get_trained_model_root(self, dataset_size: int, topk: int) -> Path:
        name = Path(self.sft_cfg.base_model_id).name
        return self._ensure_dir(
            self.models
            / "trained"
            / f"{self.ids.prune}_{self.ids.sft}"
            / f"ds{dataset_size}_topk{topk}_{name}"
        )

    def get_trained_model_eval(self, dataset_size: int, topk: int) -> Path:
        return self._ensure_dir(
            self.get_trained_model_root(dataset_size, topk)
            / self.ids.eval
            / f"top{topk}"
        )


class SuiteConfig(BaseModel):
    title: str
    output_dir: Path
    seed: int

    prune: PruneSetup
    sft: SFTSetup
    eval: EvalSetup

    timestamp: str = Field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SuiteConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))

    @computed_field
    @property
    def ids(self) -> SuiteIDs:
        # Calculate hashes
        p_id = f"prune_{config_hash(self.prune.cfg.model_dump())}"
        s_id = f"sft_{config_hash(self.sft.cfg.model_dump())}"
        e_id = f"eval_{config_hash(self.eval.cfg.model_dump())}"
        exp_id = f"exp_{string_hash(p_id + s_id + e_id)}"

        return SuiteIDs(prune=p_id, sft=s_id, eval=e_id, experiment=exp_id)

    @computed_field
    @property
    def dirs(self) -> SuiteDirs:
        return SuiteDirs(
            suite_root=self.output_dir / self.title,
            ids=self.ids,
            timestamp=self.timestamp,
            base_model_id=self.sft.cfg.base_model_id,
            prune_cfg=self.prune.cfg,
            sft_cfg=self.sft.cfg,
        )
