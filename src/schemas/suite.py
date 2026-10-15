import time
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, computed_field, field_validator

from src.schemas.dirs import SuiteDirs
from src.schemas.eval import EvalSetup
from src.schemas.ids import SuiteIDs
from src.schemas.prune import PruneSetup
from src.schemas.sft import SFTSetup
from src.utils.hashing import config_hash, string_hash


class SuiteConfig(BaseModel):
    title: str
    output_dir: Path
    seeds: list[int]
    active_seed: int | None = Field(default=None, exclude=True)
    debug_mode: bool = False
    full_data_reference: bool = False

    prune: PruneSetup
    sft: SFTSetup
    eval: EvalSetup

    timestamp: str = Field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("seeds must contain at least one seed")
        if len(value) != len(set(value)):
            raise ValueError("seeds must be unique")
        return value

    @property
    def seed(self) -> int:
        """Return the active seed for code paths that operate on one run."""
        return self.active_seed if self.active_seed is not None else self.seeds[0]

    def for_seed(self, seed: int) -> "SuiteConfig":
        if seed not in self.seeds:
            raise ValueError(f"Seed {seed} is not configured")
        return self.model_copy(update={"active_seed": seed})

    @property
    def reference_dataset_sizes(self) -> list[int]:
        sizes = list(self.prune.params.dataset_sizes)
        if self.full_data_reference and self.prune.cfg.num_samples not in sizes:
            sizes.append(self.prune.cfg.num_samples)
        return sizes

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SuiteConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))

    @computed_field
    @property
    def ids(self) -> SuiteIDs:
        # Calculate hashes
        p_id = f"prune_{config_hash(self.prune.cfg.model_dump())}_{self.prune.cfg.diversity_tag}"
        s_id = f"sft_{config_hash(self.sft.cfg.model_dump())}"
        e_id = f"eval_{config_hash(self.eval.cfg.model_dump())}"
        exp_id = f"exp_{string_hash(p_id + s_id + e_id + str(self.seeds) + str(self.full_data_reference))}"

        return SuiteIDs(prune=p_id, sft=s_id, eval=e_id, experiment=exp_id)

    @computed_field
    @property
    def dirs(self) -> SuiteDirs:
        return SuiteDirs(
            output_dir=self.output_dir,
            suite_root=self.output_dir / self.title,
            ids=self.ids,
            timestamp=self.timestamp,
            base_model_id=self.sft.cfg.base_model_id,
            prune_cfg=self.prune.cfg,
            sft_cfg=self.sft.cfg,
            active_seed=self.active_seed,
        )
