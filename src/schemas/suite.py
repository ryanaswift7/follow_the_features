import time
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, computed_field

from src.schemas.dirs import SuiteDirs
from src.schemas.eval import EvalSetup
from src.schemas.ids import SuiteIDs
from src.schemas.prune import PruneSetup
from src.schemas.sft import SFTSetup
from src.utils.hashing import config_hash, string_hash


class SuiteConfig(BaseModel):
    title: str
    output_dir: Path
    seed: int
    debug_mode: bool = False

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
        exp_id = f"exp_{string_hash(p_id + s_id + e_id + str(self.seed))}"

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
        )
