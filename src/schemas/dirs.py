from pathlib import Path

from pydantic import BaseModel, computed_field

from src.schemas.ids import SuiteIDs
from src.schemas.prune import PruneConfig
from src.schemas.sft import SFTConfig


class SuiteDirs(BaseModel):
    output_dir: Path
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
        return self._ensure_dir(self.output_dir / "models")

    @computed_field
    @property
    def datasets(self) -> Path:
        return self._ensure_dir(self.output_dir / "datasets" / self.ids.prune)

    @computed_field
    @property
    def _results_root(self) -> Path:
        return self._ensure_dir(
            self.suite_root / "results" / self.ids.experiment / self.timestamp
        )

    @computed_field
    @property
    def results(self) -> Path:
        return self._ensure_dir(self._results_root / "main")

    @computed_field
    @property
    def null_dist_results(self) -> Path:
        return self._ensure_dir(self._results_root / "null_dist")

    @computed_field
    @property
    def base_model_root(self) -> Path:
        name = Path(self.sft_cfg.base_model_id).name
        return self._ensure_dir(self.models / "base_models" / name)

    @computed_field
    @property
    def base_model_eval(self) -> Path:
        return self._ensure_dir(self.base_model_root / self.ids.eval)

    @computed_field
    @property
    def reference_model_root(self) -> Path:
        name = Path(self.sft_cfg.base_model_id).name
        return self._ensure_dir(self.models / "reference_models" / name)

    @computed_field
    @property
    def reference_model_eval(self) -> Path:
        return self._ensure_dir(self.reference_model_root / self.ids.eval)

    def get_ref_model_root_by_seed(self, seed: int):
        return self._ensure_dir(self.reference_model_root / f"seed_{seed}")

    def get_ref_model_eval_by_seed(self, seed: int):
        return self._ensure_dir(self.get_ref_model_root_by_seed(seed) / self.ids.eval)

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
