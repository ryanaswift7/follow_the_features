from pathlib import Path
from typing import Literal

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

    # --- INTERNAL HELPERS ---

    def ensure_dir(self, path: Path) -> Path:
        """Creates the directory if it doesn't exist and returns the path."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_generation_cache_subdir(max_new_tokens: int) -> Path:
        return Path("generation_cache") / f"mnt-{max_new_tokens}"

    # --- COMPUTED FIELDS (No arguments, serialized in output) ---

    @computed_field
    @property
    def models(self) -> Path:
        return self.ensure_dir(self.output_dir / "models")

    @computed_field
    @property
    def datasets(self) -> Path:
        return self.ensure_dir(self.output_dir / "datasets" / self.ids.prune)

    @property # Just a property, not a computed field since it's private
    def _results_root(self) -> Path:
        return self.ensure_dir(
            self.suite_root / "results" / self.ids.experiment # / self.timestamp
        )

    @computed_field
    @property
    def main_results(self) -> Path:
        return self.ensure_dir(self._results_root / "main")
    
    @computed_field
    @property
    def sota_results(self) -> Path:
        return self.ensure_dir(self._results_root / "sota")


    @computed_field
    @property
    def base_model_root(self) -> Path:
        name = Path(self.sft_cfg.base_model_id).name
        return self.ensure_dir(self.models / "base_models" / name)

    @computed_field
    @property
    def base_model_eval(self) -> Path:
        return self.ensure_dir(self.base_model_root / self.ids.eval)

    @computed_field
    @property
    def reference_model_root(self) -> Path:
        name = Path(self.sft_cfg.base_model_id).name
        return self.ensure_dir(self.models / "reference_models" / name)

    @computed_field
    @property
    def reference_model_summary_root(self) -> Path:
        return self.ensure_dir(self.reference_model_root / "summary")

    @computed_field
    @property
    def reference_model_summary_eval(self) -> Path:
        # Aggregate reference-model summary results live in a nested eval dir.
        return self.ensure_dir(self.reference_model_summary_root / self.ids.eval)
    
    # --- STANDARD METHODS (Take arguments, NOT properties) ---
    
    def sota_model_root(self, method: Literal['grape', 's2l']) -> Path:
        # name = Path(self.sft_cfg.base_model_id).name
        return self.ensure_dir(self.models / "sota" / method)

    def sota_model_eval(self, method: Literal['grape', 's2l']) -> Path:
        return self.ensure_dir(self.sota_model_root(method=method) / self.ids.eval)
    
    def get_sota_model_gen_cache(self, method: Literal["grape", "s2l"], max_new_tokens: int) -> Path:
        return self.ensure_dir(
            self.sota_model_root(method=method)
            / self.get_generation_cache_subdir(max_new_tokens=max_new_tokens)
        )


    def get_base_model_gen_cache(self, max_new_tokens: int) -> Path:
        return self.ensure_dir(
            self.base_model_root
            / self.get_generation_cache_subdir(max_new_tokens=max_new_tokens)
        )

    def get_ref_model_root_by_seed(self, seed: int) -> Path:
        return self.ensure_dir(self.reference_model_root / f"seed_{seed}")

    def get_ref_model_eval_by_seed(self, seed: int) -> Path:
        return self.ensure_dir(self.get_ref_model_root_by_seed(seed) / self.ids.eval)
    
    def get_ref_model_gen_cache_by_seed(self, seed: int, max_new_tokens: int) -> Path:
        return self.ensure_dir(
            self.get_ref_model_root_by_seed(seed)
            / self.get_generation_cache_subdir(max_new_tokens=max_new_tokens)
        )

    def get_base_model_alpaca_eval_model_configs_dir(self) -> Path:
        return self.ensure_dir(self.base_model_root / "alpaca_eval_model_configs")

    def get_ref_model_alpaca_eval_model_configs_dir_by_seed(self, seed: int) -> Path:
        return self.ensure_dir(
            self.get_ref_model_root_by_seed(seed)
            / "alpaca_eval_model_configs"
        )

    def get_trained_model_alpaca_eval_model_configs_dir(self, dataset_size: int, topk: int) -> Path:
        return self.ensure_dir(
            self.get_trained_model_root(dataset_size, topk)
            / "alpaca_eval_model_configs"
        )

    def get_dataset_path(self, dataset_size: int, topk: int) -> Path:
        """
        Returns path to the dataset file/folder.
        Note: We create this specific path for use by 
        the feature reports prior to the auto-creation by HF datasets lib.
        """
        name = Path(self.prune_cfg.dataset_id).name
        diversity_tag = getattr(self.prune_cfg, "diversity_tag", "divoff")
        return self.ensure_dir(self.datasets / f"{name}_{diversity_tag}_ds{dataset_size}_topk{topk}.hf")

    def get_trained_model_root(self, dataset_size: int, topk: int) -> Path:
        name = Path(self.sft_cfg.base_model_id).name
        diversity_tag = getattr(self.prune_cfg, "diversity_tag", "divoff")
        return self.ensure_dir(
            self.models
            / "trained"
            / self.ids.prune
            / self.ids.sft
            / f"ds{dataset_size}_topk{topk}_{diversity_tag}_{name}"
        )

    def get_trained_model_eval(self, dataset_size: int, topk: int) -> Path:
        return self.ensure_dir(
            self.get_trained_model_root(dataset_size, topk)
            / self.ids.eval
            # / f"top{topk}"
        )
    
    def get_trained_model_gen_cache(self, dataset_size: int, topk: int, max_new_tokens: int) -> Path:
        return self.ensure_dir(
            self.get_trained_model_root(dataset_size, topk)
            / self.get_generation_cache_subdir(max_new_tokens=max_new_tokens)
        )