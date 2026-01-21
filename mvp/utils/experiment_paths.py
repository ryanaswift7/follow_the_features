import os
import time

from mvp.utils.experiment_helpers import config_hash


def get_exp_paths(suite_dir, top_k, threshold, config):
    prune_config = config.prune
    sft_config = config.sft
    eval_config = config.eval
    prune_id = f"prune_{config_hash(prune_config.dict())}"
    sft_id = f"sft_{config_hash(sft_config.dict())}"
    eval_id = f"eval_{config_hash(eval_config.dict())}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_hash = config_hash(
        config.dict() if hasattr(config, "dict") else dict(config)
    )
    results_dir = os.path.join(suite_dir, "results", f"{experiment_hash}_{timestamp}")
    dataset_dir = os.path.join(suite_dir, "datasets")
    model_dir = os.path.join(suite_dir, "models")
    # Model output structure: models/prune_<id>_sft_<id>/<model_dir>/eval_<id>/
    model_group_dir = os.path.join(model_dir, f"{prune_id}_{sft_id}")
    model_name = config.base_model_id.split("/")[-1]
    model_subdir = os.path.join(
        model_group_dir, f"{model_name}_topk{top_k}_thresh{threshold}"
    )
    eval_subdir = os.path.join(model_subdir, eval_id)
    # New dataset structure: datasets/prune_{prune_id}/<dataset_id>_topk{top_k}_thresh{threshold}.hf
    dataset_prune_dir = os.path.join(
        dataset_dir, f"prune_{config_hash(prune_config.dict())}"
    )
    dataset_id = getattr(config, "prune").dataset_id.split("/")[-1]
    pruned_path = os.path.join(
        dataset_prune_dir, f"{dataset_id}_topk{top_k}_thresh{threshold}.hf"
    )
    return (
        results_dir,
        dataset_dir,
        model_dir,
        pruned_path,
        model_subdir,
        eval_subdir,
        timestamp,
    )


def check_outputs(dataset_path, model_dir, exp_dir):
    import glob

    print(f"[CHECK] Checking outputs in {exp_dir}")
    dataset_exists = os.path.exists(dataset_path)
    model_exists = os.path.exists(model_dir) and any(os.scandir(model_dir))
    result_files = sorted(
        glob.glob(os.path.join(exp_dir, "experiment_result_*.json")), reverse=True
    )
    result_path = result_files[0] if result_files else None
    results_exist = result_path is not None
    return dataset_exists, model_exists, results_exist, result_path
