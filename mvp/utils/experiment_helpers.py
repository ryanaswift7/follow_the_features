def aggregate_results(suite_dir, config):
    import json
    import os

    print(f"[AGGREGATE] Aggregating results from suite directory: {suite_dir}")
    results = []
    eval_config_dict = (
        config.eval.dict() if hasattr(config.eval, "dict") else dict(config.eval)
    )
    # Gather baseline results (base_model and prune_model)
    from mvp.utils.experiment_paths import config_hash, get_exp_paths

    eval_id = f"eval_{config_hash(eval_config_dict)}"
    base_model_id = config.base_model_id
    prune_model_id = getattr(config.prune, "model_id", None)
    models_dir = os.path.join(suite_dir, "models")
    # Base model
    base_eval_dir = os.path.join(
        models_dir, "base_model", base_model_id.replace("/", "_"), eval_id
    )
    base_result_path = os.path.join(base_eval_dir, "eval_results.json")
    if os.path.exists(base_result_path):
        with open(base_result_path, "r") as f:
            base_result = json.load(f)
        # If baseline is wrapped as {"eval_scores": {...}}, flatten to match trained format
        if list(base_result.keys()) == ["eval_scores"]:
            base_result = {**base_result["eval_scores"]}
        if "eval_config" not in base_result:
            base_result["eval_config"] = eval_config_dict
        results.append(base_result)
    # Prune model
    if prune_model_id:
        prune_eval_dir = os.path.join(
            models_dir, "prune_model", prune_model_id.replace("/", "_"), eval_id
        )
        prune_result_path = os.path.join(prune_eval_dir, "eval_results.json")
        if os.path.exists(prune_result_path):
            with open(prune_result_path, "r") as f:
                prune_result = json.load(f)
            if list(prune_result.keys()) == ["eval_scores"]:
                prune_result = {**prune_result["eval_scores"]}
            if "eval_config" not in prune_result:
                prune_result["eval_config"] = eval_config_dict
            results.append(prune_result)

    for top_k in config.prune.top_k_features:
        for threshold in config.prune.prune_thresholds:
            # Get canonical experiment paths
            (
                _results_dir,
                _dataset_dir,
                _model_dir,
                _pruned_path,
                _model_subdir,
                eval_subdir,
                _timestamp,
            ) = get_exp_paths(suite_dir, top_k, threshold, config)
            # Look for the canonical eval_results.json in the eval_subdir
            result_path = os.path.join(eval_subdir, "eval_results.json")
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    result_row = json.load(f)
                # Ensure eval_config is present in each experiment result
                if "eval_config" not in result_row:
                    result_row["eval_config"] = eval_config_dict
                results.append(result_row)
    return results


def config_hash(config_dict):
    import hashlib
    import json as _json

    return hashlib.sha256(
        _json.dumps(config_dict, sort_keys=True).encode()
    ).hexdigest()[:10]


def ensure_dir(path):
    import os

    print(f"[DIR] Ensuring directory exists: {path}")
    os.makedirs(path, exist_ok=True)


def get_dataset_size(dataset_path):
    """
    Robustly get the size of a dataset, whether it's a local path or a HuggingFace hub dataset.
    """
    import os

    from datasets import load_dataset, load_from_disk

    if os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        # Try to load from hub (assume 'train' split)
        ds = load_dataset(dataset_path, split="train")
    return len(ds)


def plot_experiment_results(results, output_dir):
    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(results)
    print(df)

    # Assert every row has a numeric score for each eval and the average
    if "eval_scores" in df.columns:
        for idx, row in df.iterrows():
            eval_scores = row["eval_scores"]
            # Check average
            assert "average_eval_score" in eval_scores and isinstance(
                eval_scores["average_eval_score"], (int, float)
            ), f"Row {idx} missing numeric average_eval_score: {eval_scores}"
            # Check all eval datasets
            for ds in eval_scores:
                if ds == "average_eval_score":
                    continue
                assert isinstance(eval_scores[ds], (int, float)), (
                    f"Row {idx} missing numeric score for {ds}: {eval_scores}"
                )

    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Identify all eval datasets present in results (from nested 'eval_scores')
    # Assumes each row has an 'eval_scores' dict with individual scores and 'average_eval_score'
    if "eval_scores" in df.columns:
        # Get all keys except 'average_eval_score' from the first row's eval_scores
        first_scores = df.iloc[0]["eval_scores"]
        eval_datasets = [k for k in first_scores.keys() if k != "average_eval_score"]
    else:
        eval_datasets = []
    # Per-dataset plots

    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Plot average scores across experiments (ROUGE-L vs dataset_size, legend = model names)
    if "eval_scores" in df.columns:
        plt.figure(figsize=(10, 6))
        all_dataset_sizes = [
            row["dataset_size"]
            for _, row in df.iterrows()
            if row["dataset_size"] is not None
            and isinstance(row["dataset_size"], (int, float))
        ]
        x_min, x_max = (
            (min(all_dataset_sizes), max(all_dataset_sizes))
            if all_dataset_sizes
            else (0, 1)
        )

        for model_id, group in df.groupby("model_id"):
            import os as _os

            model_name = _os.path.basename(str(model_id))
            # Identify baseline and trained models using model_type
            baseline_rows = [
                row
                for _, row in group.iterrows()
                if row.get("model_type") != "adapter_model"
            ]
            trained_rows = [
                row
                for _, row in group.iterrows()
                if row.get("model_type") == "adapter_model"
            ]
            # Plot trained models
            if trained_rows:
                ds_vals = [row.get("dataset_size", None) for row in trained_rows]
                score_vals = [
                    row["eval_scores"].get("average_eval_score", None)
                    for row in trained_rows
                ]
                filtered = [
                    (ds, score)
                    for ds, score in zip(ds_vals, score_vals)
                    if ds is not None
                    and isinstance(ds, (int, float))
                    and score is not None
                    and isinstance(score, (int, float))
                ]
                if filtered:
                    ds_vals, score_vals = zip(*filtered)
                    plt.plot(ds_vals, score_vals, marker="o", label=model_name)
            # Plot baseline horizontal line
            if baseline_rows != []:
                for row in baseline_rows:
                    score = row["eval_scores"].get("average_eval_score")

                    assert score is not None, (
                        f"Baseline for model {model_name} missing average_eval_score"
                    )
                    plt.axhline(
                        y=score,  # The Y value
                        color="gray",  # Use 'color' instead of 'colors'
                        linestyle="dashed",
                        label=f"{model_name} (baseline)",
                    )
        plt.xlabel("Dataset Size")
        plt.ylabel("Average ROUGE-L")
        plt.title("Average ROUGE-L vs Dataset Size")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "average.png"))
        plt.close()

    # Identify all eval datasets present in results (from nested 'eval_scores')
    if "eval_scores" in df.columns:
        first_scores = df.iloc[0]["eval_scores"]
        eval_datasets = [k for k in first_scores.keys() if k != "average_eval_score"]
    else:
        eval_datasets = []

    # Per-dataset plots (ROUGE-L vs dataset_size, legend = model names)
    for ds in eval_datasets:
        plt.figure(figsize=(10, 6))
        # Get all dataset sizes for x-range
        all_dataset_sizes = [
            row["dataset_size"]
            for _, row in df.iterrows()
            if row["dataset_size"] is not None
            and isinstance(row["dataset_size"], (int, float))
        ]
        x_min, x_max = (
            (min(all_dataset_sizes), max(all_dataset_sizes))
            if all_dataset_sizes
            else (0, 1)
        )
        for model_id, group in df.groupby("model_id"):
            import os as _os

            model_name = _os.path.basename(str(model_id))
            # Identify baseline and trained models using model_type
            baseline_rows = [
                row
                for _, row in group.iterrows()
                if row.get("model_type") != "adapter_model"
            ]
            trained_rows = [
                row
                for _, row in group.iterrows()
                if row.get("model_type") == "adapter_model"
            ]
            # Plot trained models
            if trained_rows:
                ds_vals = [row.get("dataset_size", None) for row in trained_rows]
                score_vals = [row["eval_scores"].get(ds, None) for row in trained_rows]
                filtered = [
                    (ds_size, score)
                    for ds_size, score in zip(ds_vals, score_vals)
                    if ds_size is not None
                    and isinstance(ds_size, (int, float))
                    and score is not None
                    and isinstance(score, (int, float))
                ]
                if filtered:
                    ds_vals, score_vals = zip(*filtered)
                    plt.plot(ds_vals, score_vals, marker="o", label=model_name)
            # Plot baseline horizontal line(s)
            if baseline_rows != []:
                for row in baseline_rows:
                    score = row["eval_scores"].get(ds)
                    assert score is not None, (
                        f"Baseline for model {model_name} missing score for {ds}"
                    )
                    plt.axhline(
                        y=score,
                        color="gray",
                        linestyle="dashed",
                        label=f"{model_name} (baseline)",
                    )
        plt.xlabel("Dataset Size")
        plt.ylabel("ROUGE-L")
        plt.title(f"ROUGE-L vs Dataset Size for {ds}")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{ds}.png"))
        plt.close()

    # Single markdown table for all results (robust to missing top_k/threshold)
    md_table = "| model | top_k | threshold | dataset_size | "
    md_table += " | ".join(eval_datasets + ["average_eval_score"])
    md_table += (
        " |\n|---|---|---|---|" + "|".join(["---"] * (len(eval_datasets) + 1)) + "|\n"
    )
    for _, row in df.iterrows():
        scores = row["eval_scores"]
        # Robustly get values, use '-' if missing
        import os as _os

        model_name = _os.path.basename(str(row.get("model_id", "-")))
        top_k = row["top_k"] if "top_k" in row and pd.notnull(row["top_k"]) else "-"
        threshold = (
            row["threshold"]
            if "threshold" in row and pd.notnull(row["threshold"])
            else "-"
        )
        dataset_size = (
            row["dataset_size"]
            if "dataset_size" in row and pd.notnull(row["dataset_size"])
            else "-"
        )
        md_table += f"| {model_name} | {top_k} | {threshold} | {dataset_size} "
        for ds in eval_datasets:
            md_table += f"| {scores.get(ds, '-')} "
        md_table += f"| {scores.get('average_eval_score', '-')} |\n"
    with open(os.path.join(tables_dir, "experiment_results.md"), "w") as f:
        f.write(md_table)
