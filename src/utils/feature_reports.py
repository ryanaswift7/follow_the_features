import torch
from pathlib import Path
import json

def write_full_feature_report_csv(
        feat_indices: torch.Tensor,
        feat_values: torch.Tensor,
        total_features: int,
        llm_id: str,
        sae_release: str, 
        sae_id: str,
        hook_name: str,
        report_path: Path) -> dict:
    
    print(f"[PRUNE] Creating Complete CSV Feature Report ({total_features} features)...")

    # Compute normalized values internally
    sum_val = feat_values.sum()
    max_val = feat_values.max()
    mean_val = feat_values.mean()
    std_val = feat_values.std() + 1e-8

    norm_sum_tensor = feat_values / sum_val
    norm_max_tensor = feat_values / max_val
    norm_z_tensor = (feat_values - mean_val) / std_val

    idx_list = feat_indices.tolist()
    raw_list = feat_values.tolist()
    norm_sum_list = norm_sum_tensor.tolist()
    norm_max_list = norm_max_tensor.tolist()
    norm_z_list = norm_z_tensor.tolist()

    # We still build the dict in memory so we can pass it directly 
    # to the Top-K JSON writer without having to re-read the CSV.
    all_info = {
        "cfg": {
            "llm": llm_id,
            "sae_release": sae_release,
            "sae_id": sae_id,
            "hook_name": hook_name,
            "total_features_in_report": total_features,
            "dataset_path": str(report_path.parent),
        },
        "features": {}
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        # Write config as commented headers
        f.write("# CONFIGURATION\n")
        f.write(f"# llm: {llm_id}\n")
        f.write(f"# sae_release: {sae_release}\n")
        f.write(f"# sae_id: {sae_id}\n")
        f.write(f"# hook_name: {hook_name}\n")
        f.write(f"# total_features_in_report: {total_features}\n")
        f.write(f"# dataset_path: {str(report_path.parent)}\n")
        f.write("# ----------------------------------------\n")
        
        # Write tabular headers
        f.write("feature_idx,raw,norm_sum,norm_max,zscore\n")
        
        # Write rows
        for idx, val, snorm, mnorm, zscore in zip(
            idx_list, raw_list, norm_sum_list, norm_max_list, norm_z_list
        ):
            idx_str = str(int(idx))
            str_val = f"{val:.6f}"
            str_snorm = f"{snorm:.6f}"
            str_mnorm = f"{mnorm:.6f}"
            str_zscore = f"{zscore:.6f}"
            
            # Write to CSV
            f.write(f"{idx_str},{str_val},{str_snorm},{str_mnorm},{str_zscore}\n")
            
            # Populate dict for the Top-K step
            all_info["features"][idx_str] = {
                "value": {
                    "raw": str_val,
                    "norm_sum": str_snorm,
                    "norm_max": str_mnorm,
                    "zscore": str_zscore,
                }
            }

    print(f"[PRUNE] Complete Feature Report saved to {str(report_path)}")
    return all_info


def write_topk_feature_report_json(
        full_report_data: dict, 
        k: int, 
        topk_report_path: Path) -> None:
    
    print(f"[PRUNE] Generating Top-{k} Feature Report from full report...")
    
    topk_report_data = {
        "cfg": full_report_data.get("cfg", {}).copy(),
        "features": {}
    }
    
    # Update the config to reflect the sliced size
    topk_report_data["cfg"]["total_features_in_report"] = k

    # Update config to reflect dataset location
    topk_report_data['cfg']['dataset_path'] = str(topk_report_path.parent)
    
    # Extract just the top k features
    count = 0
    for feat_idx, feat_data in full_report_data.get("features", {}).items():
        if count >= k:
            break
        topk_report_data["features"][feat_idx] = feat_data
        count += 1
        
    topk_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(topk_report_path, "w", encoding="utf-8") as f:
        json.dump(topk_report_data, f, indent=4)
        
    print(f"[PRUNE] Top-K Feature Report saved to {str(topk_report_path)}")

