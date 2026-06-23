import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations

def compare_top_k_spearman_large_scale(csv_paths, top_k_values, labels=None, feature_col='feature_idx'):
    """
    Computes Spearman rank correlation optimized for large datasets (~1M+ rows),
    using custom labels for cleaner output.
    """
    # 1. Validate labels
    if labels is None:
        labels = csv_paths
    elif len(labels) != len(csv_paths):
        raise ValueError("The number of labels must exactly match the number of csv_paths.")

    dfs = {}
    
    # 2. Memory-Optimized Loading
    for path, label in zip(csv_paths, labels):
        try:
            # ONLY load the feature column to save massive amounts of RAM
            df = pd.read_csv(path, usecols=[feature_col], comment='#')            
            # Create the rank column and downcast to int32 to save memory
            df['rank'] = np.arange(1, len(df) + 1, dtype=np.int32)
            
            # Store using the clean label instead of the long file path
            dfs[label] = df
            print(f"Successfully loaded {len(df):,} rows from '{label}'")
        except ValueError:
            print(f"Error: Column '{feature_col}' not found in {path}. Skipping.")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    results = []
    
    # 3. Iterate over pairs of labels
    for label1, label2 in combinations(dfs.keys(), 2):
        df1 = dfs[label1]
        df2 = dfs[label2]
        
        for k in top_k_values:
            if k == 'all':
                top_df1 = df1
                top_df2 = df2
                penalty_rank = max(len(df1), len(df2)) + 1
            else:
                top_df1 = df1.head(k).copy()
                top_df2 = df2.head(k).copy()
                penalty_rank = k + 1
            
            # 4. Fast outer merge
            merged = pd.merge(
                top_df1, 
                top_df2, 
                on=feature_col, 
                how='outer', 
                suffixes=('_1', '_2')
            )
            
            # Fill missing items with the penalty rank, keeping memory footprint low
            merged['rank_1'] = merged['rank_1'].fillna(penalty_rank).astype(np.int32)
            merged['rank_2'] = merged['rank_2'].fillna(penalty_rank).astype(np.int32)
            
            union_size = len(merged)
            
            # 5. Compute Spearman correlation
            if union_size > 1:
                rho, p_val = spearmanr(merged['rank_1'], merged['rank_2'])
            else:
                rho, p_val = float('nan'), float('nan')
                
            results.append({
                'dataset_1': label1,
                'dataset_2': label2,
                'k': k,
                'spearman_rho': rho,
                'p_value': p_val,
                'union_size': union_size
            })
            
    return pd.DataFrame(results)

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    my_csv_files = [
        "/research-storage/lab/ryanswif/feature_data_pruning/experiments/datasets/prune_04320eb0bd/full_feature_report.csv",
        "/research-storage/lab/ryanswif/feature_data_pruning/experiments/datasets/prune_2586947ca3/full_feature_report.csv",
        "/research-storage/lab/ryanswif/feature_data_pruning/experiments/datasets/prune_10b6160cac/full_feature_report.csv",
    ]
    
    # Clean, short labels for the output
    my_labels = ['UltraChat 200K', 'Dolci 100K', 'Alpaca']
    
    my_k_values = [5, 10, 1000, 'all']
    
    results_df = compare_top_k_spearman_large_scale(
        csv_paths=my_csv_files, 
        top_k_values=my_k_values, 
        labels=my_labels,              # Pass the labels here
        feature_col='feature_idx'
    )
    
    if not results_df.empty:
        fn = "feature_ranking_correlation_results"
        print("\nResults:")
        print(results_df.to_string(index=False))
        results_df.to_csv(f"{fn}.csv")
        results_df.to_markdown(f"{fn}.md")