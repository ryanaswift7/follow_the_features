import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. SET UP COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser(description="Plot Top-K Features vs Average Judge Score.")
    parser.add_argument("-f", "--file", required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output image.")
    parser.add_argument("-d", "--dataset-size", type=int, default=10000, help="Target dataset size to plot (default: 10000).")
    args = parser.parse_args()

    # 2. LOAD AND VALIDATE DATA
    if not os.path.exists(args.file):
        print(f"Error: The file '{args.file}' does not exist.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

    required_cols = ['model_type', 'topk_features', 'dataset_size', 'average_judge_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the dataset: {missing_cols}")
        print(f"Columns found: {list(df.columns)}")
        sys.exit(1)

    # Filter the dataframe to only include the targeted dataset size
    filtered_df = df[df['dataset_size'] == args.dataset_size]
    
    if filtered_df.empty:
        print(f"Error: No data found for dataset_size = {args.dataset_size}.")
        unique_sizes = sorted(df['dataset_size'].unique().tolist())
        print(f"Available dataset sizes in your file: {unique_sizes}")
        sys.exit(1)

    # 3. SET UP VISUAL STYLE
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 4. GENERATE THE PLOT
    sns.scatterplot(
        data=filtered_df,
        x='topk_features',
        y='average_judge_score',
        hue='model_type',
        s=120,
        alpha=0.8,
        palette='Set2',
        legend=False
    )

    # Add the general trendline (ci=95 is the default, so removing ci=None brings back the shading)
    sns.regplot(
        data=filtered_df, 
        x='topk_features', 
        y='average_judge_score', 
        scatter=False,
        color='gray', 
        line_kws={'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.6}
    )

    # Add (x, y) coordinate labels to each data point with a larger font
    for idx, row in filtered_df.iterrows():
        label_text = f"({row['topk_features']}, {row['average_judge_score']:.2f})"
        
        plt.annotate(
            label_text,                                         
            (row['topk_features'], row['average_judge_score']), 
            textcoords="offset points",                         
            xytext=(12, 10),                                    # Increased offset for the larger text
            ha='left',                                          
            fontsize=16,                                        # <-- BIGGER POINT LABELS
            alpha=0.9,                                          
            weight='bold'                                       
        )

    # 5. CUSTOMIZE LABELS
    # Title has been removed completely
    
    # Made axis labels much bigger and bold for readability
    plt.xlabel('Top-K Features', fontsize=18, fontweight='bold', labelpad=10)
    plt.ylabel('Average Judge Score', fontsize=18, fontweight='bold', labelpad=10)
    
    # Also increased the size of the tick marks on the axes to match the new proportions
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()

    # 6. SAVE
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to '{args.output}' for dataset size {args.dataset_size}")

if __name__ == "__main__":
    main()