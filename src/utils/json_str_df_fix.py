import pandas as pd
import json
import os
import argparse

def fix_and_export_data(input_csv_path):
    # Verify the file exists
    if not os.path.exists(input_csv_path):
        print(f"Error: The file '{input_csv_path}' was not found.")
        return

    print(f"Reading and cleaning '{input_csv_path}'...")
    
    # Determine the base directory and filename
    base_dir = os.path.dirname(input_csv_path) or '.'
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    
    # 1. Load the messy CSV (assuming an unnamed index column and a column named '0')
    try:
        df = pd.read_csv(input_csv_path, index_col=0)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # 2. Parse the stringified JSON in column '0'
    try:
        df['0'] = df['0'].apply(json.loads)
    except Exception as e:
        print(f"Error parsing JSON data: {e}")
        print("Make sure the target column contains valid JSON strings.")
        return
    
    # 3. Flatten the dictionary into separate columns
    clean_df = pd.json_normalize(df['0'])
    
    # 4. Define output paths
    csv_out = os.path.join(base_dir, f"{base_name}_clean.csv")
    md_out = os.path.join(base_dir, f"{base_name}_clean.md")
    parquet_out = os.path.join(base_dir, f"{base_name}_clean.parquet")
    
    # 5. Save the fixed dataframe
    print("Saving files...")
    clean_df.to_csv(csv_out, index=False)
    clean_df.to_markdown(md_out, index=False)
    clean_df.to_parquet(parquet_out, index=False)
    
    print("\n✅ Success! Files cleaned and saved:")
    print(f" 📄 {csv_out}")
    print(f" 📝 {md_out}")
    print(f" 📦 {parquet_out}")

if __name__ == "__main__":
    # Set up CLI argument parsing
    parser = argparse.ArgumentParser(description="Clean a messy CSV containing stringified JSON and export to CSV, Markdown, and Parquet.")
    parser.add_argument("input_path", help="Path to the messy CSV file you want to fix.")
    
    # Parse arguments from the command line
    args = parser.parse_args()
    
    # Run the function
    fix_and_export_data(args.input_path)