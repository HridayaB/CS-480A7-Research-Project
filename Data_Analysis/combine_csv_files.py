import pandas as pd
import os

# Combining the csv files into one file
def combine_csv_files(input_folder, output_file):
    all_files = []
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_folder)
                os_name = relative_path.split(os.sep)[0]

                try:
                    df = pd.read_csv(file_path)
                    df['os'] = os_name
                    all_files.append(df)
                    print(f"Loaded {file_path} (OS: {os_name}, Records: {len(df)})")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    if not all_files:
        print("No CSV files found.")
        return
    
    combined_df = pd.concat(all_files, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

    return combined_df


if __name__ == "__main__":
    input_folder = "data/C data"  
    output_file = "data/combined_C_data.csv" 
    combine_csv_files(input_folder, output_file)