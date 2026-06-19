import os
import pandas as pd
import glob

def combine_csvs():
    directory = '/home/anshu/RIS_Project/comprehensive datasets'
    output_file = '/home/anshu/RIS_Project/comprehensive datasets/combined_dataset.csv'
    
    # Get all csv files except the output file if it exists
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    all_files = [f for f in all_files if f != output_file]
    
    print(f"Found {len(all_files)} CSV files to combine.")
    
    print(f"Saving combined dataset to {output_file} iteratively...")
    
    first_file = True
    # Read each file and stream to output CSV to avoid memory issues
    for file in all_files:
        df = pd.read_csv(file)
        if first_file:
            df.to_csv(output_file, index=False)
            first_file = False
        else:
            df.to_csv(output_file, mode='a', header=False, index=False)
            
    print("Done!")

if __name__ == '__main__':
    combine_csvs()
