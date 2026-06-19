import os
import glob
import pickle
import pandas as pd
import numpy as np

input_dir = "RIS_Datasets"
output_dir = "csv dataset"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))
print(f"Found {len(pkl_files)} PKL files. Starting conversion...")

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    n_samples = data['H_real'].shape[0]
    df_list = []
    
    keys_to_extract = ['H_real', 'H_imag', 'x_real', 'x_imag', 'y_real', 'y_imag']
    for key in keys_to_extract:
        if key in data:
            arr = data[key]
            # Flatten all dimensions except the first (samples)
            flat_arr = arr.reshape(n_samples, -1)
            # Create column names
            cols = [f"{key}_{i}" for i in range(flat_arr.shape[1])]
            df_list.append(pd.DataFrame(flat_arr, columns=cols))
            
    if df_list:
        final_df = pd.concat(df_list, axis=1)
        base_name = os.path.basename(pkl_file).replace(".pkl", ".csv")
        out_path = os.path.join(output_dir, base_name)
        final_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} with shape {final_df.shape}")

print("All files have been successfully converted to CSV format.")
