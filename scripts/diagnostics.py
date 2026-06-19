import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# We import the exact model definition from the trainer script to avoid mismatch
from train_baseline_pinn import RISBaselineModel

def check_leakage(df_train, df_val):
    """Check if same parameters exist in both train and val."""
    configs_train = set(zip(df_train['N_Tx'], df_train['N_Rx'], df_train['N_RIS']))
    configs_val = set(zip(df_val['N_Tx'], df_val['N_Rx'], df_val['N_RIS']))
    overlap = configs_train.intersection(configs_val)
    print("\n--- 4. Train/Validation Split Analysis ---")
    print(f"Total training samples: {len(df_train)}")
    print(f"Total validation samples: {len(df_val)}")
    
    # Are configurations safely decoupled?
    print(f"Unique (N_Tx, N_Rx, N_RIS) configs in Train: {len(configs_train)}")
    print(f"Unique (N_Tx, N_Rx, N_RIS) configs in Val: {len(configs_val)}")
    print(f"Configs present in BOTH Train and Val sets (Overlap count): {len(overlap)}")
    if len(overlap) > 0:
        print("Warning: The same RIS antenna configurations appear in both sets.")
        print("This may not strictly be 'leakage' if SNR or other params vary, but it depends on the task.")

def plot_distributions(Y_raw, preds, save_dir):
    """Plot distribution of true targets and predictions to detect collapse."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Target distribution
    plt.subplot(1, 2, 1)
    plt.hist(Y_raw[:, 0], bins=50, alpha=0.7, label='y_real_0')
    if Y_raw.shape[1] > 1:
        plt.hist(Y_raw[:, 1], bins=50, alpha=0.7, label='y_imag_0')
    plt.title("Target Ground Truth Distribution")
    plt.legend()

    # Prediction distribution
    plt.subplot(1, 2, 2)
    plt.hist(preds[:, 0], bins=50, alpha=0.7, label='Pred y_real_0')
    if preds.shape[1] > 1:
        plt.hist(preds[:, 1], bins=50, alpha=0.7, label='Pred y_imag_0')
    plt.title("Model Prediction Distribution")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distribution_diagnostics.png"))
    plt.close()
    print(f"Saved distribution diagnostic plots to {save_dir}/distribution_diagnostics.png")

def main():
    dataset_path = "datasets/small_dataset.csv"
    model_path = "models/model_baseline_best.pth"
    save_dir = "plots"
    
    print("Loading dataset from", dataset_path)
    df = pd.read_csv(dataset_path)
    
    # Dataset Split identical to train_baseline_pinn.py
    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    
    check_leakage(df_train, df_val)
    
    print("\n--- 1. Parsing Verification ---")
    array_cols = [
        'phase_shift_real', 'phase_shift_imag', 
        'H_real', 'H_imag', 
        'x_real', 'x_imag', 
        'y_real', 'y_imag'
    ]
    
    df_subset = df_train.head(5).copy()
    for col in array_cols:
        if col in df_subset.columns and isinstance(df_subset[col].iloc[0], str):
            df_subset[col] = df_subset[col].apply(ast.literal_eval)
            
    print("Shapes of arrays for the first row in train set:")
    for col in array_cols:
        arr = np.array(df_subset[col].iloc[0])
        print(f"  {col}: {arr.shape} (Flattened length: {np.prod(arr.shape)})")
        
    print("\n--- 2. Feature Normalization Verification ---")
    scaler = StandardScaler()
    
    def get_arrays(df_target):
        df_curr = df_target.copy()
        for col in array_cols:
             if col in df_curr.columns and isinstance(df_curr[col].iloc[0], str):
                 df_curr[col] = df_curr[col].apply(ast.literal_eval)
        
        scalar_cols = ['frequency', 'wavelength', 'N_Tx', 'N_Rx', 'N_RIS', 'dx', 'dy', 'SNR_dB']
        X_scalars = df_curr[scalar_cols].values
        
        def extract_flattened(col_name):
            return np.array([np.array(row).flatten() for row in df_curr[col_name]])
            
        X_parts = [X_scalars]
        input_array_names = ['phase_shift_real', 'phase_shift_imag', 'H_real', 'H_imag', 'x_real', 'x_imag']
        for col in input_array_names:
            X_parts.append(extract_flattened(col))
            
        X_raw = np.hstack(X_parts)
        Y_raw = np.hstack([extract_flattened('y_real'), extract_flattened('y_imag')])
        return X_raw, Y_raw
    
    print("Extracting train set (this takes a moment)...")
    X_train_raw, Y_train_raw = get_arrays(df_train)
    X_val_raw, Y_val_raw = get_arrays(df_val)
    
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    
    print("Input scaled features Train mean:", np.mean(X_train_scaled), "var:", np.var(X_train_scaled))
    print("TARGET variables Train mean:", np.mean(Y_train_raw), "var:", np.var(Y_train_raw))
    print("TARGET max:", np.max(Y_train_raw), "min:", np.min(Y_train_raw))
    
    if np.var(Y_train_raw) < 1e-4 or np.var(Y_train_raw) > 1e4:
         print(">>> WARNING: Target variance is extreme! The network is NOT normalizing targets in train_baseline_pinn.py. This can cause severe underfitting or collapse! <<<")
         
    print("\n--- 3. Tensor Dimensions Verification ---")
    input_dim = X_train_raw.shape[1]
    output_dim = Y_train_raw.shape[1]
    print(f"Network Input Dimension: {input_dim}")
    print(f"Network Output Dimension: {output_dim}")
    
    print("\n--- 5/6. Model Diagnostics & Collapsed Predictions ---")
    if not os.path.exists(model_path):
        print(f"Model weight {model_path} not found. Cannot test predictions.")
        return
        
    device = torch.device('cpu')
    model = RISBaselineModel(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        preds = model(X_val_tensor).numpy()
        
    pred_var = np.var(preds, axis=0)
    print("Prediction Variance across validation samples for each output dimension:")
    print(pred_var)
    if np.mean(pred_var) < 1e-5:
        print(">>> WARNING: Predictions have nearly zero variance. THE MODEL HAS COLLAPSED TO A CONSTANT! <<<")
        
    plot_distributions(Y_val_raw, preds, save_dir)
    print("\nDiagnostics run complete.")

if __name__ == "__main__":
    main()
