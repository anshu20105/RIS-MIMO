"""
Input Sensitivity Analysis
==========================
Evaluates the impact of input parameter changes on PINN model predictions.
Varies: SNR, RIS Size, N_Tx, N_Rx, Frequency
Measures: SE, BER, y_power

Generates Response curves, Tornado charts, and Sensitivity rankings.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import ast

# Import model architecture
from train_final_model import MultiTaskDigitalTwin, MultiTaskRISDataset, unscale_tensors

# ==========================================
# Configuration
# ==========================================
DATA_FILE = "datasets/digital_twin_dataset.csv"
MODEL_PATH = "checkpoints/final_pinn_best.pth"
PLOTS_DIR = "plots/sensitivity"
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set plotting styles
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
COLORS = sns.color_palette("husl", 8)

# Parameter space to sweep
PARAM_SPACE = {
    'frequency': [3.5e9, 6e9, 26e9],
    'N_RIS': [8, 16, 32, 64, 128],
    'N_Tx': [2, 4, 8],
    'N_Rx': [2, 4, 8],
    'SNR_dB': [-10, 0, 10, 20]
}

# Indices of scalar features in the flattened X array
FEAT_IDX = {
    'frequency': 0,
    'N_Tx': 1,
    'N_Rx': 2,
    'N_RIS': 3,
    'dx': 4,
    'dy': 5,
    'SNR_dB': 6
}

# ==========================================
# Loader & Scaling Hooks
# ==========================================
def load_and_prepare_scalers():
    """Loads a subset of data to fit exactly the same scalers used in training."""
    print("Fitting exact same scalers using original data rules...")
    df = pd.read_csv(DATA_FILE)
    
    # We only need enough data to fit the scalers properly (the same training split behavior).
    # Replicate train_final_model logic
    indices = np.arange(len(df))
    snr_bins = pd.qcut(df['SNR_dB'], q=4, labels=False, duplicates='drop')
    from sklearn.model_selection import train_test_split
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=snr_bins, random_state=42)
    
    # Extract Raw Inputs for training split
    X_train_list = []
    for idx in train_idx:
        row = df.iloc[idx]
        base_features = [
            row['frequency'], row['N_Tx'], row['N_Rx'], row['N_RIS'],
            row['dx'], row['dy'], row['SNR_dB']
        ]
        p_real = ast.literal_eval(row['phase_shift_real'])
        p_imag = ast.literal_eval(row['phase_shift_imag'])
        pad_len = 128 - len(p_real)
        p_real_padded = np.pad(p_real, (0, pad_len), 'constant')
        p_imag_padded = np.pad(p_imag, (0, pad_len), 'constant')
        X_train_list.append(np.concatenate((base_features, p_real_padded, p_imag_padded)))
        
    X_train = np.array(X_train_list, dtype=np.float32)
    
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    
    yp_train = df.iloc[train_idx]['y_power'].values.astype(np.float32).reshape(-1, 1)
    se_train = df.iloc[train_idx]['SE'].values.astype(np.float32).reshape(-1, 1)
    ber_train = df.iloc[train_idx]['BER'].values.astype(np.float32).reshape(-1, 1)
    
    yp_train = np.log10(yp_train + 1e-10)
    ber_train = np.log10(ber_train + 1e-12)
    
    scaler_yp = StandardScaler().fit(yp_train)
    scaler_se = StandardScaler().fit(se_train)
    scaler_ber = StandardScaler().fit(ber_train)
    
    return scaler_X, scaler_yp, scaler_se, scaler_ber, df


# ==========================================
# Main Evaluation Loop
# ==========================================
def main():
    print("=" * 60)
    print("RUNNING PINN SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    scaler_X, scaler_yp, scaler_se, scaler_ber, df = load_and_prepare_scalers()
    
    input_dim = scaler_X.mean_.shape[0]
    model = MultiTaskDigitalTwin(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # Pick a baseline "average" vector from the dataset to perturb
    # e.g., Freq=6GHz, N_Tx=4, N_Rx=4, N_RIS=32, SNR=0
    mask = (df['frequency'] == 6e9) & (df['N_Tx'] == 4) & (df['N_Rx'] == 4) & (df['N_RIS'] == 32) & (df['SNR_dB'] == 0)
    baseline_row = df[mask].iloc[0]
    
    base_features = [
        baseline_row['frequency'], baseline_row['N_Tx'], baseline_row['N_Rx'], baseline_row['N_RIS'],
        baseline_row['dx'], baseline_row['dy'], baseline_row['SNR_dB']
    ]
    p_real = ast.literal_eval(baseline_row['phase_shift_real'])
    p_imag = ast.literal_eval(baseline_row['phase_shift_imag'])
    pad_len = 128 - len(p_real)
    p_real_padded = np.pad(p_real, (0, pad_len), 'constant')
    p_imag_padded = np.pad(p_imag, (0, pad_len), 'constant')
    
    X_base = np.concatenate((base_features, p_real_padded, p_imag_padded)).astype(np.float32)
    
    # -----------------------------------------------------
    # 1. Parameter Sweep (Response Curves)
    # -----------------------------------------------------
    print("\nRunning Parameter Sweeps...")
    sweep_results = {}
    
    # Dictionary to store baseline prediction reference point
    base_tensor = torch.tensor(scaler_X.transform(X_base.reshape(1, -1)), device=DEVICE)
    with torch.no_grad():
        yp, se, ber = model(base_tensor)
        base_yp = scaler_yp.inverse_transform(yp.cpu().numpy())[0][0]
        base_se = scaler_se.inverse_transform(se.cpu().numpy())[0][0]
        base_ber = scaler_ber.inverse_transform(ber.cpu().numpy())[0][0]
        
    for param_name, values in PARAM_SPACE.items():
        sweep_data = {'val': [], 'yp': [], 'se': [], 'ber': []}
        idx = FEAT_IDX[param_name]
        
        for val in values:
            X_mod = X_base.copy()
            X_mod[idx] = val
            X_mod_scaled = scaler_X.transform(X_mod.reshape(1, -1))
            X_tensor = torch.tensor(X_mod_scaled, device=DEVICE)
            
            with torch.no_grad():
                yp, se, ber = model(X_tensor)
                yp_unscaled = scaler_yp.inverse_transform(yp.cpu().numpy())[0][0]
                se_unscaled = scaler_se.inverse_transform(se.cpu().numpy())[0][0]
                ber_unscaled = scaler_ber.inverse_transform(ber.cpu().numpy())[0][0]
                
            sweep_data['val'].append(val)
            sweep_data['yp'].append(yp_unscaled)
            sweep_data['se'].append(se_unscaled)
            sweep_data['ber'].append(ber_unscaled)
            
        sweep_results[param_name] = sweep_data

    # Plot Response Curves
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    metrics = [('se', 'Spectral Efficiency', 'blue'),
               ('ber', 'log₁₀(BER)', 'red'),
               ('yp', 'log₁₀(y_power)', 'green')]
    
    for row, (met, y_label, color) in enumerate(metrics):
        for col, (param_name, data) in enumerate(sweep_results.items()):
            ax = axes[row, col]
            x_vals = data['val']
            y_vals = data[met]
            
            ax.plot(x_vals, y_vals, marker='o', linewidth=2, color=color)
            
            if row == 0:
                ax.set_title(f'Varying {param_name}', fontweight='bold')
            if col == 0:
                ax.set_ylabel(y_label, fontweight='bold')
                
            ax.set_xlabel(param_name)
            
            if param_name == 'frequency':
                ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "response_curves.png"))
    plt.close()
    print("  -> Saved response_curves.png")

    # -----------------------------------------------------
    # 2. Tornado Charts (Sensitivity Analysis)
    # -----------------------------------------------------
    # We will use standardized bounds for tornado chart (min vs max in sweep)
    # to evaluate absolute delta impact on predicted quantities.
    tornado_data = {
        'SE': [],
        'BER': [],
        'y_power': []
    }
    
    for param_name, data in sweep_results.items():
        # Get min and max outputs for this parameter sweep
        delta_se = max(data['se']) - min(data['se'])
        delta_ber = max(data['ber']) - min(data['ber'])
        delta_yp = max(data['yp']) - min(data['yp'])
        
        tornado_data['SE'].append((param_name, delta_se))
        tornado_data['BER'].append((param_name, delta_ber))
        tornado_data['y_power'].append((param_name, delta_yp))
        
    # Sort for Tornado plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (met_key, color, title) in enumerate([('SE', 'blue', 'SE Sensitivity (Max $\Delta$)'),
                                                 ('BER', 'red', 'log(BER) Sensitivity (Max $\Delta$)'),
                                                 ('y_power', 'green', 'log(y_power) Sensitivity (Max $\Delta$)')]):
        ax = axes[i]
        # Sort data by delta descending
        sorted_data = sorted(tornado_data[met_key], key=lambda x: x[1])
        names = [x[0] for x in sorted_data]
        deltas = [x[1] for x in sorted_data]
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, deltas, color=color, alpha=0.7, edgecolor='black', height=0.6)
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontweight='bold')
        ax.set_xlabel(f'Absolute $\Delta$ {met_key}')
        ax.set_title(title, fontweight='bold', pad=15)
        
        # Expand x-limits to fit text
        max_val = max(deltas)
        ax.set_xlim(0, max_val * 1.25)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tornado_charts.png"))
    plt.close()
    print("  -> Saved tornado_charts.png")

    # -----------------------------------------------------
    # 3. Sensitivity Rankings Table
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print("SENSITIVITY RANKINGS (Highest Impact -> Lowest)")
    print("=" * 60)
    
    for met_key in ['SE', 'BER', 'y_power']:
        sorted_data = sorted(tornado_data[met_key], key=lambda x: x[1], reverse=True)
        print(f"\nTarget: {met_key}")
        for rank, (name, delta) in enumerate(sorted_data):
            print(f"  {rank+1}. {name:<12} | Max Delta: {delta:.4f}")

    print("\nSensitivity Analysis Complete.")

if __name__ == "__main__":
    main()
