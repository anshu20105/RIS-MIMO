"""
Validation Dataset Generator to verify derived multi-task labels
(effective SNR, channel norm, y_power, SE, BER) before full generation.
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import scipy.special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Using the core functions from the main generator
import ris_dataset_generator as rgen

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_FILE = "datasets/digital_twin_dataset.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_validation():
    print(f"Loading dataset from {DATA_FILE} for validation...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples.")
    # Map the SNR_dB config for plots to match standard naming if needed
    df['SNR_dB_config'] = df['SNR_dB']
    df['SINR_eff_dB'] = 10 * np.log10(df['SINR_eff'] + 1e-10)
    return df

def generate_plots(df):
    print("\n--- Summary Statistics ---")
    print(df[['SINR_eff_dB', 'H_frob_sq', 'SE', 'BER', 'y_power']].describe())
    
    # Plot 1: Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(df['SE'], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Spectral Efficiency Distribution')
    axes[0].set_xlabel('SE (bits/s/Hz)')
    
    # Use log scale for BER to see the small values
    axes[1].hist(np.log10(df['BER'] + 1e-10), bins=30, color='lightcoral', edgecolor='black')
    axes[1].set_title('BER Distribution (log10)')
    axes[1].set_xlabel('log10(BER)')
    
    axes[2].hist(np.log10(df['y_power'] + 1e-10), bins=30, color='lightgreen', edgecolor='black')
    axes[2].set_title('y_power Distribution (log10)')
    axes[2].set_xlabel('log10(y_power)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_distributions.png"), dpi=150)
    plt.close()
    
    # Plot 2: Trends vs Configured SNR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df.boxplot(column='SE', by='SNR_dB_config', ax=axes[0])
    axes[0].set_title('SE vs Configured SNR')
    axes[0].set_ylabel('SE (bits/s/Hz)')
    axes[0].set_xlabel('SNR Config (dB)')
    
    df.boxplot(column='BER', by='SNR_dB_config', ax=axes[1])
    axes[1].set_title('BER vs Configured SNR')
    axes[1].set_ylabel('BER')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('SNR Config (dB)')
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_trends_snr.png"), dpi=150)
    plt.close()
    
    # Plot 3: Trends vs N_RIS
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df.boxplot(column='SE', by='N_RIS', ax=axes[0])
    axes[0].set_title('SE vs N_RIS')
    axes[0].set_ylabel('SE (bits/s/Hz)')
    axes[0].set_xlabel('Number of RIS Elements')
    
    df.boxplot(column='H_frob_sq', by='N_RIS', ax=axes[1])
    axes[1].set_title(r'$||H||_F^2$ vs N_RIS')
    axes[1].set_ylabel('Channel Frobenius Norm Sq')
    axes[1].set_xlabel('Number of RIS Elements')
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_trends_ris.png"), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {PLOTS_DIR}/")

if __name__ == "__main__":
    df = run_validation()
    generate_plots(df)
