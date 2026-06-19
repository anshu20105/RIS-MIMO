"""
Model Loader
=============
Loads the trained PINN checkpoint and reconstructs the exact
StandardScaler objects used during training.
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths (relative to project root — Streamlit launched from there)
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "datasets", "digital_twin_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "final_pinn_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model architecture (mirrors train_final_model.py exactly)
# ---------------------------------------------------------------------------
class MultiTaskDigitalTwin(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.GELU(),
        )
        self.branch_y_power = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.branch_se      = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.branch_ber     = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        shared = self.trunk(x)
        return self.branch_y_power(shared), self.branch_se(shared), self.branch_ber(shared)


# ---------------------------------------------------------------------------
# Scaler reconstruction (identical logic to training script)
# ---------------------------------------------------------------------------
def _reconstruct_scalers():
    """Re‑fit scalers from the training split so they match the checkpoint."""
    df = pd.read_csv(DATA_FILE)

    indices = np.arange(len(df))
    snr_bins = pd.qcut(df["SNR_dB"], q=4, labels=False, duplicates="drop")
    train_idx, _ = train_test_split(indices, test_size=0.3, stratify=snr_bins, random_state=42)

    # Build raw X for training rows
    X_train = []
    for idx in train_idx:
        row = df.iloc[idx]
        base = [row["frequency"], row["N_Tx"], row["N_Rx"], row["N_RIS"],
                row["dx"], row["dy"], row["SNR_dB"]]
        p_real = ast.literal_eval(row["phase_shift_real"])
        p_imag = ast.literal_eval(row["phase_shift_imag"])
        pad = 128 - len(p_real)
        p_real_padded = np.pad(p_real, (0, pad), "constant")
        p_imag_padded = np.pad(p_imag, (0, pad), "constant")
        X_train.append(np.concatenate((base, p_real_padded, p_imag_padded)))
    X_train = np.array(X_train, dtype=np.float32)

    scaler_X = StandardScaler().fit(X_train)

    yp = np.log10(df.iloc[train_idx]["y_power"].values.astype(np.float32).reshape(-1, 1) + 1e-10)
    se = df.iloc[train_idx]["SE"].values.astype(np.float32).reshape(-1, 1)
    ber = np.log10(df.iloc[train_idx]["BER"].values.astype(np.float32).reshape(-1, 1) + 1e-12)

    scaler_yp  = StandardScaler().fit(yp)
    scaler_se  = StandardScaler().fit(se)
    scaler_ber = StandardScaler().fit(ber)

    return scaler_X, scaler_yp, scaler_se, scaler_ber


# ---------------------------------------------------------------------------
# Public loader (call once, cache with st.cache_resource)
# ---------------------------------------------------------------------------
def load_model_and_scalers():
    """Returns (model, scaler_X, scaler_yp, scaler_se, scaler_ber, DEVICE)."""
    scaler_X, scaler_yp, scaler_se, scaler_ber = _reconstruct_scalers()

    input_dim = scaler_X.mean_.shape[0]          # 263
    model = MultiTaskDigitalTwin(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    return model, scaler_X, scaler_yp, scaler_se, scaler_ber, DEVICE
