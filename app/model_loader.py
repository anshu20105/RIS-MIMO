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
SCALERS_FILE = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pinn_scalers.pkl")
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
    """Load scalers from pickled file."""
    import pickle
    with open(SCALERS_FILE, 'rb') as f:
        return pickle.load(f)


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
