import sys
import os
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from scipy.stats import gaussian_kde

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_loader import load_model_and_scalers
from inference import predict, generate_y_distribution, generate_antenna_coordinates
from sensitivity_analysis import run_sensitivity

app = FastAPI(title="RIS-MIMO Digital Twin API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load global models & scalers
model, scaler_X, scaler_yp, scaler_se, scaler_ber, device = load_model_and_scalers()
scalers = (scaler_X, scaler_yp, scaler_se, scaler_ber, device)

class PredictRequest(BaseModel):
    freq: float
    n_tx: int
    n_rx: int
    n_ris: int
    dx: float
    dy: float
    snr_db: float
    theta: float
    d_tx_ris: float
    d_ris_rx: float
    tx_array_type: str
    tx_rows: int
    tx_cols: int
    dx_tx: float
    dy_tx: float
    rx_array_type: str
    rx_rows: int
    rx_cols: int
    dx_rx: float
    dy_rx: float

@app.post("/api/predict")
def api_predict(req: PredictRequest):
    params = req.dict()
    
    # 1. Run PINN Model prediction
    metrics = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **params)
    
    # 2. Generate coordinates
    tx_coords = generate_antenna_coordinates(
        params["tx_array_type"], params["tx_rows"], params["tx_cols"], params["dx_tx"], params["dy_tx"]
    )
    rx_coords = generate_antenna_coordinates(
        params["rx_array_type"], params["rx_rows"], params["rx_cols"], params["dx_rx"], params["dy_rx"]
    )
    
    # 3. Generate KDE distribution for received signal power
    y_power = metrics["y_power"]
    samples = generate_y_distribution(y_power, n_samples=3000)
    samples_valid = samples[samples > 0]
    
    kde = gaussian_kde(samples_valid, bw_method="scott")
    x_grid = np.linspace(float(samples_valid.min()), float(samples_valid.max()), 100)
    density = kde(x_grid)
    
    # 4. Generate correlation matrix R_mn = J_0(2 * pi * D_ij / lambda)
    # R_ij is computed directly on coords
    n_rx = rx_coords.shape[0]
    from scipy.special import j0
    D = np.zeros((n_rx, n_rx))
    for i in range(n_rx):
        for j in range(n_rx):
            D[i, j] = np.linalg.norm(rx_coords[i] - rx_coords[j])
    
    R = j0(2 * np.pi * D)
    
    return {
        "metrics": metrics,
        "tx_coords": tx_coords.tolist(),
        "rx_coords": rx_coords.tolist(),
        "kde": {
            "x": x_grid.tolist(),
            "y": density.tolist()
        },
        "correlation_matrix": R.tolist()
    }

class SensitivityRequest(BaseModel):
    params: Dict[str, Any]
    mode: str = "differential"  # "differential" or "legacy"

@app.post("/api/sensitivity")
def api_sensitivity(req: SensitivityRequest):
    sens_data = run_sensitivity(model, scalers, req.params, mode=req.mode)
    return {
        "sens_data": sens_data
    }

from fastapi.staticfiles import StaticFiles

# Create static directory if not exists
static_dir = os.path.join(current_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
