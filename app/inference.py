"""
Inference Engine
=================
Builds feature vectors from user parameters, runs the PINN model,
and computes all derived metrics.
"""

import numpy as np
import torch
from scipy.stats import rayleigh

# System bandwidth assumption (Hz) — standard 5G NR channel bandwidth
BANDWIDTH_HZ = 20e6  # 20 MHz


def build_feature_vector(freq, n_tx, n_rx, n_ris, dx, dy, snr_db, theta):
    """
    Build a 263‑dim feature vector from user‑facing parameters.

    Phase shifts:  Each RIS element has phase θ  →  exp(jθ) split into
    real and imag parts, zero‑padded to 128 each.
    """
    # Scalar features
    base = [freq, n_tx, n_rx, n_ris, dx, dy, snr_db]

    # Phase shifts: uniform θ applied to all N_RIS elements
    n_ris_int = int(n_ris)
    phases = np.full(n_ris_int, theta)
    p_real = np.cos(phases)
    p_imag = np.sin(phases)

    # Pad to 128
    pad_len = 128 - n_ris_int
    p_real_padded = np.pad(p_real, (0, max(0, pad_len)), "constant")[:128]
    p_imag_padded = np.pad(p_imag, (0, max(0, pad_len)), "constant")[:128]

    return np.concatenate((base, p_real_padded, p_imag_padded)).astype(np.float32)


def predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device,
            freq, n_tx, n_rx, n_ris, dx, dy, snr_db, theta):
    """
    Run a single forward pass and return a dict of all output metrics.
    """
    x_raw = build_feature_vector(freq, n_tx, n_rx, n_ris, dx, dy, snr_db, theta)
    x_scaled = scaler_X.transform(x_raw.reshape(1, -1))
    x_tensor = torch.tensor(x_scaled, device=device, dtype=torch.float32)

    with torch.no_grad():
        yp_pred, se_pred, ber_pred = model(x_tensor)

    # Inverse‑transform to physical units
    yp_phys  = scaler_yp.inverse_transform(yp_pred.cpu().numpy())[0, 0]   # log10(y_power)
    se_phys  = scaler_se.inverse_transform(se_pred.cpu().numpy())[0, 0]   # SE (bits/s/Hz per stream)
    ber_phys = scaler_ber.inverse_transform(ber_pred.cpu().numpy())[0, 0] # log10(BER)

    y_power = 10 ** yp_phys          # linear y_power
    se      = float(se_phys)
    ber     = 10 ** ber_phys          # linear BER

    # Derived metrics
    sinr_eff = max(2 ** (se * n_tx) - 1, 1e-12)           # inverse Shannon
    capacity = BANDWIDTH_HZ * se * n_tx                    # bits/s
    mean_y   = 0.0                                         # Rayleigh: zero‑mean
    var_y    = float(y_power)                              # E[|y|²]

    return {
        "y_power":  float(y_power),
        "se":       se,
        "ber":      float(ber),
        "sinr_eff": float(sinr_eff),
        "sinr_db":  float(10 * np.log10(max(sinr_eff, 1e-12))),
        "capacity": float(capacity),
        "capacity_mbps": float(capacity / 1e6),
        "mean_y":   mean_y,
        "var_y":    var_y,
    }


def predict_sweep(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device,
                  base_params: dict, sweep_param: str, sweep_values: list):
    """
    Sweep one parameter over a list of values, return list of result dicts.
    """
    results = []
    for val in sweep_values:
        params = base_params.copy()
        params[sweep_param] = val
        results.append(predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **params))
    return results


def generate_y_distribution(y_power, n_samples=10000):
    """
    Simulate |y| samples from Rayleigh distribution.
    σ = sqrt(y_power / 2)  →  |y| ~ Rayleigh(σ)
    """
    sigma = np.sqrt(max(y_power, 1e-15) / 2.0)
    return rayleigh.rvs(scale=sigma, size=n_samples)
