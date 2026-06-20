"""
Sensitivity Analysis (app module)
==================================
Runs parameter sweeps and returns structured data for the tornado chart.
Supports SE, BER, y_power, capacity_mbps, and sinr_db metrics.
"""

from inference import predict


# Parameter sweep ranges — 10 parameters per supervisor request
SWEEP_SPACE = {
    "freq":      [3.5e9, 6e9, 26e9],
    "n_tx":      [2, 4, 8],
    "n_rx":      [2, 4, 8],
    "n_ris":     [8, 16, 32, 64, 128],
    "dx":        [0.25, 0.5, 1.0],
    "dy":        [0.25, 0.5, 1.0],
    "theta":     [0.0, 1.5708, 3.1416, 4.7124],   # 0, π/2, π, 3π/2
    "snr_db":    [-10, 0, 10, 20],
    "d_tx_ris":  [5.0, 15.0, 30.0, 60.0, 100.0],
    "d_ris_rx":  [5.0, 15.0, 30.0, 60.0, 100.0],
}

# Human-readable labels
LABELS = {
    "freq":      "Frequency",
    "n_tx":      "N_Tx",
    "n_rx":      "N_Rx",
    "n_ris":     "N_RIS",
    "dx":        "dx (λ)",
    "dy":        "dy (λ)",
    "theta":     "Phase Shift",
    "snr_db":    "SNR (dB)",
    "d_tx_ris":  "Tx→RIS Dist",
    "d_ris_rx":  "RIS→Rx Dist",
}

# All supported target metrics
METRIC_KEYS = ["SE", "BER", "y_power", "capacity_mbps", "sinr_db"]


def run_sensitivity(model, scalers, base_params: dict):
    """
    Sweep each parameter while holding others at base_params values.

    Returns dict like:
        { 'SE':           [(param_label, delta), ...],
          'BER':          [(param_label, delta), ...],
          'y_power':      [(param_label, delta), ...],
          'capacity_mbps':[(param_label, delta), ...],
          'sinr_db':      [(param_label, delta), ...] }
    """
    scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers

    tornado = {k: [] for k in METRIC_KEYS}

    for param_key, sweep_vals in SWEEP_SPACE.items():
        lists = {k: [] for k in METRIC_KEYS}

        for val in sweep_vals:
            params = base_params.copy()
            params[param_key] = val
            res = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber,
                          device, **params)
            lists["SE"].append(res["se"])
            lists["BER"].append(res["ber"])
            lists["y_power"].append(res["y_power"])
            lists["capacity_mbps"].append(res["capacity_mbps"])
            lists["sinr_db"].append(res["sinr_db"])

        label = LABELS.get(param_key, param_key)
        for key in METRIC_KEYS:
            vals = lists[key]
            tornado[key].append((label, max(vals) - min(vals)))

    return tornado
