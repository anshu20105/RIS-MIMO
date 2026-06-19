"""
Sensitivity Analysis (app module)
==================================
Runs parameter sweeps and returns structured data for the tornado chart.
"""

from inference import predict


# Parameter sweep ranges
SWEEP_SPACE = {
    "freq":   [3.5e9, 6e9, 26e9],
    "n_ris":  [8, 16, 32, 64, 128],
    "n_tx":   [2, 4, 8],
    "n_rx":   [2, 4, 8],
    "snr_db": [-10, 0, 10, 20],
}

# Pretty labels
LABELS = {
    "freq":   "Frequency",
    "n_ris":  "RIS Size",
    "n_tx":   "N_Tx",
    "n_rx":   "N_Rx",
    "snr_db": "SNR (dB)",
}


def run_sensitivity(model, scalers, base_params: dict):
    """
    Sweep each parameter while holding others at base_params values.

    Returns dict like:
        { 'SE':      [(param_label, delta), ...],
          'BER':     [(param_label, delta), ...],
          'y_power': [(param_label, delta), ...] }
    """
    scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers

    tornado = {"SE": [], "BER": [], "y_power": []}

    for param_key, sweep_vals in SWEEP_SPACE.items():
        se_list, ber_list, yp_list = [], [], []

        for val in sweep_vals:
            params = base_params.copy()
            params[param_key] = val
            res = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber,
                          device, **params)
            se_list.append(res["se"])
            ber_list.append(res["ber"])
            yp_list.append(res["y_power"])

        label = LABELS.get(param_key, param_key)
        tornado["SE"].append((label, max(se_list) - min(se_list)))
        tornado["BER"].append((label, max(ber_list) - min(ber_list)))
        tornado["y_power"].append((label, max(yp_list) - min(yp_list)))

    return tornado
