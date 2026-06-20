"""
Sensitivity Analysis (app module)
==================================
Runs parameter sweeps and returns structured data for the tornado chart.
Supports SE, BER, y_power, capacity_mbps, and sinr_db metrics.
"""

from inference import predict


# All supported target metrics
METRIC_KEYS = ["SE", "BER", "y_power", "capacity_mbps", "sinr_db"]


def run_sensitivity(model, scalers, base_params: dict):
    """
    Sweep each parameter while holding others at base_params values.
    Dynamically builds sweep spaces according to the selected geometry modes.
    """
    scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers

    # Base common sweep parameters
    sweep_space = {
        "freq":      ([3.5e9, 6e9, 26e9], "Frequency"),
        "n_ris":     ([8, 16, 32, 64, 128], "N_RIS"),
        "theta":     ([0.0, 1.5708, 3.1416, 4.7124], "Phase Shift"),
        "snr_db":    ([-10, 0, 10, 20], "SNR (dB)"),
        "d_tx_ris":  ([5.0, 15.0, 30.0, 60.0, 100.0], "Tx→RIS Dist"),
        "d_ris_rx":  ([5.0, 15.0, 30.0, 60.0, 100.0], "RIS→Rx Dist"),
    }

    # Tx Array Geometry sweeps
    if base_params.get("tx_array_type", "Linear Array") == "Linear Array":
        sweep_space["n_tx"] = ([2, 4, 8, 16], "N_Tx")
        sweep_space["dx_tx"] = ([0.25, 0.5, 1.0], "d (Tx, λ)")
    else:
        sweep_space["tx_rows"] = ([1, 2, 4, 8], "Tx Rows")
        sweep_space["tx_cols"] = ([1, 2, 4, 8], "Tx Columns")
        sweep_space["dx_tx"] = ([0.25, 0.5, 1.0], "dx (Tx, λ)")
        sweep_space["dy_tx"] = ([0.25, 0.5, 1.0], "dy (Tx, λ)")

    # Rx Array Geometry sweeps
    if base_params.get("rx_array_type", "Linear Array") == "Linear Array":
        sweep_space["n_rx"] = ([2, 4, 8, 16], "N_Rx")
        sweep_space["dx_rx"] = ([0.25, 0.5, 1.0], "d (Rx, λ)")
    else:
        sweep_space["rx_rows"] = ([1, 2, 4, 8], "Rx Rows")
        sweep_space["rx_cols"] = ([1, 2, 4, 8], "Rx Columns")
        sweep_space["dx_rx"] = ([0.25, 0.5, 1.0], "dx (Rx, λ)")
        sweep_space["dy_rx"] = ([0.25, 0.5, 1.0], "dy (Rx, λ)")

    tornado = {k: [] for k in METRIC_KEYS}

    for param_key, (sweep_vals, label) in sweep_space.items():
        lists = {k: [] for k in METRIC_KEYS}

        for val in sweep_vals:
            params = base_params.copy()
            params[param_key] = val
            
            # If changing rows/cols directly, we must recompute n_tx or n_rx so the inference model 
            # uses the updated total elements. (And set dx/dy if required)
            if param_key in ["tx_rows", "tx_cols"]:
                params["n_tx"] = params["tx_rows"] * params["tx_cols"]
            if param_key in ["rx_rows", "rx_cols"]:
                params["n_rx"] = params["rx_rows"] * params["rx_cols"]
            if param_key in ["dx_rx", "dy_rx"]:
                params["dx"] = params["dx_rx"]
                params["dy"] = params["dy_rx"] if params.get("rx_array_type") == "Rectangular Array" else params["dx_rx"]

            res = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber,
                          device, **params)
            
            lists["SE"].append(res["se"])
            lists["BER"].append(res["ber"])
            lists["y_power"].append(res["y_power"])
            lists["capacity_mbps"].append(res["capacity_mbps"])
            lists["sinr_db"].append(res["sinr_db"])

        for k in METRIC_KEYS:
            vals = lists[k]
            tornado[k].append((label, max(vals) - min(vals)))

    return tornado
