"""
Sensitivity Analysis (app module)
==================================
Runs parameter sweeps and returns structured data for the basic sensitivity chart.
Supports SE, BER, y_power, capacity_mbps, and sinr_db metrics.
Supports both legacy Max-Δ sweep and normalized differential sensitivity.
"""

import numpy as np
from inference import predict

# All supported target metrics
METRIC_KEYS = ["SE", "BER", "y_power", "capacity_mbps", "sinr_db"]


def get_discrete_neighbors(val, valid_list):
    """
    Returns (val_minus, val_plus) as the neighboring configurations in the sorted valid list.
    """
    valid_sorted = sorted(list(set(valid_list)))
    
    if val in valid_sorted:
        idx = valid_sorted.index(val)
        if len(valid_sorted) == 1:
            return val, val
        elif idx == 0:
            return valid_sorted[0], valid_sorted[1]
        elif idx == len(valid_sorted) - 1:
            return valid_sorted[-2], valid_sorted[-1]
        else:
            return valid_sorted[idx - 1], valid_sorted[idx + 1]
    
    # If not in the list, locate adjacent interval
    if val < valid_sorted[0]:
        return valid_sorted[0], valid_sorted[1]
    if val > valid_sorted[-1]:
        return valid_sorted[-2], valid_sorted[-1]
        
    for i in range(len(valid_sorted) - 1):
        if valid_sorted[i] < val < valid_sorted[i+1]:
            return valid_sorted[i], valid_sorted[i+1]
            
    val_int = int(round(val))
    return max(1, val_int - 1), val_int + 1


def run_sensitivity(model, scalers, base_params: dict, mode: str = "differential"):
    """
    Evaluate parameter sensitivity.
    - If mode == "differential", calculates normalized differential sensitivity (elasticity):
      S_i = (∂y / ∂x_i) * (x_i / y) at the current base configuration.
    - If mode == "legacy", performs standard Max-Δ sweep over parameter ranges.
    """
    scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers

    if mode == "legacy":
        # Base common sweep parameters
        sweep_space = {
            "freq":      ([3.5e9, 6e9, 26e9], "Frequency"),
            "n_ris":     ([8, 16, 32, 64, 128], "N_RIS"),
            "theta":     ([0.0, 1.5708, 3.1416, 4.7124], "Phase Shift"),
            "snr_db":    ([-10, 0, 10, 20], "SNR"),
            "d_tx_ris":  ([5.0, 15.0, 30.0, 60.0, 100.0], "Tx→RIS Dist"),
            "d_ris_rx":  ([5.0, 15.0, 30.0, 60.0, 100.0], "RIS→Rx Dist"),
        }

        # Tx Array Geometry sweeps
        if base_params.get("tx_array_type", "Linear Array") == "Linear Array":
            sweep_space["n_tx"] = ([2, 4, 8, 16], "N_Tx")
            sweep_space["dx_tx"] = ([0.25, 0.5, 1.0], "dx (Tx, λ)")
        else:
            sweep_space["tx_rows"] = ([1, 2, 4, 8], "Tx Rows")
            sweep_space["tx_cols"] = ([1, 2, 4, 8], "Tx Columns")
            sweep_space["dx_tx"] = ([0.25, 0.5, 1.0], "dx (Tx, λ)")
            sweep_space["dy_tx"] = ([0.25, 0.5, 1.0], "dy (Tx, λ)")

        # Rx Array Geometry sweeps
        if base_params.get("rx_array_type", "Linear Array") == "Linear Array":
            sweep_space["n_rx"] = ([2, 4, 8, 16], "N_Rx")
            sweep_space["dx_rx"] = ([0.25, 0.5, 1.0], "dx (Rx, λ)")
        else:
            sweep_space["rx_rows"] = ([1, 2, 4, 8], "Rx Rows")
            sweep_space["rx_cols"] = ([1, 2, 4, 8], "Rx Columns")
            sweep_space["dx_rx"] = ([0.25, 0.5, 1.0], "dx (Rx, λ)")
            sweep_space["dy_rx"] = ([0.25, 0.5, 1.0], "dy (Rx, λ)")

        sens_data = {k: [] for k in METRIC_KEYS}

        for param_key, (sweep_vals, label) in sweep_space.items():
            lists = {k: [] for k in METRIC_KEYS}

            for val in sweep_vals:
                params = base_params.copy()
                params[param_key] = val
                
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
                sens_data[k].append((label, max(vals) - min(vals)))

        return sens_data

    # Differential Sensitivity mode
    # Compute base prediction reference point
    res_base = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **base_params)

    # 10 specific inputs requested
    inputs_to_eval = [
        ("freq", "Frequency", "continuous", None),
        ("n_tx", "N_Tx", "discrete", [2, 4, 8, 16]),
        ("n_rx", "N_Rx", "discrete", [2, 4, 8, 16]),
        ("n_ris", "N_RIS", "discrete", [8, 16, 32, 64, 128]),
        ("dx", "dx (λ)", "continuous", None),
        ("dy", "dy (λ)", "continuous", None),
        ("theta", "Phase Shift", "continuous_theta", None),
        ("snr_db", "SNR", "continuous_snr", None),
        ("d_tx_ris", "Tx→RIS Dist", "continuous", None),
        ("d_ris_rx", "RIS→Rx Dist", "continuous", None),
    ]

    sens_data = {k: [] for k in METRIC_KEYS}

    for key, label, p_type, opt_list in inputs_to_eval:
        params_minus = base_params.copy()
        params_plus = base_params.copy()

        # Handle dx / dy defaults if they are missing in base_params
        if key not in base_params:
            if key == "dx":
                base_params[key] = base_params.get("dx_rx", 0.5)
            elif key == "dy":
                base_params[key] = base_params.get("dy_rx", base_params.get("dx_rx", 0.5))

        val = base_params[key]

        if p_type == "discrete":
            val_minus, val_plus = get_discrete_neighbors(val, opt_list)
            params_minus[key] = val_minus
            params_plus[key] = val_plus
            dx_val = val_plus - val_minus
            x_base = val
        elif p_type == "continuous":
            delta = 0.05 * val
            val_minus = val - delta
            val_plus = val + delta
            params_minus[key] = val_minus
            params_plus[key] = val_plus
            dx_val = val_plus - val_minus
            x_base = val
        elif p_type == "continuous_theta":
            delta = max(0.05 * abs(val), 0.01)
            val_minus = val - delta
            val_plus = val + delta
            params_minus[key] = val_minus
            params_plus[key] = val_plus
            dx_val = val_plus - val_minus
            x_base = val
        elif p_type == "continuous_snr":
            # Linear SNR linearization for sensitivity calculation
            snr_lin = 10 ** (val / 10.0)
            snr_lin_minus = 0.95 * snr_lin
            snr_lin_plus = 1.05 * snr_lin
            
            params_minus[key] = 10 * np.log10(snr_lin_minus)
            params_plus[key] = 10 * np.log10(snr_lin_plus)
            dx_val = snr_lin_plus - snr_lin_minus
            x_base = snr_lin
        else:
            dx_val = 0.0
            x_base = val

        # Evaluate model predictions at perturbed configurations
        res_minus = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **params_minus)
        res_plus = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **params_plus)

        # For each target metric, compute normalized sensitivity coefficient
        for metric in METRIC_KEYS:
            pred_key = {
                "SE": "se",
                "BER": "ber",
                "y_power": "y_power",
                "capacity_mbps": "capacity_mbps",
                "sinr_db": "sinr_db",
            }[metric]

            y_minus = res_minus[pred_key]
            y_plus = res_plus[pred_key]
            y_base = res_base[pred_key]

            # Compute central finite difference
            if dx_val != 0:
                dydx = (y_plus - y_minus) / dx_val
            else:
                dydx = 0.0

            # Compute local elasticity S_i = (dy/dx) * (x/y)
            denom = y_base if abs(y_base) > 1e-15 else 1e-15
            s_val = dydx * (x_base / denom)

            # Prevent NaN or infinite values
            if np.isnan(s_val) or np.isinf(s_val):
                s_val = 0.0

            sens_data[metric].append((label, s_val))

    return sens_data
