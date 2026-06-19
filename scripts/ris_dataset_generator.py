import numpy as np
import pandas as pd
import os
import itertools
import time
from scipy.special import j0

# ============================================================
# Upgraded RIS-Assisted MIMO Dataset Generator (PINN Compatible)
# ============================================================

# -----------------------------
# Global Parameters
# -----------------------------
# For full generation, this will create 3 * 5 * 3 * 3 * 3 * 3 * 4 = 4860 scenarios.
N_DATA = 10  # Reduced to keep the combined single CSV small (48600 rows total)

FREQUENCIES = [3.5e9, 6e9, 26e9]
RIS_SIZES = [8, 16, 32, 64, 128]
TX_VALUES = [2, 4, 8]
RX_VALUES = [2, 4, 8]
DX_VALUES = [0.25, 0.5, 1.0] # In terms of wavelength
DY_VALUES = [0.25, 0.5, 1.0] # In terms of wavelength
SNR_VALUES = [-10, 0, 10, 20]

C = 3e8  # Speed of light
SAVE_FILE = "comprehensive_dataset.csv"

# -----------------------------
# Vectorized Core Functions
# -----------------------------

def get_ris_dimensions(n_ris):
    """
    Approximates the RIS as a 2D grid (Nx x Ny) for element spacing math.
    """
    if n_ris == 8:
        return 4, 2
    elif n_ris == 16:
        return 4, 4
    elif n_ris == 32:
        return 8, 4
    elif n_ris == 64:
        return 8, 8
    elif n_ris == 128:
        return 16, 8
    else:
        # Default fallback to roughly square
        nx = int(np.sqrt(n_ris))
        ny = n_ris // nx
        return nx, ny

def get_ris_spatial_correlation(n_ris, dx_wl, dy_wl):
    """
    Computes spatial correlation matrix R_RIS for the RIS elements based on
    the Jakes isotropic scattering model (J0(2 * pi * d)).
    dx_wl, dy_wl are the spacing in units of wavelength.
    """
    nx, ny = get_ris_dimensions(n_ris)
    
    # Actually just generating grid points
    # Re-assure nx * ny == n_ris
    actual_elements = nx * ny
    if actual_elements != n_ris:
        # If dimensions don't perfectly multiply, just pad to 1D
        nx, ny = n_ris, 1
        
    positions = []
    for i in range(nx):
        for j in range(ny):
            positions.append((i * dx_wl, j * dy_wl))
            
    positions = np.array(positions)
    n_actual = len(positions)
    
    R = np.zeros((n_actual, n_actual))
    for i in range(n_actual):
        for j in range(n_actual):
            dist = np.sqrt((positions[i,0] - positions[j,0])**2 + (positions[i,1] - positions[j,1])**2)
            R[i,j] = j0(2 * np.pi * dist)
            
    return R

def generate_cascaded_channel(n_data, n_tx, n_rx, n_ris, dx_wl, dy_wl):
    """
    Generates cascaded channels with spatial correlation at the RIS.
    For simplicity, assume Tx and Rx antennas are uncorrelated.
    """
    R_ris = get_ris_spatial_correlation(n_ris, dx_wl, dy_wl)
    
    # Compute R_ris^(1/2) using Cholesky or SVD
    # Use SVD to be safe against numerical precision issues making R non-pd
    U, S, Vh = np.linalg.svd(R_ris)
    R_half = U @ np.diag(np.sqrt(np.abs(S))) @ Vh
    
    # 1. Tx to RIS channel (n_ris x n_tx)
    # H_d = R_half @ H_iid
    H_iid_d = (np.random.randn(n_data, n_ris, n_tx) + 1j * np.random.randn(n_data, n_ris, n_tx)) / np.sqrt(2)
    # R_half is (n_ris, n_ris). We can multiply batch:
    H_d = np.matmul(R_half, H_iid_d)
    
    # 2. RIS to Rx channel (n_rx x n_ris)
    # G = G_iid @ R_half
    G_iid = (np.random.randn(n_data, n_rx, n_ris) + 1j * np.random.randn(n_data, n_rx, n_ris)) / np.sqrt(2)
    G = np.matmul(G_iid, R_half)
    
    return G, H_d

def generate_ris_phases(n_data, n_ris):
    """Generates random phase shifts and formats them into a diagonal matrix."""
    phases = np.random.uniform(0, 2 * np.pi, (n_data, n_ris))
    # Create batch of diagonal matrices containing the phase shifts
    # Using np.exp(1j * phase)
    theta_diag = np.exp(1j * phases)
    # To construct diagonal matrix from a vector batch:
    # A cleaner approach is element-wise broadcasting later or explicit diag
    # For full matrix Theta: shape (N, n_ris, n_ris)
    Theta = np.zeros((n_data, n_ris, n_ris), dtype=complex)
    idx = np.arange(n_ris)
    Theta[:, idx, idx] = theta_diag
    
    return theta_diag, Theta

def generate_transmitted_signals(n_data, n_tx):
    """Generates all x vectors at once. Shape: (N, n_tx, 1)"""
    return (np.random.randn(n_data, n_tx, 1) + 1j * np.random.randn(n_data, n_tx, 1)) / np.sqrt(2)

import scipy.special

# -----------------------------
# Main Dataset Generation Loop
# -----------------------------
SAVE_FILE_V2 = "datasets/digital_twin_dataset.csv"

def generate_dataset():
    total_start_time = time.time()
    scenario_count = 0
    scenarios = list(itertools.product(FREQUENCIES, RIS_SIZES, TX_VALUES, RX_VALUES, DX_VALUES, DY_VALUES, SNR_VALUES))
    
    print(f"Total Scenarios to run: {len(scenarios)}")
    
    for scenario_idx, (freq, n_ris, n_tx, n_rx, dx, dy, snr) in enumerate(scenarios):
        scenario_start = time.time()
        
        wavelength = C / freq
        
        # 1. Vectorized Generation steps
        G, H_d = generate_cascaded_channel(N_DATA, n_tx, n_rx, n_ris, dx, dy)
        phase_shifts, Theta = generate_ris_phases(N_DATA, n_ris)
        
        # Calculate End-to-End Equivalent Channel: H_eq = G @ Theta @ H_d
        H_eq = np.matmul(G, np.matmul(Theta, H_d))
        
        # Transmitted Signal
        x = generate_transmitted_signals(N_DATA, n_tx)
        
        # Rx Signal
        signal = np.matmul(H_eq, x)
        
        # Noise calculation
        signal_power = np.sum(np.abs(signal)**2, axis=(1, 2), keepdims=True)
        snr_linear = 10 ** (snr / 10)
        noise_std = np.sqrt(signal_power / (snr_linear * n_rx * 2))
        noise_var = noise_std**2 * 2  # Total complex variance per receive antenna
        
        noise = noise_std * (np.random.randn(N_DATA, n_rx, 1) + 1j * np.random.randn(N_DATA, n_rx, 1))
        y = signal + noise
        
        # ----- NEW MULTI-TASK DERIVED LABELS -----
        # 1. y_power (Total received envelope power)
        y_power = np.sum(np.abs(y)**2, axis=(1, 2))  # shape: (N_DATA,)
        
        se_arr = np.zeros(N_DATA)
        sinr_eff_arr = np.zeros(N_DATA)
        ber_arr = np.zeros(N_DATA)
        frob_sq_arr = np.zeros(N_DATA)
        
        for i in range(N_DATA):
            H_i = H_eq[i]
            # Channel Covariance Matrix (R_eq = H_eq @ H_eq.H)
            R_eq = H_i @ H_i.conj().T
            
            # Effective SINR per receive antenna (assuming equal power allocation without CSIT)
            # Power allocated per tx antenna is P_tx / N_tx (we set P_tx = 1)
            # Frobenius norm squared = Trace(R_eq)
            frobenius_sq = np.real(np.trace(R_eq))
            frob_sq_arr[i] = frobenius_sq
            
            # eff_sinr = signal power / noise floor
            # Note: noise_var[i] is already a scalar shape (1,1). We extract it.
            nv = float(noise_var[i][0][0])
            # If noise is extremely small or zero, bound it
            if nv < 1e-12: nv = 1e-12
            
            eff_sinr = (frobenius_sq / n_tx) / nv
            sinr_eff_arr[i] = eff_sinr
            
            # Spectral Efficiency (Shannon limits without water-filling)
            # SE = (1/N_tx) * log2(det(I + (P/N_tx * sigma^2) * R_eq))
            I_rx = np.eye(n_rx)
            capacity_matrix = I_rx + (1.0 / (n_tx * nv)) * R_eq
            # np.linalg.det on complex matrices returns complex, we take real part
            det_val = np.real(np.linalg.det(capacity_matrix))
            # Protect against numerical issues returning negative determinents
            if det_val < 1.0: det_val = 1.0
            
            se = np.log2(det_val) / n_tx
            se_arr[i] = se
            
            # BER (QPSK / 4-QAM AWGN approximation bound based on effective SINR)
            # BER ~= 0.5 * erfc(sqrt(SINR / 2))
            ber = 0.5 * scipy.special.erfc(np.sqrt(eff_sinr / 2.0))
            ber_arr[i] = ber
            
        # 2. Package into Dataset Lists
        data_rows = []
        for i in range(N_DATA):
            data_rows.append({
                "frequency": freq,
                "wavelength": wavelength,
                "N_Tx": n_tx,
                "N_Rx": n_rx,
                "N_RIS": n_ris,
                "dx": dx,
                "dy": dy,
                "SNR_dB": snr,
                
                # --- NEW LABELS ---
                "H_frob_sq": frob_sq_arr[i],
                "SINR_eff": sinr_eff_arr[i],
                "SE": se_arr[i],
                "BER": ber_arr[i],
                "y_power": y_power[i],
                # ------------------
                
                # Using standard list string representation for arrays so Pandas saves it cleanly in CSV
                "phase_shift_real": str(np.real(phase_shifts[i]).tolist()),
                "phase_shift_imag": str(np.imag(phase_shifts[i]).tolist()),
                "H_real": str(np.real(H_eq[i]).flatten().tolist()),
                "H_imag": str(np.imag(H_eq[i]).flatten().tolist()),
                "x_real": str(np.real(x[i]).flatten().tolist()),
                "x_imag": str(np.imag(x[i]).flatten().tolist()),
                "y_real": str(np.real(y[i]).flatten().tolist()),
                "y_imag": str(np.imag(y[i]).flatten().tolist())
            })
            
        df = pd.DataFrame(data_rows)
        
        # 3. Save to Disk
        mode = 'w' if scenario_idx == 0 else 'a'
        header = True if scenario_idx == 0 else False
        df.to_csv(SAVE_FILE_V2, mode=mode, header=header, index=False)
        
        scenario_count += 1
        
        if (scenario_idx + 1) % 50 == 0:
            print(f"[{scenario_idx + 1}/{len(scenarios)}] Appended scenarios to {SAVE_FILE_V2} | Time: {time.time() - scenario_start:.2f}s")
            
    print("=" * 60)
    print(f"Generation Complete! Total Scenarios: {scenario_count}")
    print(f"Total Execution Time: {time.time() - total_start_time:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    generate_dataset()