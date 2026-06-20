import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "datasets", "digital_twin_dataset.csv")

def extract_scalers():
    print("Reading CSV...", flush=True)
    df = pd.read_csv(DATA_FILE)

    indices = np.arange(len(df))
    snr_bins = pd.qcut(df["SNR_dB"], q=4, labels=False, duplicates="drop")
    train_idx, _ = train_test_split(indices, test_size=0.3, stratify=snr_bins, random_state=42)

    X_train = []
    print("Building X_train...", flush=True)
    for idx in train_idx:
        row = df.iloc[idx]
        base = [row["frequency"], row["N_Tx"], row["N_Rx"], row["N_RIS"], row["dx"], row["dy"], row["SNR_dB"]]
        p_real = ast.literal_eval(row["phase_shift_real"])
        p_imag = ast.literal_eval(row["phase_shift_imag"])
        pad = 128 - len(p_real)
        p_real_padded = np.pad(p_real, (0, pad), "constant")
        p_imag_padded = np.pad(p_imag, (0, pad), "constant")
        X_train.append(np.concatenate((base, p_real_padded, p_imag_padded)))
    
    X_train = np.array(X_train, dtype=np.float32)

    print("Fitting Scaler X...", flush=True)
    scaler_X = StandardScaler().fit(X_train)

    print("Fitting Target Scalers...", flush=True)
    yp = np.log10(df.iloc[train_idx]["y_power"].values.astype(np.float32).reshape(-1, 1) + 1e-10)
    se = df.iloc[train_idx]["SE"].values.astype(np.float32).reshape(-1, 1)
    ber = np.log10(df.iloc[train_idx]["BER"].values.astype(np.float32).reshape(-1, 1) + 1e-12)

    scaler_yp  = StandardScaler().fit(yp)
    scaler_se  = StandardScaler().fit(se)
    scaler_ber = StandardScaler().fit(ber)

    return scaler_X, scaler_yp, scaler_se, scaler_ber

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pinn_scalers.pkl")
    scalers = extract_scalers()
    with open(out_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {out_path}", flush=True)
