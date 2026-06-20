import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from model_loader import _reconstruct_scalers

print("Reconstructing scalers from dataset...")
scaler_X, scaler_yp, scaler_se, scaler_ber = _reconstruct_scalers()

out_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'pinn_scalers.pkl')
with open(out_path, 'wb') as f:
    pickle.dump((scaler_X, scaler_yp, scaler_se, scaler_ber), f)

print(f"Scalers saved successfully to {out_path}")
