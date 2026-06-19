import pickle
import os

FILE_PATH = "RIS_Datasets/RIS_8_Tx_2_Rx_2.pkl"

# Check if the file exists before attempting to open it
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Dataset file not found at: {FILE_PATH}")

# Load the optimized dataset
with open(FILE_PATH, "rb") as f:
    data = pickle.load(f)

# ------------------------------------------------------------
# Extract Metadata
# ------------------------------------------------------------
meta = data.get("meta", {})
n_samples = meta.get("samples", data["H_real"].shape[0])

print("=" * 50)
print("             RIS DATASET INSPECTION                   ")
print("=" * 50)
print(f"Total Samples (Batch Size) : {n_samples}")
print(f"RIS Element Size           : {meta.get('RIS_size', 'N/A')}")
print(f"Tx Antennas                : {meta.get('N_Tx', 'N/A')}")
print(f"Rx Antennas                : {meta.get('N_Rx', 'N/A')}")
print(f"Target SNR                 : {meta.get('SNR_dB', 'N/A')} dB")
print("-" * 50)

# ------------------------------------------------------------
# Inspect Tensor Shapes
# ------------------------------------------------------------
print("Global Dictionary Keys:")
print(list(data.keys()))
print("-" * 50)

print("Vectorized Tensor Shapes (N_Samples, Dim1, Dim2):")
# We use f-strings to neatly format the shape dimensions
print(f"  H_real shape : {data['H_real'].shape}")
print(f"  H_imag shape : {data['H_imag'].shape}")
print(f"  x_real shape : {data['x_real'].shape}")
print(f"  x_imag shape : {data['x_imag'].shape}")
print(f"  y_real shape : {data['y_real'].shape}")
print(f"  y_imag shape : {data['y_imag'].shape}")
print("=" * 50)