"""
Single-Configuration Controlled Experiment
------------------------------------------
Filters the comprehensive dataset to a single RIS-MIMO configuration:
  Frequency = 26 GHz, N_RIS = 32, N_Tx = 4, N_Rx = 4

Goal: Determine if the baseline NN can learn a fixed-dimension system
      before attempting a universal multi-config model.
"""

import os
import ast
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Dataset
# -------------------------------------------------------------

class RISDatasetSingleConfig(Dataset):
    """
    Dataset for a single RIS-MIMO configuration — no padding needed.
    All rows have identical array dimensions.
    """
    def __init__(self, df, scaler=None, fit_scaler=False, y_scaler=None, fit_y_scaler=False):
        self.df = df.reset_index(drop=True)
        
        t0 = time.time()
        print(f"  Parsing {len(self.df)} rows...")
        
        array_cols = [
            'phase_shift_real', 'phase_shift_imag',
            'H_real', 'H_imag',
            'x_real', 'x_imag',
            'y_real', 'y_imag'
        ]
        
        for col in array_cols:
            if col in self.df.columns and isinstance(self.df[col].iloc[0], str):
                self.df[col] = self.df[col].apply(ast.literal_eval)
        
        print(f"  Parsing completed in {time.time() - t0:.1f}s")
        
        # Scalars
        scalar_cols = ['frequency', 'wavelength', 'N_Tx', 'N_Rx', 'N_RIS', 'dx', 'dy', 'SNR_dB']
        X_scalars = self.df[scalar_cols].values
        
        # Arrays — all same length in single-config mode
        def extract(col_name):
            return np.array([np.array(row).flatten() for row in self.df[col_name]])
        
        X_phases_r = extract('phase_shift_real')
        X_phases_i = extract('phase_shift_imag')
        X_H_r = extract('H_real')
        X_H_i = extract('H_imag')
        X_x_r = extract('x_real')
        X_x_i = extract('x_imag')
        
        Y_r = extract('y_real')
        Y_i = extract('y_imag')
        
        self.X_raw = np.hstack([X_scalars, X_phases_r, X_phases_i, X_H_r, X_H_i, X_x_r, X_x_i])
        self.Y_raw = np.hstack([Y_r, Y_i])
        
        print(f"  Feature shape: {self.X_raw.shape}, Target shape: {self.Y_raw.shape}")
        
        # Input normalization
        if fit_scaler and scaler is not None:
            self.X = scaler.fit_transform(self.X_raw)
        elif scaler is not None:
            self.X = scaler.transform(self.X_raw)
        else:
            self.X = self.X_raw
        
        # Target normalization
        if fit_y_scaler and y_scaler is not None:
            self.Y = y_scaler.fit_transform(self.Y_raw)
        elif y_scaler is not None:
            self.Y = y_scaler.transform(self.Y_raw)
        else:
            self.Y = self.Y_raw
        
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.Y_tensor = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.Y_tensor[idx]


# -------------------------------------------------------------
# 2. Model
# -------------------------------------------------------------

class RISBaselineModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.15))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------------------
# 3. Training
# -------------------------------------------------------------

def evaluate(model, dataloader, criterion, device, y_scaler=None):
    model.eval()
    total_loss = 0.0
    total_mse_phys = 0.0
    total_samples = 0
    all_preds_scaled = []

    with torch.no_grad():
        for bX, bY in dataloader:
            bX, bY = bX.to(device), bY.to(device)
            preds = model(bX)
            loss = criterion(preds, bY)
            bs = bX.size(0)
            total_loss += loss.item() * bs

            preds_np = preds.cpu().numpy()
            targets_np = bY.cpu().numpy()
            all_preds_scaled.append(preds_np)

            if y_scaler is not None:
                p = y_scaler.inverse_transform(preds_np)
                t = y_scaler.inverse_transform(targets_np)
            else:
                p, t = preds_np, targets_np
            total_mse_phys += np.sum((p - t) ** 2)
            total_samples += bs

    avg_loss = total_loss / total_samples
    output_dim = preds_np.shape[1]
    rmse_phys = np.sqrt(total_mse_phys / (total_samples * output_dim))
    all_preds_scaled = np.vstack(all_preds_scaled)
    pred_var = np.mean(np.var(all_preds_scaled, axis=0))
    return avg_loss, rmse_phys, pred_var


def train(model, train_loader, val_loader, epochs, lr, device, save_dir,
          y_scaler=None, patience=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mse_phys = 0.0
        total_samples = 0

        for bX, bY in train_loader:
            bX, bY = bX.to(device), bY.to(device)
            optimizer.zero_grad()
            preds = model(bX)
            
            # Data loss — PINN physics loss hook here
            data_loss = criterion(preds, bY)
            total_loss = data_loss
            
            total_loss.backward()
            optimizer.step()

            bs = bX.size(0)
            running_loss += data_loss.item() * bs
            total_samples += bs

            preds_np = preds.detach().cpu().numpy()
            targets_np = bY.detach().cpu().numpy()
            if y_scaler is not None:
                p = y_scaler.inverse_transform(preds_np)
                t = y_scaler.inverse_transform(targets_np)
            else:
                p, t = preds_np, targets_np
            running_mse_phys += np.sum((p - t) ** 2)

        epoch_train_loss = running_loss / total_samples
        output_dim = preds_np.shape[1]
        train_rmse = np.sqrt(running_mse_phys / (total_samples * output_dim))

        val_loss, val_rmse, pred_var = evaluate(model, val_loader, criterion, device, y_scaler)

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"TrLoss: {epoch_train_loss:.4f} VaLoss: {val_loss:.4f} | "
              f"TrRMSE: {train_rmse:.3f} VaRMSE: {val_rmse:.3f} | "
              f"PVar: {pred_var:.4f} LR: {lr_now:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "model_single_config_best.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best: epoch {best_epoch} (VaLoss={best_val_loss:.4f})")
                break

    torch.save(model.state_dict(), os.path.join(save_dir, "model_single_config_final.pth"))
    return train_losses, val_losses


# -------------------------------------------------------------
# 4. Visualization
# -------------------------------------------------------------

def generate_plots(model, test_loader, device, save_dir, y_scaler, train_losses, val_losses):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds, all_truths = [], []
    with torch.no_grad():
        for bX, bY in test_loader:
            bX = bX.to(device)
            preds = model(bX).cpu().numpy()
            all_preds.append(preds)
            all_truths.append(bY.numpy())

    all_preds = np.vstack(all_preds)
    all_truths = np.vstack(all_truths)

    if y_scaler is not None:
        all_preds = y_scaler.inverse_transform(all_preds)
        all_truths = y_scaler.inverse_transform(all_truths)

    # --- Metrics ---
    residuals = all_preds - all_truths
    mse = np.mean(residuals ** 2)
    output_dim = all_truths.shape[1]
    rmse = np.sqrt(mse / output_dim)
    r2 = r2_score(all_truths, all_preds)
    pred_var = np.mean(np.var(all_preds, axis=0))
    truth_var = np.mean(np.var(all_truths, axis=0))

    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION (Best Model)")
    print(f"{'='*60}")
    print(f"  Test RMSE (physical): {rmse:.4f}")
    print(f"  Test MSE  (physical): {mse:.4f}")
    print(f"  R² Score:             {r2:.6f}")
    print(f"  Prediction Variance:  {pred_var:.4f}")
    print(f"  Truth Variance:       {truth_var:.4f}")
    print(f"  Variance Ratio:       {pred_var / truth_var:.4f}")
    print(f"{'='*60}\n")

    # --- 1. Learning Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, scaled)")
    ax.set_title("Learning Curves — Single Config (26 GHz, Tx4, Rx4, RIS32)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "single_config_loss_curve.png"), dpi=150)
    plt.close(fig)

    # --- 2. Prediction vs Truth ---
    num_samples = min(100, all_truths.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(all_truths[:num_samples, 0], 'o-', ms=4, label="True y_real[0]", alpha=0.8)
    axes[0].plot(all_preds[:num_samples, 0], 'x--', ms=4, label="Pred y_real[0]", alpha=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Prediction vs Truth — Test Set (RMSE={rmse:.3f}, R²={r2:.4f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(all_truths[:num_samples, output_dim // 2], 'o-', ms=4, label="True y_imag[0]", alpha=0.8)
    axes[1].plot(all_preds[:num_samples, output_dim // 2], 'x--', ms=4, label="Pred y_imag[0]", alpha=0.8)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "single_config_pred_vs_truth.png"), dpi=150)
    plt.close(fig)

    # --- 3. Residual Histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    flat_residuals = residuals.flatten()
    ax.hist(flat_residuals, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero')
    ax.set_xlabel("Residual (Predicted − True)")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution — μ={np.mean(flat_residuals):.3f}, σ={np.std(flat_residuals):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "single_config_residual_hist.png"), dpi=150)
    plt.close(fig)

    return rmse, r2, pred_var, truth_var


# -------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Single-config controlled experiment")
    parser.add_argument("--data_file", type=str, default="datasets/comprehensive_dataset.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    # ---- Load & Filter ----
    print(f"Loading dataset from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    print(f"Full dataset: {len(df)} rows")

    # Filter to single configuration
    freq_26ghz = 26e9
    mask = (
        (df['frequency'] == freq_26ghz) &
        (df['N_RIS'] == 32) &
        (df['N_Tx'] == 4) &
        (df['N_Rx'] == 4)
    )
    df_filtered = df[mask].copy()
    print(f"Filtered to 26 GHz, Tx=4, Rx=4, RIS=32: {len(df_filtered)} samples")

    if len(df_filtered) == 0:
        print("ERROR: No samples found. Check filter values.")
        print(f"  Available frequencies: {sorted(df['frequency'].unique())}")
        print(f"  Available N_Tx: {sorted(df['N_Tx'].unique())}")
        print(f"  Available N_Rx: {sorted(df['N_Rx'].unique())}")
        print(f"  Available N_RIS: {sorted(df['N_RIS'].unique())}")
        return

    # ---- Statistics ----
    print(f"\n{'='*60}")
    print("FILTERED DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  Samples: {len(df_filtered)}")
    for col in ['frequency', 'wavelength', 'dx', 'dy', 'SNR_dB']:
        if col in df_filtered.columns:
            print(f"  {col}: min={df_filtered[col].min():.4g}, max={df_filtered[col].max():.4g}, unique={df_filtered[col].nunique()}")
    print(f"{'='*60}\n")

    # ---- Split ----
    df_train, df_temp = train_test_split(df_filtered, test_size=0.3, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    print(f"Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    scaler = StandardScaler()
    y_scaler = StandardScaler()

    print("\nBuilding datasets...")
    train_dataset = RISDatasetSingleConfig(df_train, scaler=scaler, fit_scaler=True, y_scaler=y_scaler, fit_y_scaler=True)

    print(f"\n  Target Mean: {y_scaler.mean_}")
    print(f"  Target Std:  {np.sqrt(y_scaler.var_)}")

    val_dataset = RISDatasetSingleConfig(df_val, scaler=scaler, y_scaler=y_scaler)
    test_dataset = RISDatasetSingleConfig(df_test, scaler=scaler, y_scaler=y_scaler)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    input_dim = train_dataset[0][0].shape[0]
    output_dim = train_dataset[0][1].shape[0]

    print(f"\n  Input Dim:  {input_dim}")
    print(f"  Output Dim: {output_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = RISBaselineModel(input_dim=input_dim, output_dim=output_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    models_dir = os.path.join(project_dir, "models")
    plots_dir = os.path.join(project_dir, "plots")

    print(f"\n{'='*60}")
    print(f"TRAINING: {args.epochs} epochs, BS={args.batch_size}, LR={args.lr}, Patience={args.patience}")
    print(f"{'='*60}")

    train_losses, val_losses = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        save_dir=models_dir, y_scaler=y_scaler, patience=args.patience
    )

    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(os.path.join(models_dir, "model_single_config_best.pth"), map_location=device))

    rmse, r2, pred_var, truth_var = generate_plots(
        model, test_loader, device, plots_dir, y_scaler, train_losses, val_losses
    )

    print("Single-config experiment completed successfully!")


if __name__ == "__main__":
    main()
