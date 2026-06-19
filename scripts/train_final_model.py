"""
Final PINN Model Training Script
=================================
Trains the Multi-Task Digital Twin with recommended PINN configuration:
    lambda_SE  = 0.5
    lambda_BER = 0.5
    lambda_y   = 0.01

Runs Baseline and PINN sequentially, exports metrics to JSON, and saves
the best PINN checkpoint as checkpoints/final_pinn_best.pth.
"""

import ast
import json
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# Configuration
# ==========================================
DATA_FILE = "datasets/digital_twin_dataset.csv"
PLOTS_DIR = "plots/final"
CHECKPOINT_DIR = "checkpoints"
REPORTS_DIR = "reports"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 150
PATIENCE = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recommended PINN hyperparameters
LAMBDA_SE = 0.5
LAMBDA_BER = 0.5
LAMBDA_Y = 0.01

# ==========================================
# Physics Loss Functions
# ==========================================
def pinn_loss_se_limits(se_pred_unscaled, x_raw):
    """SE bounded by Shannon theoretical limit."""
    n_tx = x_raw[:, 1]
    n_rx = x_raw[:, 2]
    n_ris = x_raw[:, 3]
    snr_db = x_raw[:, 6]
    snr_linear = 10.0 ** (snr_db / 10.0)
    max_gain = snr_linear * n_tx * n_rx * (n_ris ** 2)
    se_bound = torch.log2(1.0 + max_gain) / n_tx
    diff = se_pred_unscaled.squeeze() - se_bound
    loss = nn.ReLU()(diff)
    return loss.mean(), (diff > 0).sum().item()

def pinn_loss_ber_logic(se_pred_unscaled, ber_pred_unscaled):
    """Monotonic inverse relationship SE vs BER."""
    batch_sz = se_pred_unscaled.size(0)
    if batch_sz < 2:
        return torch.tensor(0.0, device=DEVICE), 0
    idx = torch.randperm(batch_sz, device=DEVICE)
    delta_se = se_pred_unscaled - se_pred_unscaled[idx]
    delta_ber = ber_pred_unscaled - ber_pred_unscaled[idx]
    product = delta_se * delta_ber
    loss = nn.ReLU()(product)
    return loss.mean(), (product > 0).sum().item()

def pinn_loss_y_power_consistency(yp_pred_unscaled, se_pred_unscaled, ber_pred_unscaled):
    """y_power consistency: higher y_power -> higher SE, lower BER."""
    batch_sz = yp_pred_unscaled.size(0)
    if batch_sz < 2:
        return torch.tensor(0.0, device=DEVICE), 0
    idx = torch.randperm(batch_sz, device=DEVICE)
    delta_yp = yp_pred_unscaled - yp_pred_unscaled[idx]
    delta_se = se_pred_unscaled - se_pred_unscaled[idx]
    delta_ber = ber_pred_unscaled - ber_pred_unscaled[idx]
    product_se = delta_yp * delta_se
    loss_se = nn.ReLU()(-product_se)
    violations_se = (product_se < 0).sum().item()
    product_ber = delta_yp * delta_ber
    loss_ber = nn.ReLU()(product_ber)
    violations_ber = (product_ber > 0).sum().item()
    return loss_se.mean() + loss_ber.mean(), violations_se + violations_ber

# ==========================================
# Dataset
# ==========================================
class MultiTaskRISDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading dataset from {csv_file}...")
        start = time.time()
        self.df = pd.read_csv(csv_file)
        X_list = []
        for index, row in self.df.iterrows():
            base_features = [
                row['frequency'], row['N_Tx'], row['N_Rx'], row['N_RIS'],
                row['dx'], row['dy'], row['SNR_dB']
            ]
            p_real = ast.literal_eval(row['phase_shift_real'])
            p_imag = ast.literal_eval(row['phase_shift_imag'])
            pad_len = 128 - len(p_real)
            p_real_padded = np.pad(p_real, (0, pad_len), 'constant')
            p_imag_padded = np.pad(p_imag, (0, pad_len), 'constant')
            x_feat = np.concatenate((base_features, p_real_padded, p_imag_padded))
            X_list.append(x_feat)
        self.X_raw = np.array(X_list, dtype=np.float32)
        self.X_scaled = np.zeros_like(self.X_raw)
        self.y_power = self.df['y_power'].values.astype(np.float32).reshape(-1, 1)
        self.se = self.df['SE'].values.astype(np.float32).reshape(-1, 1)
        self.ber = self.df['BER'].values.astype(np.float32).reshape(-1, 1)
        print(f"Dataset loaded in {time.time()-start:.2f}s.")

    def __len__(self):
        return len(self.X_raw)

    def __getitem__(self, idx):
        return self.X_scaled[idx], self.X_raw[idx], self.y_power[idx], self.se[idx], self.ber[idx]

# ==========================================
# Neural Network
# ==========================================
class MultiTaskDigitalTwin(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskDigitalTwin, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU()
        )
        self.branch_y_power = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.branch_se = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.branch_ber = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        shared_features = self.trunk(x)
        return self.branch_y_power(shared_features), self.branch_se(shared_features), self.branch_ber(shared_features)

# ==========================================
# Helpers
# ==========================================
def unscale_tensors(pred_tensor, scaler):
    mean = torch.tensor(scaler.mean_, device=DEVICE, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, device=DEVICE, dtype=torch.float32)
    return pred_tensor * scale + mean

# ==========================================
# Training Pipeline
# ==========================================
def train_model(dataset, train_idx, val_idx, test_idx, scaler_X, scaler_yp, scaler_se, scaler_ber,
                pinn=False, lambda_se=0.0, lambda_ber=0.0, lambda_y=0.0, tag="Baseline"):
    print(f"\n{'='*60}")
    print(f"Training {tag} Model...")
    print(f"  PINN={pinn}, lambda_se={lambda_se}, lambda_ber={lambda_ber}, lambda_y={lambda_y}")
    print(f"{'='*60}\n")

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = dataset.X_raw.shape[1]
    model = MultiTaskDigitalTwin(input_dim=input_dim).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_phys_loss': []}
    ckpt_name = f"final_{tag.lower()}_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_d_loss, train_p_loss = 0.0, 0.0

        for batch_x_scale, batch_x_raw, batch_yp, batch_se, batch_ber in train_loader:
            batch_x_scale, batch_x_raw = batch_x_scale.to(DEVICE), batch_x_raw.to(DEVICE)
            batch_yp, batch_se, batch_ber = batch_yp.to(DEVICE), batch_se.to(DEVICE), batch_ber.to(DEVICE)

            optimizer.zero_grad()
            yp_pred, se_pred, ber_pred = model(batch_x_scale)
            data_loss = criterion(yp_pred, batch_yp) + criterion(se_pred, batch_se) + criterion(ber_pred, batch_ber)

            if pinn:
                yp_u = unscale_tensors(yp_pred, scaler_yp)
                se_u = unscale_tensors(se_pred, scaler_se)
                ber_u = unscale_tensors(ber_pred, scaler_ber)
                l_se, _ = pinn_loss_se_limits(se_u, batch_x_raw)
                l_ber, _ = pinn_loss_ber_logic(se_u, ber_u)
                l_yp, _ = pinn_loss_y_power_consistency(yp_u, se_u, ber_u)
                phys_loss = lambda_se * l_se + lambda_ber * l_ber + lambda_y * l_yp
            else:
                phys_loss = torch.tensor(0.0)

            total_loss = data_loss + phys_loss
            total_loss.backward()
            optimizer.step()
            train_d_loss += data_loss.item()
            train_p_loss += phys_loss.item()

        train_d_loss /= len(train_loader)
        train_p_loss /= len(train_loader)
        history['train_loss'].append(train_d_loss)
        history['train_phys_loss'].append(train_p_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x_scale, batch_x_raw, batch_yp, batch_se, batch_ber in val_loader:
                batch_x_scale = batch_x_scale.to(DEVICE)
                batch_yp, batch_se, batch_ber = batch_yp.to(DEVICE), batch_se.to(DEVICE), batch_ber.to(DEVICE)
                yp_pred, se_pred, ber_pred = model(batch_x_scale)
                v_loss = criterion(yp_pred, batch_yp) + criterion(se_pred, batch_se) + criterion(ber_pred, batch_ber)
                val_loss += v_loss.item()
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d} | Data L: {train_d_loss:.4f} | Phys L: {train_p_loss:.6f} | Val L: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, ckpt_name))
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ---- Evaluation ----
    print(f"\n  Evaluating best {tag} model...")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, ckpt_name), map_location=DEVICE, weights_only=True))
    model.eval()

    test_yp_true, test_yp_pred = [], []
    test_se_true, test_se_pred = [], []
    test_ber_true, test_ber_pred = [], []
    t_v_se, t_v_ber, t_v_yp = 0, 0, 0

    with torch.no_grad():
        for batch_x_scale, batch_x_raw, batch_yp, batch_se, batch_ber in test_loader:
            batch_x_scale, batch_x_raw = batch_x_scale.to(DEVICE), batch_x_raw.to(DEVICE)
            yp_pred, se_pred, ber_pred = model(batch_x_scale)

            yp_u = unscale_tensors(yp_pred, scaler_yp)
            se_u = unscale_tensors(se_pred, scaler_se)
            ber_u = unscale_tensors(ber_pred, scaler_ber)

            _, v_se = pinn_loss_se_limits(se_u, batch_x_raw)
            _, v_ber = pinn_loss_ber_logic(se_u, ber_u)
            _, v_yp = pinn_loss_y_power_consistency(yp_u, se_u, ber_u)
            t_v_se += v_se
            t_v_ber += v_ber
            t_v_yp += v_yp

            test_yp_true.extend(batch_yp.cpu().numpy())
            test_yp_pred.extend(yp_pred.cpu().numpy())
            test_se_true.extend(batch_se.cpu().numpy())
            test_se_pred.extend(se_pred.cpu().numpy())
            test_ber_true.extend(batch_ber.cpu().numpy())
            test_ber_pred.extend(ber_pred.cpu().numpy())

    # Inverse transform
    yp_true_phys = scaler_yp.inverse_transform(np.array(test_yp_true).reshape(-1, 1))
    yp_pred_phys = scaler_yp.inverse_transform(np.array(test_yp_pred).reshape(-1, 1))
    se_true_phys = scaler_se.inverse_transform(np.array(test_se_true).reshape(-1, 1))
    se_pred_phys = scaler_se.inverse_transform(np.array(test_se_pred).reshape(-1, 1))
    ber_true_phys = scaler_ber.inverse_transform(np.array(test_ber_true).reshape(-1, 1))
    ber_pred_phys = scaler_ber.inverse_transform(np.array(test_ber_pred).reshape(-1, 1))

    metrics = {
        'tag': tag,
        'yp_r2': float(r2_score(yp_true_phys, yp_pred_phys)),
        'yp_rmse': float(np.sqrt(mean_squared_error(yp_true_phys, yp_pred_phys))),
        'yp_mae': float(mean_absolute_error(yp_true_phys, yp_pred_phys)),
        'se_r2': float(r2_score(se_true_phys, se_pred_phys)),
        'se_rmse': float(np.sqrt(mean_squared_error(se_true_phys, se_pred_phys))),
        'se_mae': float(mean_absolute_error(se_true_phys, se_pred_phys)),
        'ber_r2': float(r2_score(ber_true_phys, ber_pred_phys)),
        'ber_rmse': float(np.sqrt(mean_squared_error(ber_true_phys, ber_pred_phys))),
        'ber_mae': float(mean_absolute_error(ber_true_phys, ber_pred_phys)),
        'violations_se': t_v_se,
        'violations_ber': t_v_ber,
        'violations_yp': t_v_yp,
        'best_val_loss': float(best_val_loss),
        'total_params': total_params,
    }

    print(f"\n  --- {tag} Test Set Metrics ---")
    print(f"  y_power  | R²: {metrics['yp_r2']:.4f} | RMSE: {metrics['yp_rmse']:.4f} | MAE: {metrics['yp_mae']:.4f}")
    print(f"  SE       | R²: {metrics['se_r2']:.4f} | RMSE: {metrics['se_rmse']:.4f} | MAE: {metrics['se_mae']:.4f}")
    print(f"  BER      | R²: {metrics['ber_r2']:.4f} | RMSE: {metrics['ber_rmse']:.4f} | MAE: {metrics['ber_mae']:.4f}")
    print(f"  Violations — SE: {t_v_se} | BER: {t_v_ber} | y_power: {t_v_yp}")

    return metrics, history, {
        'yp_true': yp_true_phys, 'yp_pred': yp_pred_phys,
        'se_true': se_true_phys, 'se_pred': se_pred_phys,
        'ber_true': ber_true_phys, 'ber_pred': ber_pred_phys,
    }

# ==========================================
# Main
# ==========================================
def main():
    print("=" * 60)
    print("FINAL MODEL TRAINING — RIS-MIMO PINN Digital Twin")
    print(f"Recommended Config: λ_SE={LAMBDA_SE}, λ_BER={LAMBDA_BER}, λ_y={LAMBDA_Y}")
    print("=" * 60)

    dataset = MultiTaskRISDataset(DATA_FILE)

    indices = np.arange(len(dataset))
    snr_bins = pd.qcut(dataset.df['SNR_dB'], q=4, labels=False, duplicates='drop')
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=snr_bins, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    scaler_X = StandardScaler()
    dataset.X_scaled[train_idx] = scaler_X.fit_transform(dataset.X_raw[train_idx])
    dataset.X_scaled[val_idx] = scaler_X.transform(dataset.X_raw[val_idx])
    dataset.X_scaled[test_idx] = scaler_X.transform(dataset.X_raw[test_idx])

    dataset.y_power = np.log10(dataset.y_power + 1e-10)
    dataset.ber = np.log10(dataset.ber + 1e-12)

    scaler_yp = StandardScaler()
    dataset.y_power[train_idx] = scaler_yp.fit_transform(dataset.y_power[train_idx])
    dataset.y_power[val_idx] = scaler_yp.transform(dataset.y_power[val_idx])
    dataset.y_power[test_idx] = scaler_yp.transform(dataset.y_power[test_idx])

    scaler_se = StandardScaler()
    dataset.se[train_idx] = scaler_se.fit_transform(dataset.se[train_idx])
    dataset.se[val_idx] = scaler_se.transform(dataset.se[val_idx])
    dataset.se[test_idx] = scaler_se.transform(dataset.se[test_idx])

    scaler_ber = StandardScaler()
    dataset.ber[train_idx] = scaler_ber.fit_transform(dataset.ber[train_idx])
    dataset.ber[val_idx] = scaler_ber.transform(dataset.ber[val_idx])
    dataset.ber[test_idx] = scaler_ber.transform(dataset.ber[test_idx])

    print(f"\nDataset: {len(dataset)} samples | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"Device: {DEVICE}")

    # Train Baseline
    baseline_metrics, baseline_hist, baseline_preds = train_model(
        dataset, train_idx, val_idx, test_idx,
        scaler_X, scaler_yp, scaler_se, scaler_ber,
        pinn=False, tag="Baseline"
    )

    # Train PINN (recommended config)
    pinn_metrics, pinn_hist, pinn_preds = train_model(
        dataset, train_idx, val_idx, test_idx,
        scaler_X, scaler_yp, scaler_se, scaler_ber,
        pinn=True, lambda_se=LAMBDA_SE, lambda_ber=LAMBDA_BER, lambda_y=LAMBDA_Y,
        tag="PINN"
    )

    # ---- Export Metrics ----
    all_metrics = {
        'config': {
            'lambda_se': LAMBDA_SE, 'lambda_ber': LAMBDA_BER, 'lambda_y': LAMBDA_Y,
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'patience': PATIENCE
        },
        'baseline': baseline_metrics,
        'pinn': pinn_metrics,
        'baseline_history': {k: [float(v) for v in vals] for k, vals in baseline_hist.items()},
        'pinn_history': {k: [float(v) for v in vals] for k, vals in pinn_hist.items()},
    }
    metrics_path = os.path.join(REPORTS_DIR, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # ---- Comparison Table ----
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<25} {'Baseline':>12} {'PINN':>12} {'Delta':>12}")
    print("-" * 61)
    for key, label in [('se_r2', 'SE R²'), ('ber_r2', 'BER R²'), ('yp_r2', 'y_power R²'),
                       ('se_rmse', 'SE RMSE'), ('ber_rmse', 'BER RMSE'), ('yp_rmse', 'y_power RMSE'),
                       ('violations_se', 'SE Violations'), ('violations_ber', 'BER Violations'),
                       ('violations_yp', 'y_power Violations')]:
        bv = baseline_metrics[key]
        pv = pinn_metrics[key]
        delta = pv - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<23} {bv:>12.4f} {pv:>12.4f} {sign}{delta:>11.4f}")
    print("=" * 80)

    # ---- Quick Plots ----
    # Training curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(baseline_hist['train_loss'], label='Baseline Train', linewidth=1.5)
    axes[0].plot(baseline_hist['val_loss'], label='Baseline Val', linewidth=1.5, linestyle='--')
    axes[0].plot(pinn_hist['train_loss'], label='PINN Train', linewidth=1.5)
    axes[0].plot(pinn_hist['val_loss'], label='PINN Val', linewidth=1.5, linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Curves: Baseline vs PINN', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pinn_hist['train_phys_loss'], label='Physics Loss', linewidth=1.5, color='red')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Physics Loss', fontsize=12)
    axes[1].set_title('PINN Physics Loss Convergence', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "final_training_curves.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Prediction scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, label in zip(axes, ['se', 'ber', 'yp'], ['Spectral Efficiency', 'log₁₀(BER)', 'log₁₀(y_power)']):
        true = pinn_preds[f'{key}_true'].flatten()
        pred = pinn_preds[f'{key}_pred'].flatten()
        r2 = pinn_metrics[f'{key}_r2']
        ax.scatter(true, pred, alpha=0.3, s=8, color='steelblue')
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
        ax.set_xlabel(f'True {label}', fontsize=11)
        ax.set_ylabel(f'Predicted {label}', fontsize=11)
        ax.set_title(f'{label} (R² = {r2:.4f})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "final_scatter_plots.png"), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {PLOTS_DIR}/")
    print("\n✓ Final model training complete!")

if __name__ == "__main__":
    main()
