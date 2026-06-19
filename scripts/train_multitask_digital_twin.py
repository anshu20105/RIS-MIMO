"""
Stage 3, 4, 5: Multi-Task PyTorch Digital Twin Model Training (Baseline vs PINN)
Predicts: [y_power, Spectral Efficiency, BER] from RIS-MIMO physical configs.
"""

import ast
import time
import os
import argparse
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
# Global Settings
# ==========================================
DATA_FILE = "datasets/digital_twin_dataset.csv"
PLOTS_DIR = "plots"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 150
PATIENCE = 15  # Early stopping patience
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# Phase 1, 2 & 3 PINN Physics Losses
# ==========================================
def pinn_loss_se_limits(se_pred_unscaled, x_raw):
    """PHYSICS CONSTRAINT 1: SE bounded by Shannon theoretical limit."""
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
    """PHYSICS CONSTRAINT 2: Monotonic inverse relationship SE vs BER."""
    batch_sz = se_pred_unscaled.size(0)
    if batch_sz < 2:
        return torch.tensor(0.0, device=DEVICE), 0
        
    idx = torch.randperm(batch_sz, device=DEVICE)
    se_shuffled = se_pred_unscaled[idx]
    ber_shuffled = ber_pred_unscaled[idx]
    
    delta_se = se_pred_unscaled - se_shuffled
    delta_ber = ber_pred_unscaled - ber_shuffled
    
    # Product > 0 implies both went same direction = violation of inverse monotonicity
    product = delta_se * delta_ber
    loss = nn.ReLU()(product)
    
    return loss.mean(), (product > 0).sum().item()

def pinn_loss_y_power_consistency(yp_pred_unscaled, se_pred_unscaled, ber_pred_unscaled):
    """
    PHYSICS CONSTRAINT 3: y_power consistency.
    Higher y_power -> Higher SE (positive correlation)
    Higher y_power -> Lower BER (inverse correlation)
    """
    batch_sz = yp_pred_unscaled.size(0)
    if batch_sz < 2:
        return torch.tensor(0.0, device=DEVICE), 0
    
    idx = torch.randperm(batch_sz, device=DEVICE)
    yp_shuffled = yp_pred_unscaled[idx]
    se_shuffled = se_pred_unscaled[idx]
    ber_shuffled = ber_pred_unscaled[idx]
    
    delta_yp = yp_pred_unscaled - yp_shuffled
    delta_se = se_pred_unscaled - se_shuffled
    delta_ber = ber_pred_unscaled - ber_shuffled
    
    # Rule 1: y_power and SE must move in the same direction (positive correlation)
    # Violation: product is negative (one goes up, other goes down)
    product_se = delta_yp * delta_se
    loss_se = nn.ReLU()(-product_se)
    violations_se = (product_se < 0).sum().item()
    
    # Rule 2: y_power and BER must move in opposite directions (inverse correlation)
    # Violation: product is positive (both go same direction)
    product_ber = delta_yp * delta_ber
    loss_ber = nn.ReLU()(product_ber)
    violations_ber = (product_ber > 0).sum().item()
    
    total_loss = loss_se.mean() + loss_ber.mean()
    total_violations = violations_se + violations_ber
    
    return total_loss, total_violations

# ==========================================
# Dataset Loader
# ==========================================
class MultiTaskRISDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading dataset from {csv_file}...")
        start = time.time()
        self.df = pd.read_csv(csv_file)
        
        X_list = []
        for index, row in self.df.iterrows():
            base_features = [
                row['frequency'],
                row['N_Tx'],
                row['N_Rx'],
                row['N_RIS'],
                row['dx'],
                row['dy'],
                row['SNR_dB']
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
# Neural Network Architecture
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
# Unscaling Helper
# ==========================================
def unscale_tensors(pred_tensor, scaler):
    mean = torch.tensor(scaler.mean_, device=DEVICE, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, device=DEVICE, dtype=torch.float32)
    return pred_tensor * scale + mean

# ==========================================
# Training Pipeline
# ==========================================
def train_model(args):
    model_name = "PINN" if args.pinn else "Baseline"
    print(f"\n==============================================")
    print(f"Initializing {model_name} Multi-Task Training...")
    print(f"==============================================\n")
    
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
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = dataset.X_raw.shape[1]
    model = MultiTaskDigitalTwin(input_dim=input_dim).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {
        'train_data_loss': [], 'train_phys_loss': [], 'train_total_loss': [],
        'val_loss': [], 'val_violations_se': [], 'val_violations_ber': [], 'val_violations_yp': []
    }
    
    for epoch in range(EPOCHS):
        model.train()
        train_d_loss, train_p_loss, train_t_loss = 0.0, 0.0, 0.0
        
        for batch_x_scale, batch_x_raw, batch_yp, batch_se, batch_ber in train_loader:
            batch_x_scale, batch_x_raw = batch_x_scale.to(DEVICE), batch_x_raw.to(DEVICE)
            batch_yp, batch_se, batch_ber = batch_yp.to(DEVICE), batch_se.to(DEVICE), batch_ber.to(DEVICE)
            
            optimizer.zero_grad()
            yp_pred, se_pred, ber_pred = model(batch_x_scale)
            
            data_loss = criterion(yp_pred, batch_yp) + criterion(se_pred, batch_se) + criterion(ber_pred, batch_ber)
            
            if args.pinn:
                yp_pred_unscaled = unscale_tensors(yp_pred, scaler_yp)
                se_pred_unscaled = unscale_tensors(se_pred, scaler_se)
                ber_pred_unscaled = unscale_tensors(ber_pred, scaler_ber)
                
                loss_phys_se, _ = pinn_loss_se_limits(se_pred_unscaled, batch_x_raw)
                loss_phys_ber, _ = pinn_loss_ber_logic(se_pred_unscaled, ber_pred_unscaled)
                loss_phys_yp, _ = pinn_loss_y_power_consistency(yp_pred_unscaled, se_pred_unscaled, ber_pred_unscaled)
                
                loss_phys_total = (args.lambda_se * loss_phys_se) + (args.lambda_ber * loss_phys_ber) + (args.lambda_y * loss_phys_yp)
            else:
                loss_phys_total = torch.tensor(0.0)
            
            total_loss = data_loss + loss_phys_total
            
            total_loss.backward()
            optimizer.step()
            
            train_d_loss += data_loss.item()
            train_p_loss += loss_phys_total.item()
            train_t_loss += total_loss.item()
            
        train_d_loss /= len(train_loader)
        train_p_loss /= len(train_loader)
        train_t_loss /= len(train_loader)
        
        history['train_data_loss'].append(train_d_loss)
        history['train_phys_loss'].append(train_p_loss)
        history['train_total_loss'].append(train_t_loss)
        
        # Validation
        model.eval()
        val_loss, t_v_se, t_v_ber, t_v_yp = 0.0, 0, 0, 0
        
        with torch.no_grad():
            for batch_x_scale, batch_x_raw, batch_yp, batch_se, batch_ber in val_loader:
                batch_x_scale, batch_x_raw = batch_x_scale.to(DEVICE), batch_x_raw.to(DEVICE)
                batch_yp, batch_se, batch_ber = batch_yp.to(DEVICE), batch_se.to(DEVICE), batch_ber.to(DEVICE)
                
                yp_pred, se_pred, ber_pred = model(batch_x_scale)
                v_data_loss = criterion(yp_pred, batch_yp) + criterion(se_pred, batch_se) + criterion(ber_pred, batch_ber)
                val_loss += v_data_loss.item()
                
                yp_pred_unscaled = unscale_tensors(yp_pred, scaler_yp)
                se_pred_unscaled = unscale_tensors(se_pred, scaler_se)
                ber_pred_unscaled = unscale_tensors(ber_pred, scaler_ber)
                
                _, batch_v_se = pinn_loss_se_limits(se_pred_unscaled, batch_x_raw)
                _, batch_v_ber = pinn_loss_ber_logic(se_pred_unscaled, ber_pred_unscaled)
                _, batch_v_yp = pinn_loss_y_power_consistency(yp_pred_unscaled, se_pred_unscaled, ber_pred_unscaled)
                t_v_se += batch_v_se
                t_v_ber += batch_v_ber
                t_v_yp += batch_v_yp

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['val_violations_se'].append(t_v_se)
        history['val_violations_ber'].append(t_v_ber)
        history['val_violations_yp'].append(t_v_yp)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {(epoch+1):03d} | Train Data L: {train_d_loss:.4f} | Train Phys L: {train_p_loss:.6f} | Val L: {val_loss:.4f} | V_SE: {t_v_se} V_BER: {t_v_ber} V_YP: {t_v_yp}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{model_name.lower()}_best.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # -----------------------------
    # Evaluation
    # -----------------------------
    print(f"\nEvaluating Best {model_name} Model...")
    best_weights = torch.load(os.path.join(CHECKPOINT_DIR, f"{model_name.lower()}_best.pth"), map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_weights)
    model.eval()
    
    test_yp_true, test_yp_pred = [], []
    test_se_true, test_se_pred = [], []
    test_ber_true, test_ber_pred = [], []
    test_violations_se, test_violations_ber, test_violations_yp = 0, 0, 0
    
    with torch.no_grad():
        for batch_x_scale, batch_x_raw, batch_yp, batch_se, batch_ber in test_loader:
            batch_x_scale, batch_x_raw = batch_x_scale.to(DEVICE), batch_x_raw.to(DEVICE)
            yp_pred, se_pred, ber_pred = model(batch_x_scale)
            
            yp_pred_unscaled = unscale_tensors(yp_pred, scaler_yp)
            se_pred_unscaled = unscale_tensors(se_pred, scaler_se)
            ber_pred_unscaled = unscale_tensors(ber_pred, scaler_ber)
            
            _, batch_v_se = pinn_loss_se_limits(se_pred_unscaled, batch_x_raw)
            _, batch_v_ber = pinn_loss_ber_logic(se_pred_unscaled, ber_pred_unscaled)
            _, batch_v_yp = pinn_loss_y_power_consistency(yp_pred_unscaled, se_pred_unscaled, ber_pred_unscaled)
            test_violations_se += batch_v_se
            test_violations_ber += batch_v_ber
            test_violations_yp += batch_v_yp
            
            test_yp_true.extend(batch_yp.cpu().numpy())
            test_yp_pred.extend(yp_pred.cpu().numpy())
            test_se_true.extend(batch_se.cpu().numpy())
            test_se_pred.extend(se_pred.cpu().numpy())
            test_ber_true.extend(batch_ber.cpu().numpy())
            test_ber_pred.extend(ber_pred.cpu().numpy())
    
    test_yp_true_phys = scaler_yp.inverse_transform(np.array(test_yp_true).reshape(-1, 1))
    test_yp_pred_phys = scaler_yp.inverse_transform(np.array(test_yp_pred).reshape(-1, 1))
    test_se_true_phys = scaler_se.inverse_transform(np.array(test_se_true).reshape(-1, 1))
    test_se_pred_phys = scaler_se.inverse_transform(np.array(test_se_pred).reshape(-1, 1))
    test_ber_true_log = scaler_ber.inverse_transform(np.array(test_ber_true).reshape(-1, 1))
    test_ber_pred_log = scaler_ber.inverse_transform(np.array(test_ber_pred).reshape(-1, 1))
    
    yp_r2 = r2_score(test_yp_true_phys, test_yp_pred_phys)
    yp_rmse = np.sqrt(mean_squared_error(test_yp_true_phys, test_yp_pred_phys))
    yp_mae = mean_absolute_error(test_yp_true_phys, test_yp_pred_phys)
    
    se_r2 = r2_score(test_se_true_phys, test_se_pred_phys)
    se_rmse = np.sqrt(mean_squared_error(test_se_true_phys, test_se_pred_phys))
    se_mae = mean_absolute_error(test_se_true_phys, test_se_pred_phys)
    
    ber_r2 = r2_score(test_ber_true_log, test_ber_pred_log)
    ber_rmse = np.sqrt(mean_squared_error(test_ber_true_log, test_ber_pred_log))
    ber_mae = mean_absolute_error(test_ber_true_log, test_ber_pred_log)
    
    print("\n--- Physical Test Set Metrics ---")
    print(f"[{model_name}] y_power            | R2: {yp_r2:.4f} | RMSE: {yp_rmse:.4f} | MAE: {yp_mae:.4f}")
    print(f"[{model_name}] Spectral Efficiency | R2: {se_r2:.4f} | RMSE: {se_rmse:.4f} | MAE: {se_mae:.4f}")
    print(f"[{model_name}] Log(BER)           | R2: {ber_r2:.4f} | RMSE: {ber_rmse:.4f} | MAE: {ber_mae:.4f}")
    print(f"[{model_name}] Violations SE: {test_violations_se} | BER: {test_violations_ber} | y_power: {test_violations_yp}")
    
    np.save(os.path.join(CHECKPOINT_DIR, f"{model_name}_history.npy"), history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn", action="store_true", help="Enable PINN Physics Constraints")
    parser.add_argument("--lambda_se", type=float, default=0.5, help="Weight for SE physics loss")
    parser.add_argument("--lambda_ber", type=float, default=0.0, help="Weight for BER physics loss")
    parser.add_argument("--lambda_y", type=float, default=0.0, help="Weight for y_power consistency loss")
    args = parser.parse_args()
    
    train_model(args)
