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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Dataset Loader & Preprocessing
# -------------------------------------------------------------

class RISDataset(Dataset):
    """
    Custom PyTorch Dataset for the RIS-MIMO PINN Data.
    Features: frequency, wavelength, N_Tx, N_Rx, N_RIS, dx, dy, SNR_dB, 
              phase_shift_real, phase_shift_imag, H_real, H_imag, x_real, x_imag
    Labels: y_real, y_imag
    """
    def __init__(self, df, scaler=None, fit_scaler=False, y_scaler=None, fit_y_scaler=False):
        """
        Args:
            df (pd.DataFrame): The raw dataframe loaded from CSV.
            scaler (StandardScaler): Scikit-learn scaler for input features.
            fit_scaler (bool): If True, it will fit the scaler to the data.
            y_scaler (StandardScaler): Scaler for target outputs.
            fit_y_scaler (bool): If True, fits target scaler to the data.
        """
        self.df = df.reset_index(drop=True)
        self.scaler = scaler
        
        # Parse string representations back into numerical vectors.
        t0 = time.time()
        print(f"Parsing {len(self.df)} rows of string arrays...")
        
        # Columns that are stored as strings of lists
        array_cols = [
            'phase_shift_real', 'phase_shift_imag', 
            'H_real', 'H_imag', 
            'x_real', 'x_imag', 
            'y_real', 'y_imag'
        ]
        
        for col in array_cols:
            if col in self.df.columns and isinstance(self.df[col].iloc[0], str):
                self.df[col] = self.df[col].apply(ast.literal_eval)
        
        print(f"Parsing completed in {time.time() - t0:.1f}s")
        
        # 1. Scalar System Parameters (Scalars per instance)
        scalar_cols = ['frequency', 'wavelength', 'N_Tx', 'N_Rx', 'N_RIS', 'dx', 'dy', 'SNR_dB']
        X_scalars = self.df[scalar_cols].values # Shape: (N, 8)
        
        # 2. Array Parameters — zero-pad to max length per column
        #    (different antenna configs produce different array sizes)
        def extract_padded(col_name, pad_length=None):
            """Flatten each row and zero-pad to uniform length."""
            arrays = [np.array(row).flatten() for row in self.df[col_name]]
            if pad_length is None:
                pad_length = max(len(a) for a in arrays)
            padded = np.zeros((len(arrays), pad_length), dtype=np.float64)
            for i, a in enumerate(arrays):
                padded[i, :len(a)] = a
            return padded, pad_length

        # Feature array columns and their padded lengths
        feature_array_cols = [
            'phase_shift_real', 'phase_shift_imag',
            'H_real', 'H_imag',
            'x_real', 'x_imag'
        ]
        target_array_cols = ['y_real', 'y_imag']
        
        feature_arrays = []
        self.pad_lengths = {}
        for col in feature_array_cols:
            arr, plen = extract_padded(col)
            feature_arrays.append(arr)
            self.pad_lengths[col] = plen
            
        # 3. Targets (y_real, y_imag)
        target_arrays = []
        for col in target_array_cols:
            arr, plen = extract_padded(col)
            target_arrays.append(arr)
            self.pad_lengths[col] = plen
        
        print(f"Padded array dimensions: { {k: v for k, v in self.pad_lengths.items()} }")
        
        # 4. Concatenate features and targets
        self.X_raw = np.hstack([X_scalars] + feature_arrays)
        self.Y_raw = np.hstack(target_arrays)
        
        # Input Normalization
        if fit_scaler and self.scaler is not None:
            self.X = self.scaler.fit_transform(self.X_raw)
            print("Fitted and transformed inputs via StandardScaler.")
        elif self.scaler is not None:
            self.X = self.scaler.transform(self.X_raw)
            print("Transformed inputs via pre-fitted StandardScaler.")
        else:
            self.X = self.X_raw
            
        # Target Normalization
        if fit_y_scaler and y_scaler is not None:
            self.Y = y_scaler.fit_transform(self.Y_raw)
            print("Fitted and transformed targets via y_scaler.")
        elif y_scaler is not None:
            self.Y = y_scaler.transform(self.Y_raw)
            print("Transformed targets via pre-fitted y_scaler.")
        else:
            self.Y = self.Y_raw 
        
        # Convert to PyTorch tensors
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.Y_tensor = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.Y_tensor[idx]


# -------------------------------------------------------------
# 2. Architecture Definition
# -------------------------------------------------------------

class RISBaselineModel(nn.Module):
    """
    Fully Connected Neural Network Baseline for PINN.
    Input: System params + flattened RIS phases, cascaded H, and tx signal x.
    Output: Flattened received signal y.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128, 64]):
        super(RISBaselineModel, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # Build Hidden Layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU()) # GELU performs well for physical continuous signals
            layers.append(nn.Dropout(0.1)) # Small dropout for regularization
            in_dim = h_dim
            
        # Output Layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------------------
# 3. Training & Evaluation Engine
# -------------------------------------------------------------

def evaluate(model, dataloader, criterion, device, y_scaler=None):
    """Evaluate model and return scaled loss, physical RMSE, and prediction variance."""
    model.eval()
    total_loss = 0.0
    total_mse_phys = 0.0
    total_samples = 0
    all_preds_scaled = []
    
    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            batch_size = batch_X.size(0)
            total_loss += loss.item() * batch_size
            
            preds_np = predictions.cpu().numpy()
            targets_np = batch_Y.cpu().numpy()
            all_preds_scaled.append(preds_np)
            
            if y_scaler is not None:
                preds_phys = y_scaler.inverse_transform(preds_np)
                targets_phys = y_scaler.inverse_transform(targets_np)
            else:
                preds_phys = preds_np
                targets_phys = targets_np
                
            total_mse_phys += np.sum((preds_phys - targets_phys)**2)
            total_samples += batch_size
            
    avg_loss = total_loss / total_samples
    rmse_phys = np.sqrt(total_mse_phys / (total_samples * preds_np.shape[1]))
    
    all_preds_scaled = np.vstack(all_preds_scaled)
    pred_var = np.mean(np.var(all_preds_scaled, axis=0))
    
    return avg_loss, rmse_phys, pred_var

def train(model, train_loader, val_loader, epochs, lr, device, save_dir, 
          y_scaler=None, patience=10):
    """
    Train with early stopping. PINN-compatible: physics_loss can be added
    as an additional term inside the training loop in the future.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mse_phys = 0.0
        total_samples = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # Data-driven loss (MSE on scaled targets)
            data_loss = criterion(predictions, batch_Y)
            
            # -----------------------------------------------
            # PINN Hook: Add physics_loss here in the future
            # total_loss = data_loss + lambda_phys * physics_loss
            # -----------------------------------------------
            total_loss = data_loss
            
            total_loss.backward()
            optimizer.step()
            
            batch_size = batch_X.size(0)
            running_loss += data_loss.item() * batch_size
            total_samples += batch_size
            
            # Physical-unit RMSE tracking
            preds_np = predictions.detach().cpu().numpy()
            targets_np = batch_Y.detach().cpu().numpy()
            if y_scaler is not None:
                preds_phys = y_scaler.inverse_transform(preds_np)
                targets_phys = y_scaler.inverse_transform(targets_np)
            else:
                preds_phys = preds_np
                targets_phys = targets_np
                
            running_mse_phys += np.sum((preds_phys - targets_phys)**2)
            
        epoch_train_loss = running_loss / total_samples
        output_dim = preds_np.shape[1]
        train_rmse_phys = np.sqrt(running_mse_phys / (total_samples * output_dim))
        
        epoch_val_loss, val_rmse_phys, pred_var = evaluate(
            model, val_loader, criterion, device, y_scaler=y_scaler
        )
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        # Step the LR scheduler
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f} | "
              f"Train RMSE: {train_rmse_phys:.4f}, Val RMSE: {val_rmse_phys:.4f} | "
              f"Pred Var: {pred_var:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "model_baseline_best.pth"))
            print(f"  -> New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                print(f"Best model was at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
                break
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "model_baseline_final.pth"))
    
    return train_losses, val_losses

# -------------------------------------------------------------
# 4. Visualization & Post-Processing
# -------------------------------------------------------------

def plot_learning_curves(train_losses, val_losses, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss (MSE)")
    plt.plot(val_losses, label="Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline Model Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

def plot_predictions(model, test_loader, device, save_dir, y_scaler=None, num_samples=50):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_truths = []
    
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy()
            all_preds.append(preds)
            all_truths.append(batch_Y.numpy())
            
    all_preds = np.vstack(all_preds)
    all_truths = np.vstack(all_truths)
    
    if y_scaler is not None:
        all_preds = y_scaler.inverse_transform(all_preds)
        all_truths = y_scaler.inverse_transform(all_truths)
        
    rmse = np.sqrt(np.mean((all_preds - all_truths)**2) / all_truths.shape[1])
    mse = np.mean((all_preds - all_truths)**2)
    print(f"Overall Test RMSE (Physical Units): {rmse:.6f}")
    print(f"Overall Test MSE  (Physical Units): {mse:.6f}")
    
    # Prediction variance in physical units
    pred_var = np.mean(np.var(all_preds, axis=0))
    truth_var = np.mean(np.var(all_truths, axis=0))
    print(f"Prediction Variance: {pred_var:.4f}")
    print(f"Ground Truth Variance: {truth_var:.4f}")
    print(f"Variance Ratio (pred/truth): {pred_var / truth_var:.4f}")
    
    plt.figure(figsize=(12, 6))
    limit = min(num_samples, all_truths.shape[0])
    plt.plot(all_truths[:limit, 0], 'o-', label="True $y_{real}[0]$")
    plt.plot(all_preds[:limit, 0], 'x--', label="Pred $y_{real}[0]$")
    plt.xlabel("Sample Index")
    plt.ylabel("Signal Amplitude")
    plt.title("Prediction vs Ground Truth for Test Samples (Physical Units)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "prediction_vs_truth.png"))
    plt.close()


# -------------------------------------------------------------
# 5. Main Execution Script
# -------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train baseline PINN model for RIS Dataset")
    parser.add_argument("--data_file", type=str, default="datasets/comprehensive_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()
    
    dataset_path = args.data_file
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        print("Please specify a valid dataset path via --data_file")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    t0 = time.time()
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows in {time.time() - t0:.1f}s")
    
    # ---- Dataset Statistics ----
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    scalar_cols = ['frequency', 'wavelength', 'N_Tx', 'N_Rx', 'N_RIS', 'dx', 'dy', 'SNR_dB']
    for col in scalar_cols:
        if col in df.columns:
            print(f"  {col}: min={df[col].min():.4g}, max={df[col].max():.4g}, unique={df[col].nunique()}")
    
    print(f"\nUnique RIS configs (N_Tx, N_Rx, N_RIS):")
    configs = df.groupby(['N_Tx', 'N_Rx', 'N_RIS']).size().reset_index(name='count')
    for _, row in configs.iterrows():
        print(f"  Tx={int(row['N_Tx'])}, Rx={int(row['N_Rx'])}, RIS={int(row['N_RIS'])}: {row['count']} samples")
    print("="*60 + "\n")
    
    # Shuffle and split Dataframe
    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    
    print(f"Dataset Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    print("\nBuilding Training Dataset Wrapper...")
    train_dataset = RISDataset(df_train, scaler=scaler, fit_scaler=True, y_scaler=y_scaler, fit_y_scaler=True)
    
    print("\n--- Target Scaling Diagnostics ---")
    print(f"Target Mean (per dim): {y_scaler.mean_}")
    print(f"Target Std  (per dim): {np.sqrt(y_scaler.var_)}")
    print(f"Target Var  (per dim): {y_scaler.var_}")
    print(f"Scaled Y Train mean: {np.mean(train_dataset.Y):.6f}, var: {np.var(train_dataset.Y):.6f}")
    
    print("\nBuilding Validation Dataset Wrapper...")
    val_dataset = RISDataset(df_val, scaler=scaler, fit_scaler=False, y_scaler=y_scaler, fit_y_scaler=False)
    
    print("\nBuilding Test Dataset Wrapper...")
    test_dataset = RISDataset(df_test, scaler=scaler, fit_scaler=False, y_scaler=y_scaler, fit_y_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    input_dim = train_dataset[0][0].shape[0]
    output_dim = train_dataset[0][1].shape[0]
    
    print(f"\nNetwork Input Dimension: {input_dim}")
    print(f"Network Output Dimension: {output_dim}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RISBaselineModel(input_dim=input_dim, output_dim=output_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Setup directories relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    models_dir = os.path.join(project_dir, "models")
    plots_dir = os.path.join(project_dir, "plots")
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}, Patience: {args.patience}")
    print("="*60)
    
    train_losses, val_losses = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=args.epochs, 
        lr=args.lr, 
        device=device,
        save_dir=models_dir,
        y_scaler=y_scaler,
        patience=args.patience
    )
    
    print("\nPlotting learning curves...")
    plot_learning_curves(train_losses, val_losses, save_dir=plots_dir)
    
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION (Best Model)")
    print("="*60)
    model.load_state_dict(torch.load(os.path.join(models_dir, "model_baseline_best.pth"), map_location=device))
    plot_predictions(model, test_loader, device=device, save_dir=plots_dir, y_scaler=y_scaler)
    
    print("\nTraining Script completed successfully!")

if __name__ == "__main__":
    main()
