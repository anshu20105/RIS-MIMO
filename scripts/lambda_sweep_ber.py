import subprocess
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

lambdas = [0.01, 0.05, 0.1, 0.5]
results = []

print("Starting Lambda Sweep for BER Physics Loss (running with fixed lambda_se=0.5)...")

for l_ber in lambdas:
    print(f"\n--- Running PINN with lambda_ber = {l_ber} ---")
    cmd = ["python3", "-u", "scripts/train_multitask_digital_twin.py", "--pinn", "--lambda_se", "0.5", "--lambda_ber", str(l_ber)]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    out = process.stdout
    
    # Check for success
    if process.returncode != 0:
        print(f"Error running lambda {l_ber}:\n{process.stderr}")
        continue
        
    try:
        epochs = re.findall(r"Epoch \d+ \| Train Data L: ([\d\.]+) \| Train Phys L: ([\d\.]+) \| Val L: ([\d\.]+)", out)
        if epochs:
            train_data = float(epochs[-1][0])
            train_phys = float(epochs[-1][1])
            val_total = float(epochs[-1][2])
            train_total = train_data + train_phys
        else:
            train_data = train_phys = train_total = val_total = 0.0

        r2_se = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: ([\d\.-]+)", out).group(1))
        r2_ber = float(re.search(r"\[PINN\] Log\(BER\)           \| R2: ([\d\.-]+)", out).group(1))
        
        rmse_se = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: [\d\.-]+ \| RMSE: ([\d\.]+)", out).group(1))
        rmse_ber = float(re.search(r"\[PINN\] Log\(BER\)           \| R2: [\d\.-]+ \| RMSE: ([\d\.]+)", out).group(1))
        
        mae_se = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: [\d\.-]+ \| RMSE: [\d\.]+ \| MAE: ([\d\.]+)", out).group(1))
        mae_ber = float(re.search(r"\[PINN\] Log\(BER\)           \| R2: [\d\.-]+ \| RMSE: [\d\.]+ \| MAE: ([\d\.]+)", out).group(1))
        
        violations_se = int(re.search(r"Physical Constraint Violations SE \(Test Set\): (\d+)", out).group(1))
        violations_ber = int(re.search(r"Physical Constraint Violations BER \(Test Set\): (\d+)", out).group(1))
        
        res = {
            "lambda_ber": l_ber,
            "Train Data L": train_data,
            "Train Phys L": train_phys,
            "Train Total": train_total,
            "Val Total L": val_total,
            "SE R2": r2_se,
            "BER R2": r2_ber,
            "SE RMSE": rmse_se,
            "BER RMSE": rmse_ber,
            "SE MAE": mae_se,
            "BER MAE": mae_ber,
            "Violations SE": violations_se,
            "Violations BER": violations_ber
        }
        results.append(res)
        print(f"Completed! SE R2 = {r2_se:.4f}, BER R2 = {r2_ber:.4f}")
        print(f"Violations SE = {violations_se}, Violations BER = {violations_ber}")
        
    except Exception as e:
        print(f"Could not parse output for lambda {l_ber}. Error: {e}")

# --------------------------
# Plotting
# --------------------------
df = pd.DataFrame(results)

os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# 1. R2 vs lambda
axes[0].plot(df['lambda_ber'], df['BER R2'], marker='o', linewidth=2, color='purple', label='BER R²')
axes[0].plot(df['lambda_ber'], df['SE R2'], marker='^', linewidth=2, color='blue', label='SE R²', linestyle='--')
axes[0].set_title('Predictive R² vs. $\lambda_{BER}$')
axes[0].set_xlabel('$\lambda_{BER}$')
axes[0].set_ylabel('Test R² Score')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 2. BER Violations vs lambda
axes[1].plot(df['lambda_ber'], df['Violations BER'], marker='s', linewidth=2, color='red')
axes[1].set_title('BER Monotonicity Violations vs. $\lambda_{BER}$')
axes[1].set_xlabel('$\lambda_{BER}$')
axes[1].set_ylabel('# of BER Monotonicity Violations')
axes[1].grid(True, alpha=0.3)

# 3. SE Violations vs lambda (To check if fixing BER breaks SE)
axes[2].plot(df['lambda_ber'], df['Violations SE'], marker='D', linewidth=2, color='orange')
axes[2].set_title('SE Limits Violations vs. $\lambda_{BER}$')
axes[2].set_xlabel('$\lambda_{BER}$')
axes[2].set_ylabel('# of SE Size Limit Violations')
axes[2].grid(True, alpha=0.3)

# 4. Loss Components vs lambda
axes[3].plot(df['lambda_ber'], df['Train Data L'], marker='^', label='Train Data Loss')
axes[3].plot(df['lambda_ber'], df['Train Phys L'], marker='v', label='Train Physics Loss (weighted)')
axes[3].set_title('Loss Components vs. $\lambda_{BER}$')
axes[3].set_xlabel('$\lambda_{BER}$')
axes[3].set_ylabel('MSE Loss')
axes[3].set_yscale('log')
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.savefig("plots/lambda_ber_sweep.png", dpi=150)
print("\nPlot saved successfully to plots/lambda_ber_sweep.png")

print("\nMarkdown Table:\n")
md = df.to_markdown(index=False) if hasattr(df, "to_markdown") else str(df)
print(md)
