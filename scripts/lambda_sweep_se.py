import subprocess
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

lambdas = [0.01, 0.05, 0.1, 0.5]
results = []

print("Starting Lambda Sweep for SE Physics Loss...")

for l_se in lambdas:
    print(f"\n--- Running PINN with lambda_se = {l_se} ---")
    cmd = ["python3", "-u", "scripts/train_multitask_digital_twin.py", "--pinn", "--lambda_se", str(l_se)]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    out = process.stdout
    
    # Check for success
    if process.returncode != 0:
        print(f"Error running lambda {l_se}:\n{process.stderr}")
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

        r2_val = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: ([\d\.-]+)", out).group(1))
        rmse_val = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: [\d\.-]+ \| RMSE: ([\d\.]+)", out).group(1))
        mae_val = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: [\d\.-]+ \| RMSE: [\d\.]+ \| MAE: ([\d\.]+)", out).group(1))
        violations = int(re.search(r"Physical Constraint Violations \(Test Set\): (\d+)", out).group(1))
        
        res = {
            "lambda_se": l_se,
            "Train Data L": train_data,
            "Train Phys L": train_phys,
            "Train Total": train_total,
            "Val Total L": val_total,
            "SE R2": r2_val,
            "SE RMSE": rmse_val,
            "SE MAE": mae_val,
            "Violations": violations
        }
        results.append(res)
        print(f"Completed! SE R2 = {r2_val}, Violations = {violations}")
        
    except Exception as e:
        print(f"Could not parse output for lambda {l_se}. Error: {e}")
        # print(out)

# --------------------------
# Plotting
# --------------------------
df = pd.DataFrame(results)

# Append Baseline limits for visual reference loosely
# Baseline stats (from run_baseline.log):
# Train Data: ~0.389, Train Phys: 0, Val L: ~0.58, 
# SE R2: 0.8741, RMSE: 0.7641, Violations: 180

os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. R2 vs lambda
axes[0].plot(df['lambda_se'], df['SE R2'], marker='o', linewidth=2, color='blue')
axes[0].axhline(y=0.8741, color='red', linestyle='--', label='Baseline R2 (0.874)')
axes[0].set_title('SE R² vs. Lambda')
axes[0].set_xlabel('$\lambda_{SE}$')
axes[0].set_ylabel('Spectral Efficiency R²')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 2. Violations vs lambda
axes[1].plot(df['lambda_se'], df['Violations'], marker='s', linewidth=2, color='orange')
axes[1].axhline(y=180, color='red', linestyle='--', label='Baseline Violations (180)')
axes[1].set_title('Constraint Violations vs. Lambda')
axes[1].set_xlabel('$\lambda_{SE}$')
axes[1].set_ylabel('# of Violations in Test Set')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# 3. Loss Components vs lambda
axes[2].plot(df['lambda_se'], df['Train Data L'], marker='^', label='Train Data Loss')
axes[2].plot(df['lambda_se'], df['Train Phys L'], marker='v', label='Train Physics Loss (weighted)')
axes[2].plot(df['lambda_se'], df['Val Total L'], marker='d', label='Validation Total Loss')
axes[2].set_title('Loss Components vs. Lambda')
axes[2].set_xlabel('$\lambda_{SE}$')
axes[2].set_ylabel('MSE Loss')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig("plots/lambda_se_sweep.png", dpi=150)
print("\nPlot saved successfully to plots/lambda_se_sweep.png")

print("\nMarkdown Table:\n")
md = df.to_markdown(index=False)
print(md)
