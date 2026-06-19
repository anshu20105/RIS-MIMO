import subprocess
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

lambdas = [0.01, 0.05, 0.1, 0.5]
results = []

print("Starting Lambda Sweep for y_power Physics Loss")
print("Fixed: lambda_se=0.5, lambda_ber=0.5\n")

for l_y in lambdas:
    print(f"--- Running PINN with lambda_y = {l_y} ---")
    cmd = [
        "python3", "-u", "scripts/train_multitask_digital_twin.py",
        "--pinn", "--lambda_se", "0.5", "--lambda_ber", "0.5", "--lambda_y", str(l_y)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    out = process.stdout
    
    if process.returncode != 0:
        print(f"Error running lambda {l_y}:\n{process.stderr}")
        continue
        
    try:
        epochs = re.findall(
            r"Epoch \d+ \| Train Data L: ([\d\.]+) \| Train Phys L: ([\d\.]+) \| Val L: ([\d\.]+)", out
        )
        if epochs:
            train_data = float(epochs[-1][0])
            train_phys = float(epochs[-1][1])
            val_total = float(epochs[-1][2])
        else:
            train_data = train_phys = val_total = 0.0

        yp_r2 = float(re.search(r"\[PINN\] y_power\s+\| R2: ([\d\.-]+)", out).group(1))
        se_r2 = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: ([\d\.-]+)", out).group(1))
        ber_r2 = float(re.search(r"\[PINN\] Log\(BER\)\s+\| R2: ([\d\.-]+)", out).group(1))
        
        yp_rmse = float(re.search(r"\[PINN\] y_power\s+\| R2: [\d\.-]+ \| RMSE: ([\d\.]+)", out).group(1))
        se_rmse = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: [\d\.-]+ \| RMSE: ([\d\.]+)", out).group(1))
        ber_rmse = float(re.search(r"\[PINN\] Log\(BER\)\s+\| R2: [\d\.-]+ \| RMSE: ([\d\.]+)", out).group(1))
        
        yp_mae = float(re.search(r"\[PINN\] y_power\s+\| R2: [\d\.-]+ \| RMSE: [\d\.]+ \| MAE: ([\d\.]+)", out).group(1))
        se_mae = float(re.search(r"\[PINN\] Spectral Efficiency \| R2: [\d\.-]+ \| RMSE: [\d\.]+ \| MAE: ([\d\.]+)", out).group(1))
        ber_mae = float(re.search(r"\[PINN\] Log\(BER\)\s+\| R2: [\d\.-]+ \| RMSE: [\d\.]+ \| MAE: ([\d\.]+)", out).group(1))
        
        v_match = re.search(r"Violations SE: (\d+) \| BER: (\d+) \| y_power: (\d+)", out)
        v_se = int(v_match.group(1))
        v_ber = int(v_match.group(2))
        v_yp = int(v_match.group(3))
        
        res = {
            "lambda_y": l_y,
            "Train Data L": train_data,
            "Train Phys L": train_phys,
            "Val L": val_total,
            "yp R2": yp_r2, "SE R2": se_r2, "BER R2": ber_r2,
            "yp RMSE": yp_rmse, "SE RMSE": se_rmse, "BER RMSE": ber_rmse,
            "yp MAE": yp_mae, "SE MAE": se_mae, "BER MAE": ber_mae,
            "V_SE": v_se, "V_BER": v_ber, "V_YP": v_yp
        }
        results.append(res)
        print(f"Done! yp R2={yp_r2:.4f} SE R2={se_r2:.4f} BER R2={ber_r2:.4f}")
        print(f"  Violations SE={v_se} BER={v_ber} YP={v_yp}\n")
        
    except Exception as e:
        print(f"Parse error for lambda {l_y}: {e}")

# --------------------------
# Plotting
# --------------------------
df = pd.DataFrame(results)
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Row 1: R2 scores
axes[0, 0].plot(df['lambda_y'], df['yp R2'], marker='o', linewidth=2, color='green')
axes[0, 0].set_title(r'y_power R$^2$ vs $\lambda_y$')
axes[0, 0].set_xlabel(r'$\lambda_y$')
axes[0, 0].set_ylabel(r'R$^2$')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df['lambda_y'], df['SE R2'], marker='^', linewidth=2, color='blue')
axes[0, 1].set_title(r'SE R$^2$ vs $\lambda_y$')
axes[0, 1].set_xlabel(r'$\lambda_y$')
axes[0, 1].set_ylabel(r'R$^2$')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(df['lambda_y'], df['BER R2'], marker='s', linewidth=2, color='purple')
axes[0, 2].set_title(r'BER R$^2$ vs $\lambda_y$')
axes[0, 2].set_xlabel(r'$\lambda_y$')
axes[0, 2].set_ylabel(r'R$^2$')
axes[0, 2].grid(True, alpha=0.3)

# Row 2: Violations
axes[1, 0].plot(df['lambda_y'], df['V_SE'], marker='D', linewidth=2, color='orange', label='SE')
axes[1, 0].set_title(r'SE Violations vs $\lambda_y$')
axes[1, 0].set_xlabel(r'$\lambda_y$')
axes[1, 0].set_ylabel('# Violations')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df['lambda_y'], df['V_BER'], marker='v', linewidth=2, color='red', label='BER')
axes[1, 1].set_title(r'BER Violations vs $\lambda_y$')
axes[1, 1].set_xlabel(r'$\lambda_y$')
axes[1, 1].set_ylabel('# Violations')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(df['lambda_y'], df['V_YP'], marker='p', linewidth=2, color='teal')
axes[1, 2].set_title(r'y_power Violations vs $\lambda_y$')
axes[1, 2].set_xlabel(r'$\lambda_y$')
axes[1, 2].set_ylabel('# Violations')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/lambda_y_sweep.png", dpi=150)
print("\nPlot saved to plots/lambda_y_sweep.png")

# Print table
print("\n--- Results Table ---")
for _, row in df.iterrows():
    print(f"\nlambda_y={row['lambda_y']}:")
    print(f"  R2:  yp={row['yp R2']:.4f}  SE={row['SE R2']:.4f}  BER={row['BER R2']:.4f}")
    print(f"  RMSE: yp={row['yp RMSE']:.4f}  SE={row['SE RMSE']:.4f}  BER={row['BER RMSE']:.4f}")
    print(f"  MAE:  yp={row['yp MAE']:.4f}  SE={row['SE MAE']:.4f}  BER={row['BER MAE']:.4f}")
    print(f"  Violations: SE={int(row['V_SE'])}  BER={int(row['V_BER'])}  YP={int(row['V_YP'])}")
