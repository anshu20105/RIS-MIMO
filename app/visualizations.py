"""
Visualizations
===============
Plotly‑based interactive charts for the Digital Twin dashboard.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from inference import predict_sweep, generate_y_distribution

# Shared Plotly template
_TEMPLATE = "plotly_dark"
_COLORS = px.colors.qualitative.Set2

# New Theme Colors
COLOR_PRIMARY = "#FFFFFF"
COLOR_SECONDARY = "#A0A0B0"
COLOR_ACCENT_BG = "#18182D"

COLOR_CYAN = "#00D2FF"
COLOR_GREEN = "#48CFAD"
COLOR_PURPLE = "#6C63FF"
COLOR_PINK = "#FF6B6B"
COLOR_YELLOW = "#FFCE54"


def _base_layout(fig, title="", xaxis="", yaxis="", height=420):
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=title, font=dict(size=16, color=COLOR_PRIMARY)),
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
        font=dict(family="Inter, sans-serif", color=COLOR_SECONDARY),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    )
    return fig


# ------------------------------------------------------------------ #
# A. Histogram of received signal distribution |y|
# ------------------------------------------------------------------ #
def plot_y_histogram(y_power):
    samples = generate_y_distribution(y_power, n_samples=10_000)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=60,
        marker_color=COLOR_CYAN, opacity=0.85,
        name="|y| samples",
    ))
    sigma = np.sqrt(max(y_power, 1e-15) / 2.0)
    x_th = np.linspace(0, samples.max(), 200)
    pdf = (x_th / sigma**2) * np.exp(-x_th**2 / (2 * sigma**2))
    # Scale PDF to histogram counts
    bin_width = (samples.max() - samples.min()) / 60
    fig.add_trace(go.Scatter(
        x=x_th, y=pdf * len(samples) * bin_width,
        mode="lines", name="Rayleigh PDF",
        line=dict(color=COLOR_PURPLE, width=2.5),
    ))
    _base_layout(fig, "Received Signal |y| Distribution", "|y| Amplitude", "Count")
    return fig


# ------------------------------------------------------------------ #
# B. SE vs RIS Size
# ------------------------------------------------------------------ #
def plot_se_vs_ris(model, scalers, base_params):
    scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers
    ris_vals = [8, 16, 32, 64, 128]
    results = predict_sweep(model, scaler_X, scaler_yp, scaler_se, scaler_ber,
                            device, base_params, "n_ris", ris_vals)
    se_vals = [r["se"] for r in results]
    cap_vals = [r["capacity_mbps"] for r in results]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=ris_vals, y=se_vals, mode="lines+markers",
        name="SE (bits/s/Hz)", line=dict(color=COLOR_CYAN, width=3),
        marker=dict(size=10, symbol="diamond"),
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=ris_vals, y=cap_vals, name="Capacity (Mbps)",
        marker_color="rgba(0, 210, 255, 0.25)", width=5,
    ), secondary_y=True)

    fig.update_layout(
        template=_TEMPLATE, height=420,
        title=dict(text="Spectral Efficiency vs RIS Size", font=dict(size=16, color=COLOR_PRIMARY)),
        xaxis_title="Number of RIS Elements",
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(family="Inter, sans-serif", color=COLOR_SECONDARY),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.0)"), # Hide secondary grid
    )
    fig.update_yaxes(title_text="SE (bits/s/Hz)", secondary_y=False)
    fig.update_yaxes(title_text="Capacity (Mbps)", secondary_y=True)
    return fig


# ------------------------------------------------------------------ #
# C. BER vs SNR
# ------------------------------------------------------------------ #
def plot_ber_vs_snr(model, scalers, base_params):
    scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers
    snr_vals = [-10, -5, 0, 5, 10, 15, 20]
    results = predict_sweep(model, scaler_X, scaler_yp, scaler_se, scaler_ber,
                            device, base_params, "snr_db", snr_vals)
    ber_vals = [r["ber"] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snr_vals, y=ber_vals, mode="lines+markers",
        line=dict(color=COLOR_PINK, width=3),
        marker=dict(size=10, symbol="circle"),
        name="BER",
        fill="tozeroy", fillcolor="rgba(255,107,107,0.1)",
    ))
    fig.update_yaxes(type="log", title_text="BER")
    _base_layout(fig, "Bit Error Rate vs SNR", "SNR (dB)", "")
    fig.update_yaxes(title_text="BER (log scale)")
    return fig


# ------------------------------------------------------------------ #
# D. Capacity vs Frequency
# ------------------------------------------------------------------ #
def plot_capacity_vs_freq(model, scalers, base_params):
    scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers
    freq_vals = [3.5e9, 6e9, 26e9]
    freq_labels = ["3.5 GHz", "6 GHz", "26 GHz"]
    results = predict_sweep(model, scaler_X, scaler_yp, scaler_se, scaler_ber,
                            device, base_params, "freq", freq_vals)
    cap_vals = [r["capacity_mbps"] for r in results]
    sinr_vals = [r["sinr_db"] for r in results]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=freq_labels, y=cap_vals, name="Capacity (Mbps)",
        marker_color=[COLOR_PURPLE, COLOR_GREEN, COLOR_PINK],
        text=[f"{c:.1f}" for c in cap_vals], textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=freq_labels, y=sinr_vals, mode="lines+markers",
        name="Eff. SINR (dB)", line=dict(color=COLOR_YELLOW, width=3),
        marker=dict(size=12, symbol="star"),
    ), secondary_y=True)

    fig.update_layout(
        template=_TEMPLATE, height=420,
        title=dict(text="Channel Capacity vs Carrier Frequency", font=dict(size=16, color=COLOR_PRIMARY)),
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(family="Inter, sans-serif", color=COLOR_SECONDARY),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    fig.update_yaxes(title_text="Capacity (Mbps)", secondary_y=False)
    fig.update_yaxes(title_text="SINR (dB)", secondary_y=True)
    return fig


# ------------------------------------------------------------------ #
# E. Radial Bar Chart for Sensitivity
# ------------------------------------------------------------------ #
def plot_sensitivity_radial(sensitivity_data: dict, target_metric="SE"):
    """
    Renders the sensitivity data as a radial bar chart (polar bar), matching
    the right-panel visualization in the Stitch mockup.
    """
    # Extract data for the single target metric (e.g. Spectral Efficiency)
    data = sensitivity_data.get(target_metric, [])
    # Sort descending so the largest impact is mapped optimally
    sorted_d = sorted(data, key=lambda x: abs(x[1]), reverse=True)
    
    names = [x[0] for x in sorted_d]
    deltas = [abs(x[1]) for x in sorted_d]
    
    # Normalize deltas for the polar chart display to represent percentages (mockup shows 85%, 73%, etc.)
    max_val = max(deltas) if deltas else 1.0
    percentages = [(d / max_val) * 100 for d in deltas]
    
    # Color palette matching mockup (Blue/cyan, pink, cyan, green)
    colors = [COLOR_PURPLE, COLOR_PINK, COLOR_CYAN, COLOR_GREEN, COLOR_YELLOW, "#A0A0B0"]
    colors = colors[:len(names)]
    
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=percentages,
        theta=names,
        width=[0.8] * len(names),
        marker_color=colors,
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.9
    ))
    
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, max(percentages)+5]),
            angularaxis=dict(rotation=90, direction="clockwise", gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(0,0,0,0)")
        ),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=40, r=40),
        height=300
    )
    return fig


# ------------------------------------------------------------------ #
# F. Comparison radar chart
# ------------------------------------------------------------------ #
def plot_comparison_radar(baseline_metrics: dict, user_metrics: dict):
    """Radar chart comparing baseline vs user scenario."""
    categories = ["SE", "Capacity\n(Mbps)", "1/BER\n(quality)", "Power", "SINR\n(dB)"]

    def _safe(v):
        return max(float(v), 1e-12)

    base_vals = [
        _safe(baseline_metrics["se"]),
        _safe(baseline_metrics["capacity_mbps"]),
        1.0 / max(_safe(baseline_metrics["ber"]), 1e-12),
        _safe(baseline_metrics["y_power"]),
        _safe(baseline_metrics["sinr_db"]),
    ]
    user_vals = [
        _safe(user_metrics["se"]),
        _safe(user_metrics["capacity_mbps"]),
        1.0 / max(_safe(user_metrics["ber"]), 1e-12),
        _safe(user_metrics["y_power"]),
        _safe(user_metrics["sinr_db"]),
    ]

    max_vals = [max(b, u, 1e-12) for b, u in zip(base_vals, user_vals)]
    base_norm = [b / m for b, m in zip(base_vals, max_vals)]
    user_norm = [u / m for u, m in zip(user_vals, max_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=base_norm + [base_norm[0]], theta=categories + [categories[0]],
        fill="toself", name="Baseline",
        fillcolor="rgba(108,99,255,0.15)", line=dict(color=COLOR_PURPLE, width=2),
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_norm + [user_norm[0]], theta=categories + [categories[0]],
        fill="toself", name="Your Config",
        fillcolor="rgba(72, 207, 173,0.15)", line=dict(color=COLOR_GREEN, width=2),
    ))
    fig.update_layout(
        template=_TEMPLATE, height=400,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1.05], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        font=dict(family="Inter, sans-serif", color=COLOR_SECONDARY),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=40, b=60),
    )
    return fig
