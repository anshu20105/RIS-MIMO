"""
Visualizations
===============
Plotly-based interactive charts for the Digital Twin dashboard.
Includes: KDE signal distribution, SE/BER/Capacity sweeps,
Rx correlation (magnitude, phase, NxN heatmap + table), and tornado chart.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from inference import predict_sweep, generate_y_distribution, generate_rx_complex_samples

# Shared Plotly template
_TEMPLATE = "plotly_dark"

# Theme Colors
COLOR_PRIMARY   = "#FFFFFF"
COLOR_SECONDARY = "#A0A0B0"
COLOR_ACCENT_BG = "#18182D"
COLOR_CYAN      = "#00D2FF"
COLOR_GREEN     = "#48CFAD"
COLOR_PURPLE    = "#6C63FF"
COLOR_PINK      = "#FF6B6B"
COLOR_YELLOW    = "#FFCE54"


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
# A. KDE of received signal distribution |y|
# ------------------------------------------------------------------ #
def plot_y_kde(y_power):
    """
    KDE-based smooth probability density of |y| (replaces histogram).
    Uses scipy.stats.gaussian_kde on Rayleigh-distributed |y| samples.
    """
    samples = generate_y_distribution(y_power, n_samples=10_000)
    samples_valid = samples[samples > 0]

    kde = gaussian_kde(samples_valid, bw_method="scott")
    x_grid = np.linspace(samples_valid.min(), samples_valid.max(), 500)
    density = kde(x_grid)

    fig = go.Figure()

    # Shaded fill under the KDE curve
    fig.add_trace(go.Scatter(
        x=x_grid, y=density,
        fill="tozeroy",
        fillcolor="rgba(0, 210, 255, 0.12)",
        line=dict(color=COLOR_CYAN, width=2.5),
        name="KDE",
        mode="lines",
    ))

    # Vertical line at the mode (peak of density)
    peak_x = x_grid[np.argmax(density)]
    fig.add_vline(
        x=peak_x, line_dash="dot",
        line_color=COLOR_PURPLE, line_width=1.5,
        annotation_text=f"Peak: {peak_x:.3f}",
        annotation_font_color=COLOR_PURPLE,
    )

    _base_layout(fig, "Received Signal Distribution (KDE)", "|y| Amplitude", "Probability Density")
    return fig


# ------------------------------------------------------------------ #
# B. SE vs RIS Size
# ------------------------------------------------------------------ #
def plot_se_vs_ris(model, scalers, base_params):
    scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers
    ris_vals = [8, 16, 32, 64, 128]
    results = predict_sweep(model, scaler_X, scaler_yp, scaler_se, scaler_ber,
                            device, base_params, "n_ris", ris_vals)
    se_vals  = [r["se"] for r in results]
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
        yaxis2=dict(gridcolor="rgba(255,255,255,0.0)"),
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
    freq_vals   = [3.5e9, 6e9, 26e9]
    freq_labels = ["3.5 GHz", "6 GHz", "26 GHz"]
    results  = predict_sweep(model, scaler_X, scaler_yp, scaler_se, scaler_ber,
                             device, base_params, "freq", freq_vals)
    cap_vals  = [r["capacity_mbps"] for r in results]
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
# E. Rx Antenna Correlation: magnitude + phase subplots
# ------------------------------------------------------------------ #
def compute_rx_correlation_matrix(y_power, n_rx, n_samples=5000):
    """
    Compute the full NxN Rx correlation matrix R where R[m,n] = E[y_m * y_n*].
    Returns complex ndarray of shape (n_rx, n_rx).
    """
    y_samples = generate_rx_complex_samples(y_power, n_rx, n_samples)
    # R[m,n] = mean over samples of y_m * conj(y_n)
    R = (y_samples @ y_samples.conj().T) / n_samples
    return R


def plot_rx_correlation(y_power, n_rx, m, n):
    """
    Two-panel plot: correlation magnitude and phase between Rx antennas m and n.
    R_mn = E[y_m * conj(y_n)] computed over Monte-Carlo samples.
    """
    n_rx = max(n_rx, 2)
    m = min(m, n_rx - 1)
    n = min(n, n_rx - 1)

    R = compute_rx_correlation_matrix(y_power, n_rx, n_samples=8000)
    R_mn = R[m, n]
    mag   = abs(R_mn)
    phase = np.angle(R_mn, deg=True)

    # Also show all off-diagonal magnitudes |R_mn| for m≠n for context
    all_mag   = [abs(R[i, j]) for i in range(n_rx) for j in range(n_rx)]
    all_phase = [np.angle(R[i, j], deg=True) for i in range(n_rx) for j in range(n_rx)]
    labels    = [f"R[{i},{j}]" for i in range(n_rx) for j in range(n_rx)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Correlation Magnitude |R_mn|", "Correlation Phase ∠R_mn (°)"],
    )

    # — Magnitude bar chart —
    bar_colors = [COLOR_CYAN if (ii == m * n_rx + n or ii == n * n_rx + m) else
                  "rgba(108,99,255,0.45)" for ii in range(n_rx * n_rx)]
    fig.add_trace(go.Bar(
        x=labels, y=all_mag,
        marker_color=bar_colors,
        name="Magnitude",
        text=[f"{v:.3f}" for v in all_mag],
        textposition="outside",
        textfont=dict(size=9),
    ), row=1, col=1)

    # — Phase scatter —
    phase_colors = [COLOR_YELLOW if (ii == m * n_rx + n or ii == n * n_rx + m) else
                    "rgba(255,206,84,0.35)" for ii in range(n_rx * n_rx)]
    fig.add_trace(go.Bar(
        x=labels, y=all_phase,
        marker_color=phase_colors,
        name="Phase (°)",
    ), row=1, col=2)

    fig.update_layout(
        template=_TEMPLATE, height=420,
        title=dict(
            text=f"Rx Antenna Cross-Correlation  (R[{m},{n}]: |·| = {mag:.4f}, ∠ = {phase:.1f}°)",
            font=dict(size=15, color=COLOR_PRIMARY),
        ),
        showlegend=False,
        margin=dict(l=50, r=30, t=70, b=80),
        font=dict(family="Inter, sans-serif", color=COLOR_SECONDARY),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickangle=45),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis2=dict(gridcolor="rgba(255,255,255,0.05)", tickangle=45),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def plot_rx_correlation_heatmap(y_power, n_rx):
    """
    NxN heatmap of |R_mn| values for all Rx antenna pairs.
    """
    n_rx = max(n_rx, 2)
    R = compute_rx_correlation_matrix(y_power, n_rx, n_samples=8000)
    mag_matrix = np.abs(R)

    labels = [f"Rx{i}" for i in range(n_rx)]
    fig = go.Figure(go.Heatmap(
        z=mag_matrix,
        x=labels,
        y=labels,
        colorscale="Viridis",
        text=[[f"{mag_matrix[i, j]:.3f}" for j in range(n_rx)] for i in range(n_rx)],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="| R_mn |", titlefont=dict(color=COLOR_SECONDARY),
                      tickfont=dict(color=COLOR_SECONDARY)),
    ))
    fig.update_layout(
        template=_TEMPLATE, height=350,
        title=dict(text=f"Rx Correlation Heatmap ({n_rx}×{n_rx})", font=dict(size=15, color=COLOR_PRIMARY)),
        margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Inter, sans-serif", color=COLOR_SECONDARY),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig, mag_matrix, labels


# ------------------------------------------------------------------ #
# F. Tornado Chart for Sensitivity Analysis
# ------------------------------------------------------------------ #
def plot_tornado_chart(sensitivity_data: dict, target_metric: str = "SE"):
    """
    Horizontal diverging tornado chart of parameter impact on the chosen metric.
    Parameters are sorted by absolute delta (descending) — most impactful at top.
    """
    _METRIC_DISPLAY = {
        "SE":           "Spectral Efficiency (bits/s/Hz)",
        "BER":          "Bit Error Rate",
        "y_power":      "Received Power (linear)",
        "capacity_mbps":"Capacity (Mbps)",
        "sinr_db":      "SINR (dB)",
    }

    data = sensitivity_data.get(target_metric, [])
    if not data:
        fig = go.Figure()
        _base_layout(fig, f"No sensitivity data for {target_metric}")
        return fig

    sorted_d  = sorted(data, key=lambda x: abs(x[1]), reverse=True)
    names     = [x[0] for x in sorted_d]
    deltas    = [x[1] for x in sorted_d]

    # Rank label text
    rank_labels = [f"#{i+1}" for i in range(len(names))]

    # Colour: positive impact → cyan; zero/tiny → muted grey
    max_delta = max(abs(d) for d in deltas) if deltas else 1.0
    bar_colors = []
    for d in deltas:
        frac = abs(d) / max(max_delta, 1e-12)
        if frac > 0.6:
            bar_colors.append(COLOR_CYAN)
        elif frac > 0.3:
            bar_colors.append(COLOR_GREEN)
        elif frac > 0.1:
            bar_colors.append(COLOR_YELLOW)
        else:
            bar_colors.append(COLOR_PINK)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=deltas,
        orientation="h",
        marker_color=bar_colors,
        marker_line_color="rgba(0,0,0,0)",
        text=[f"Δ = {d:.4f}  {rank_labels[i]}" for i, d in enumerate(deltas)],
        textposition="outside",
        textfont=dict(size=10, color=COLOR_SECONDARY),
        name=target_metric,
    ))

    metric_label = _METRIC_DISPLAY.get(target_metric, target_metric)
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(
            text=f"Sensitivity Tornado — Impact on {metric_label}",
            font=dict(size=15, color=COLOR_PRIMARY),
        ),
        height=max(380, 50 * len(names) + 80),
        margin=dict(l=110, r=120, t=60, b=50),
        font=dict(family="Inter, sans-serif", color=COLOR_SECONDARY),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=f"Δ {metric_label}",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            autorange="reversed",  # highest rank at top
        ),
        showlegend=False,
    )
    # Add vertical zero line annotation
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    return fig


# ------------------------------------------------------------------ #
# G. Radial Bar Chart for Sensitivity (legacy — kept for sidebar card)
# ------------------------------------------------------------------ #
def plot_sensitivity_radial(sensitivity_data: dict, target_metric="SE"):
    data = sensitivity_data.get(target_metric, [])
    sorted_d = sorted(data, key=lambda x: abs(x[1]), reverse=True)

    names       = [x[0] for x in sorted_d]
    deltas      = [abs(x[1]) for x in sorted_d]
    max_val     = max(deltas) if deltas else 1.0
    percentages = [(d / max_val) * 100 for d in deltas]

    colors = [COLOR_PURPLE, COLOR_PINK, COLOR_CYAN, COLOR_GREEN, COLOR_YELLOW,
              "#A0A0B0", "#FF9FF3", "#A29BFE", "#FD79A8", "#00CEC9"]
    colors = colors[:len(names)]

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=percentages,
        theta=names,
        width=[0.8] * len(names),
        marker_color=colors,
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.9,
    ))

    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, max(percentages) + 5]),
            angularaxis=dict(rotation=90, direction="clockwise",
                             gridcolor="rgba(255,255,255,0.1)",
                             linecolor="rgba(0,0,0,0)"),
        ),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=40, r=40),
        height=300,
    )
    return fig


# ------------------------------------------------------------------ #
# H. Comparison Radar Chart
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

    max_vals   = [max(b, u, 1e-12) for b, u in zip(base_vals, user_vals)]
    base_norm  = [b / m for b, m in zip(base_vals, max_vals)]
    user_norm  = [u / m for u, m in zip(user_vals, max_vals)]

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
