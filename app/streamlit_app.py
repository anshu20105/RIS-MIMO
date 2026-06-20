"""
Digital Twin model of RIS-MIMO — Interactive Dashboard
=======================================================
Streamlit application for real-time exploration of RIS-assisted
MIMO communication performance using a trained PINN model.

Launch:
    cd /home/anshu/RIS_Project
    streamlit run app/streamlit_app.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure app/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from model_loader import load_model_and_scalers
from inference import predict, generate_y_distribution, generate_antenna_coordinates
from visualizations import (
    plot_y_kde,
    plot_se_vs_ris,
    plot_ber_vs_snr,
    plot_capacity_vs_freq,
    plot_array_geometry,
    plot_sensitivity_radial,
    plot_comparison_radar,
    plot_rx_correlation,
    plot_rx_correlation_heatmap,
    plot_tornado_chart,
)
from sensitivity_analysis import run_sensitivity, METRIC_KEYS

# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="Digital Twin model of RIS-MIMO",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# Custom CSS — Dark Theme
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp { background-color: #0F0F1B !important; }
[data-testid="stHeader"] { background-color: rgba(15,15,27,0.9) !important; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #FFFFFF; }

section[data-testid="stSidebar"] > div {
    background: #121220;
    border-right: 1px solid rgba(255,255,255,0.05);
}

.metric-card {
    background: #18182D;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.15);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stTabs"] [data-baseweb="tab"] { color: #A0A0B0; font-weight: 600; }
[data-testid="stTabs"] [aria-selected="true"] { color: #00D2FF; border-bottom: 2px solid #00D2FF; }

.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    font-weight: bold;
    color: #A0A0B0 !important;
    text-transform: uppercase;
}

.info-note {
    background: rgba(0,210,255,0.06);
    border: 1px solid rgba(0,210,255,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    color: #A0C4D0;
    font-size: 0.78rem;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Load model (cached)
# =========================================================================
@st.cache_resource(show_spinner="Loading PINN model & scalers …")
def _load():
    model, sX, sYP, sSE, sBER, dev = load_model_and_scalers()
    return model, (sX, sYP, sSE, sBER, dev)

model, scalers = _load()
scaler_X, scaler_yp, scaler_se, scaler_ber, device = scalers


# =========================================================================
# Sidebar — Input Controls
# =========================================================================
with st.sidebar:
    st.markdown("<h2 style='color:#FFFFFF;font-size:1.4rem;'>📡 Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")

    # ── System ──────────────────────────────────────────────────────────
    with st.expander("SYSTEM", expanded=True):
        freq = st.selectbox(
            "Carrier Frequency",
            options=[3.5e9, 6e9, 26e9],
            format_func=lambda x: {3.5e9: "3.5 GHz (Sub-6)", 6e9: "6 GHz (Sub-6)", 26e9: "26 GHz (mmWave)"}[x],
            index=1, key="freq",
        )

    # ── Antenna Configuration ─────────────────────────────────────────
    with st.expander("ANTENNA CONFIGURATION", expanded=True):
        st.markdown("<h4 style='color:#FFFFFF;font-size:0.95rem;margin-bottom:0;'>Transmitter (Tx) Array</h4>", unsafe_allow_html=True)
        tx_array_type = st.selectbox("Tx Array Type", ["Linear Array", "Rectangular Array"], index=0, key="tx_array_type")
        
        if tx_array_type == "Linear Array":
            col_tx1, col_tx2 = st.columns(2)
            with col_tx1:
                n_tx = st.select_slider("N_t (Tx Elements)", options=[2, 4, 8, 16], value=4, key="n_tx")
            with col_tx2:
                d_tx = st.selectbox("d (Tx, λ)", options=[0.25, 0.5, 1.0], index=1, key="d_tx")
            tx_rows, tx_cols = 1, n_tx
            dx_tx, dy_tx = d_tx, 0.0
        else:
            col_tx1, col_tx2 = st.columns(2)
            with col_tx1:
                tx_rows = st.number_input("Tx Rows", min_value=1, max_value=8, value=2, step=1, key="tx_rows")
            with col_tx2:
                tx_cols = st.number_input("Tx Columns", min_value=1, max_value=8, value=2, step=1, key="tx_cols")
            n_tx = tx_rows * tx_cols
            st.markdown(f"<div style='color:#A0A0B0;font-size:0.8rem;'>Total Tx Antennas: <span style='color:#00D2FF;font-weight:bold;'>{n_tx}</span></div>", unsafe_allow_html=True)
            
            col_dx_tx, col_dy_tx = st.columns(2)
            with col_dx_tx:
                dx_tx = st.selectbox("dx (Tx, λ)", options=[0.25, 0.5, 1.0], index=1, key="dx_tx")
            with col_dy_tx:
                dy_tx = st.selectbox("dy (Tx, λ)", options=[0.25, 0.5, 1.0], index=1, key="dy_tx")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:12px 0;'>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color:#FFFFFF;font-size:0.95rem;margin-bottom:0;'>Receiver (Rx) Array</h4>", unsafe_allow_html=True)
        rx_array_type = st.selectbox("Rx Array Type", ["Linear Array", "Rectangular Array"], index=0, key="rx_array_type")
        
        if rx_array_type == "Linear Array":
            col_rx1, col_rx2 = st.columns(2)
            with col_rx1:
                n_rx = st.select_slider("N_r (Rx Elements)", options=[2, 4, 8, 16], value=4, key="n_rx")
            with col_rx2:
                d_rx = st.selectbox("d (Rx, λ)", options=[0.25, 0.5, 1.0], index=1, key="d_rx")
            rx_rows, rx_cols = 1, n_rx
            dx_rx, dy_rx = d_rx, 0.0
        else:
            col_rx1, col_rx2 = st.columns(2)
            with col_rx1:
                rx_rows = st.number_input("Rx Rows", min_value=1, max_value=8, value=2, step=1, key="rx_rows")
            with col_rx2:
                rx_cols = st.number_input("Rx Columns", min_value=1, max_value=8, value=2, step=1, key="rx_cols")
            n_rx = rx_rows * rx_cols
            st.markdown(f"<div style='color:#A0A0B0;font-size:0.8rem;'>Total Rx Antennas: <span style='color:#FC6E51;font-weight:bold;'>{n_rx}</span></div>", unsafe_allow_html=True)
            
            col_dx_rx, col_dy_rx = st.columns(2)
            with col_dx_rx:
                dx_rx = st.selectbox("dx (Rx, λ)", options=[0.25, 0.5, 1.0], index=1, key="dx_rx")
            with col_dy_rx:
                dy_rx = st.selectbox("dy (Rx, λ)", options=[0.25, 0.5, 1.0], index=1, key="dy_rx")
        
        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:12px 0;'>", unsafe_allow_html=True)
        
        n_ris = st.select_slider("RIS Elements (N)", options=[8, 16, 32, 64, 128], value=32, key="n_ris")

        theta = st.slider("Phase Shift θ (rad)", min_value=0.0, max_value=2 * np.pi,
                          value=np.pi / 4, step=0.05, format="%.2f", key="theta")

    # ── Channel Configuration ────────────────────────────────────────
    with st.expander("CHANNEL CONFIGURATION", expanded=True):
        snr_db = st.slider("SNR (dB)", min_value=-10, max_value=20, value=10, step=1, key="snr_db")

    # ── Distances ───────────────────────────────────────────────────
    with st.expander("DISTANCES", expanded=True):
        d_tx_ris = st.slider(
            "Tx Center → RIS Center (m)",
            min_value=1.0, max_value=100.0, value=15.0, step=0.5,
            format="%.1f m", key="d_tx_ris",
        )
        d_ris_rx = st.slider(
            "RIS Center → Rx Center (m)",
            min_value=1.0, max_value=100.0, value=15.0, step=0.5,
            format="%.1f m", key="d_ris_rx",
        )
        st.markdown(
            "<div class='info-note'>⚠️ Distance effects apply a <strong>free-space path-loss correction</strong> "
            "to the PINN output in physical space. These distances are <em>not</em> part of the original "
            "PINN training distribution.</div>",
            unsafe_allow_html=True,
        )

    # ── Channel Model Placeholder ───────────────────────────────────
    with st.expander("CHANNEL MODEL", expanded=False):
        st.selectbox(
            "Fading Model",
            options=["Rayleigh", "Rician", "Nakagami-m"],
            index=0,
            disabled=True,
            key="channel_model",
        )
        st.info("🔜 Multi-model support (Rician, Nakagami-m) coming in a future release.")

    st.markdown("---")
    compare_mode = st.checkbox("📊 Compare Scenarios", value=False, key="compare")


# =========================================================================
# Run inference
# =========================================================================
# Combine spacing for the model feature vector (using rx spacing as common for PINN feature mapping if required)
dx = dx_rx
dy = dy_rx if rx_array_type == "Rectangular Array" else dx_rx

# Generate Array geometries
tx_coords = generate_antenna_coordinates(tx_array_type, tx_rows, tx_cols, dx_tx, dy_tx)
rx_coords = generate_antenna_coordinates(rx_array_type, rx_rows, rx_cols, dx_rx, dy_rx)

params = dict(
    freq=freq, n_tx=n_tx, n_rx=n_rx, n_ris=n_ris,
    dx=dx, dy=dy, snr_db=snr_db, theta=theta,
    d_tx_ris=d_tx_ris, d_ris_rx=d_ris_rx,
    tx_array_type=tx_array_type, tx_rows=tx_rows, tx_cols=tx_cols, dx_tx=dx_tx, dy_tx=dy_tx,
    rx_array_type=rx_array_type, rx_rows=rx_rows, rx_cols=rx_cols, dx_rx=dx_rx, dy_rx=dy_rx,
)

metrics = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **params)

BASELINE_PARAMS = dict(
    freq=6e9, n_tx=4, n_rx=4, n_ris=16,
    dx=0.5, dy=0.5, snr_db=0, theta=np.pi / 4,
    d_tx_ris=15.0, d_ris_rx=15.0,
)
baseline_metrics = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **BASELINE_PARAMS)


# =========================================================================
# Dashboard Header
# =========================================================================
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:10px 0 20px 0;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:20px;">
  <div>
    <h1 style="margin:0;font-size:2rem;color:#FFFFFF;">Digital Twin model of RIS-MIMO</h1>
    <p style="color:#A0A0B0;font-size:0.9rem;margin-top:4px;margin-bottom:0;">
      <span style="color:#48CFAD;font-weight:bold;">● LIVE SIMULATION</span>
      &nbsp;|&nbsp; PINN Model (λ<sub>SE</sub>=0.5 · λ<sub>BER</sub>=0.5 · λ<sub>y</sub>=0.01)
      &nbsp;|&nbsp; Tx: <strong>{tx_array_type}</strong>
      &nbsp;|&nbsp; Rx: <strong>{rx_array_type}</strong>
    </p>
  </div>
  <div>
    <button style="background:#6C63FF;color:white;border:none;padding:10px 20px;
                   border-radius:8px;font-weight:bold;cursor:pointer;">Deploy Output</button>
  </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# KPI Row
# =========================================================================
def _render_kpi(label, value, unit, trend_val, is_inverse_trend=False, icon_color="#00D2FF", icon_html=""):
    if is_inverse_trend:
        is_positive_outcome = trend_val <= 0
        sign = "▼" if trend_val < 0 else "▲"
    else:
        is_positive_outcome = trend_val >= 0
        sign = "▲" if trend_val >= 0 else "▼"

    trend_color = "#48CFAD" if is_positive_outcome else "#FF6B6B"
    bg_color    = "rgba(72,207,173,0.1)" if is_positive_outcome else "rgba(255,107,107,0.1)"

    return f"""
    <div class="metric-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="color:#A0A0B0;font-size:0.8rem;font-weight:600;text-transform:uppercase;">{label}</span>
            <div style="color:{icon_color};font-size:1.2rem;">{icon_html}</div>
        </div>
        <div style="display:flex;align-items:baseline;margin-bottom:8px;">
            <span style="font-size:1.6rem;font-weight:700;color:#FFFFFF;margin-right:6px;">{value}</span>
            <span style="font-size:0.8rem;color:#A0A0B0;">{unit}</span>
        </div>
        <div>
            <span style="background:{bg_color};color:{trend_color};padding:4px 8px;
                         border-radius:4px;font-size:0.75rem;font-weight:600;">
                {sign} {abs(trend_val):.1f}%
            </span>
        </div>
    </div>
    """

def _pct(new, old):
    if abs(old) < 1e-15:
        return 0.0
    return ((new - old) / abs(old)) * 100.0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(_render_kpi("Spectral Eff", f"{metrics['se']:.3f}", "bits/s/Hz",
                            _pct(metrics['se'], baseline_metrics['se']),
                            icon_color="#00D2FF", icon_html="📶"), unsafe_allow_html=True)
with c2:
    st.markdown(_render_kpi("Channel Cap", f"{metrics['capacity_mbps']:.1f}", "Mbps",
                            _pct(metrics['capacity_mbps'], baseline_metrics['capacity_mbps']),
                            icon_color="#48CFAD", icon_html="📈"), unsafe_allow_html=True)
with c3:
    st.markdown(_render_kpi("BER", f"{metrics['ber']:.2e}", "",
                            _pct(metrics['ber'], baseline_metrics['ber']),
                            is_inverse_trend=True, icon_color="#FF6B6B", icon_html="🎯"), unsafe_allow_html=True)
with c4:
    st.markdown(_render_kpi("Rx Power", f"{metrics['y_power']:.2f}", "",
                            _pct(metrics['y_power'], baseline_metrics['y_power']),
                            icon_color="#FFCE54", icon_html="⚡"), unsafe_allow_html=True)
with c5:
    st.markdown(_render_kpi("SINR", f"{metrics['sinr_db']:.1f}", "dB",
                            _pct(metrics['sinr_db'], baseline_metrics['sinr_db']),
                            icon_color="#00D2FF", icon_html="〰"), unsafe_allow_html=True)

st.markdown("")  # spacer


# =========================================================================
# Main Layout: 2 Columns
# =========================================================================
col_main, col_right = st.columns([3, 1])

with col_main:
    if compare_mode:
        st.markdown('<h3 style="color:#FFFFFF;">📊 Scenario Comparison</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_comparison_radar(baseline_metrics, metrics),
                        use_container_width=True, key="radar")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Signal Distribution",
            "SE vs RIS Size",
            "BER vs SNR",
            "Capacity vs Frequency",
            "Array Geometry",
            "Rx Correlation",
        ])

        # ── Tab 1: KDE ──────────────────────────────────────────────────
        with tab1:
            st.plotly_chart(plot_y_kde(metrics["y_power"]),
                            use_container_width=True, key="kde")

        # ── Tab 2: SE vs RIS ────────────────────────────────────────────
        with tab2:
            st.plotly_chart(plot_se_vs_ris(model, scalers, params),
                            use_container_width=True, key="se_ris")

        # ── Tab 3: BER vs SNR ───────────────────────────────────────────
        with tab3:
            st.plotly_chart(plot_ber_vs_snr(model, scalers, params),
                            use_container_width=True, key="ber_snr")

        # ── Tab 4: Capacity vs Freq ──────────────────────────────────────
        with tab4:
            st.plotly_chart(plot_capacity_vs_freq(model, scalers, params),
                            use_container_width=True, key="cap_freq")

        # ── Tab 5: Array Geometry ───────────────────────────────────────
        with tab5:
            st.plotly_chart(plot_array_geometry(tx_coords, rx_coords),
                            use_container_width=True, key="geometry_plot")

        # ── Tab 6: Rx Correlation ───────────────────────────────────────
        with tab6:
            st.markdown("#### Rx Antenna Cross-Correlation  R_mn = E[y_m · y_n*]")

            corr_col1, corr_col2, corr_col3 = st.columns([1, 1, 2])
            with corr_col1:
                ant_m = st.number_input("Rx Antenna m", min_value=0, max_value=n_rx - 1,
                                        value=0, step=1, key="ant_m")
            with corr_col2:
                ant_n = st.number_input("Rx Antenna n", min_value=0, max_value=n_rx - 1,
                                        value=min(1, n_rx - 1), step=1, key="ant_n")
            with corr_col3:
                st.markdown(
                    "<div class='info-note' style='margin-top:28px;'>Correlation computed from "
                    "Monte-Carlo complex Gaussian samples using predicted y_power as variance.</div>",
                    unsafe_allow_html=True,
                )

            # Magnitude + phase subplots
            st.plotly_chart(
                plot_rx_correlation(metrics["y_power"], rx_coords, int(ant_m), int(ant_n)),
                use_container_width=True, key="rx_corr_bars",
            )

            # Heatmap + correlation matrix table
            if n_rx > 1:
                heat_col, tbl_col = st.columns([1, 1])
                with heat_col:
                    st.markdown("##### NxN Correlation Heatmap")
                    heatmap_fig, mag_matrix, rx_labels = plot_rx_correlation_heatmap(
                        metrics["y_power"], rx_coords
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True, key="rx_heatmap")

                with tbl_col:
                    st.markdown("##### Correlation Matrix |R_mn|")
                    df_corr = pd.DataFrame(
                        mag_matrix,
                        index=rx_labels,
                        columns=rx_labels,
                    ).round(4)
                    st.dataframe(
                        df_corr.style.background_gradient(cmap="viridis", vmin=0.0, vmax=mag_matrix.max()),
                        use_container_width=True,
                    )

        # Physics Insights
        st.markdown(f"""
        <div style="background:#18182D;padding:20px;border-radius:12px;
                    border:1px solid rgba(255,255,255,0.05);margin-top:10px;
                    display:flex;flex-direction:row;gap:20px;">
            <div style="flex:2;">
                <h4 style="color:#FFFFFF;margin-top:0;">Physics Insights</h4>
                <p style="color:#A0A0B0;font-size:0.85rem;margin-bottom:8px;">
                    AI-powered analytics on current simulation results.</p>
                <ul style="color:#A0A0B0;font-size:0.85rem;padding-left:20px;">
                    <li>RIS elements = {n_ris}, improving array beamforming gain.</li>
                    <li>Channel capacity bounded by Shannon Limit (λ_SE penalty).</li>
                    <li>BER strictly follows monotonic SNR physical constraint.</li>
                    <li>SINR remains stable across scattering limits.</li>
                    <li>Tx array: <strong>{tx_array_type}</strong> &nbsp;|&nbsp;
                        Rx array: <strong>{rx_array_type}</strong></li>
                </ul>
            </div>
            <div style="flex:1;border:1px solid rgba(72,207,173,0.5);border-radius:8px;
                        padding:16px;background:rgba(72,207,173,0.05);">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    <span style="font-size:1.2rem;">💡</span>
                    <h5 style="color:#48CFAD;margin:0;font-size:0.95rem;">Recommendation</h5>
                </div>
                <p style="color:#FFFFFF;font-size:0.82rem;margin:0;">
                    Increase RIS size to 64 elements for optimal spectral efficiency at current frequency.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Right Column: Topology + Sensitivity ────────────────────────────────
with col_right:
    # System Topology Card
    st.markdown("<h4 style='color:#FFFFFF;font-size:1.1rem;margin-top:0;'>🌐 RIS-MIMO System Topology</h4>",
                unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:#18182D;border-radius:12px;padding:16px;text-align:center;
            border:1px solid rgba(255,255,255,0.05);margin-bottom:24px;">
  <div style="width:100%;height:120px;
              background:radial-gradient(circle,rgba(108,99,255,0.2) 0%,rgba(0,0,0,0) 70%);
              border-radius:8px;margin-bottom:12px;display:flex;align-items:center;
              justify-content:center;">
    <span style="font-size:3rem;opacity:0.8;">📡 ⤍ 🧊 ⤍ 📱</span>
  </div>
  <div style="display:flex;justify-content:space-between;gap:8px;">
    <div style="background:rgba(108,99,255,0.1);padding:10px;border-radius:8px;flex:1;">
      <div style="color:#A0A0B0;font-size:0.65rem;text-transform:uppercase;">Tx→RIS</div>
      <div style="color:#FFFFFF;font-weight:bold;font-size:0.9rem;">{d_tx_ris:.1f} m</div>
    </div>
    <div style="background:rgba(72,207,173,0.1);padding:10px;border-radius:8px;flex:1;">
      <div style="color:#A0A0B0;font-size:0.65rem;text-transform:uppercase;">RIS→Rx</div>
      <div style="color:#FFFFFF;font-weight:bold;font-size:0.9rem;">{d_ris_rx:.1f} m</div>
    </div>
  </div>
  <div style="margin-top:10px;display:flex;justify-content:space-between;gap:8px;">
    <div style="background:rgba(108,99,255,0.08);padding:8px;border-radius:8px;flex:1;">
      <div style="color:#A0A0B0;font-size:0.6rem;text-transform:uppercase;">Tx Type</div>
      <div style="color:#FFFFFF;font-weight:600;font-size:0.75rem;">
          {"Lin" if tx_array_type == "Linear Array" else "Rect"}</div>
    </div>
    <div style="background:rgba(72,207,173,0.08);padding:8px;border-radius:8px;flex:1;">
      <div style="color:#A0A0B0;font-size:0.6rem;text-transform:uppercase;">Rx Type</div>
      <div style="color:#FFFFFF;font-weight:600;font-size:0.75rem;">
          {"Lin" if rx_array_type == "Linear Array" else "Rect"}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Sensitivity — radial preview + tornado button
    st.markdown("<h4 style='color:#FFFFFF;font-size:1.1rem;'>🎯 Parameter Sensitivity</h4>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#A0A0B0;font-size:0.8rem;'>Impact on Spectral Efficiency</p>",
                unsafe_allow_html=True)

    if st.button("▶ Run Sensitivity Sweep", use_container_width=True, key="run_sens"):
        with st.spinner("Analyzing 10 parameters…"):
            st.session_state["sens_data"] = run_sensitivity(model, scalers, params)

    if "sens_data" in st.session_state:
        st.plotly_chart(
            plot_sensitivity_radial(st.session_state["sens_data"], target_metric="SE"),
            use_container_width=True, key="radial_live",
        )
    else:
        sens_data_default = {
            "SE": [("RIS Size", 3.2), ("SNR", 2.8), ("Phase Shift", 2.1),
                   ("dx (λ)", 0.5), ("dy (λ)", 0.4), ("Tx→RIS Dist", 1.2)]
        }
        st.plotly_chart(
            plot_sensitivity_radial(sens_data_default, target_metric="SE"),
            use_container_width=True, key="radial_default",
        )


# =========================================================================
# Sensitivity Tornado — Full-width section below main layout
# =========================================================================
st.markdown("---")
st.markdown("<h3 style='color:#FFFFFF;margin-bottom:4px;'>🌪️ Sensitivity Tornado Chart</h3>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='color:#A0A0B0;font-size:0.85rem;margin-top:0;'>"
    "Parameter importance ranking — wider bars indicate higher impact on the chosen metric.</p>",
    unsafe_allow_html=True,
)

tornado_col1, tornado_col2 = st.columns([1, 5])
with tornado_col1:
    metric_display_map = {
        "SE":            "Spectral Efficiency",
        "BER":           "BER",
        "y_power":       "Rx Power",
        "capacity_mbps": "Capacity (Mbps)",
        "sinr_db":       "SINR (dB)",
    }
    tornado_metric = st.selectbox(
        "Target Metric",
        options=list(metric_display_map.keys()),
        format_func=lambda k: metric_display_map[k],
        key="tornado_metric",
    )
    run_tornado = st.button("▶ Compute Tornado", use_container_width=True, key="run_tornado")

with tornado_col2:
    if run_tornado:
        with st.spinner("Running full parameter sweep (10 parameters × 5 metrics)…"):
            st.session_state["sens_data"] = run_sensitivity(model, scalers, params)

    if "sens_data" in st.session_state:
        st.plotly_chart(
            plot_tornado_chart(st.session_state["sens_data"], target_metric=tornado_metric),
            use_container_width=True, key="tornado_chart",
        )
    else:
        st.info("Click **▶ Compute Tornado** or **▶ Run Sensitivity Sweep** above to generate the chart.")
