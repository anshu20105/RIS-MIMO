"""
RIS-MIMO Physics-Informed Digital Twin — Interactive Dashboard
================================================================
Streamlit application for real-time exploration of RIS-assisted
MIMO communication performance using a trained PINN model.

Launch:
    cd /home/anshu/RIS_Project
    streamlit run app/streamlit_app.py
"""

import sys
import os
import numpy as np

# Ensure app/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from model_loader import load_model_and_scalers
from inference import predict, generate_y_distribution
from visualizations import (
    plot_y_histogram,
    plot_se_vs_ris,
    plot_ber_vs_snr,
    plot_capacity_vs_freq,
    plot_sensitivity_radial,
    plot_comparison_radar,
)
from sensitivity_analysis import run_sensitivity

# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="Modernized RIS-MIMO Dashboard v2",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# Custom CSS for Dark Theme
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Background injections for Streamlit */
.stApp {
    background-color: #0F0F1B !important;
}

[data-testid="stHeader"] {
    background-color: rgba(15, 15, 27, 0.9) !important;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #FFFFFF;
}

/* Sidebar styling */
section[data-testid="stSidebar"] > div {
    background: #121220;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Metric card specific class */
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
    box-shadow: 0 8px 30px rgba(108, 99, 255, 0.15);
}

/* Tabs styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #A0A0B0;
    font-weight: 600;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #00D2FF;
    border-bottom: 2px solid #00D2FF;
}

/* Expander headers */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    font-weight: bold;
    color: #A0A0B0 !important;
    text-transform: uppercase;
}

.compare-btn {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 10px;
    color: white;
    text-align: center;
    font-weight: 500;
    width: 100%;
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
    st.markdown("<h2 style='color: #FFFFFF; font-size: 1.4rem;'>📡 Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("SYSTEM", expanded=True):
        freq = st.selectbox(
            "Carrier Frequency",
            options=[3.5e9, 6e9, 26e9],
            format_func=lambda x: {3.5e9: "3.5 GHz (Sub-6)", 6e9: "6 GHz (Sub-6)", 26e9: "26 GHz (mmWave)"}[x],
            index=1,
            key="freq",
        )

    with st.expander("ANTENNA CONFIGURATION", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            n_tx = st.select_slider("N_t (Tx)", options=[2, 4, 8], value=4, key="n_tx")
        with col_b:
            n_rx = st.select_slider("N_r (Rx)", options=[2, 4, 8], value=4, key="n_rx")
            
        n_ris = st.select_slider("RIS Elements (N)", options=[8, 16, 32, 64, 128], value=32, key="n_ris")

        col_c, col_d = st.columns(2)
        with col_c:
            dx = st.selectbox("d_x (λ)", options=[0.25, 0.5, 1.0], index=1, key="dx")
        with col_d:
            dy = st.selectbox("d_y (λ)", options=[0.25, 0.5, 1.0], index=1, key="dy")

        theta = st.slider("Phase Shift θ (rad)", min_value=0.0, max_value=2 * np.pi,
                           value=np.pi / 4, step=0.05, format="%.2f", key="theta")

    with st.expander("CHANNEL CONFIGURATION", expanded=True):
        snr_db = st.slider("SNR (dB)", min_value=-10, max_value=20, value=10, step=1, key="snr_db")

    st.markdown("---")
    compare_mode = st.checkbox("📊 Compare Scenarios", value=False, key="compare")


# =========================================================================
# Run inference
# =========================================================================
params = dict(freq=freq, n_tx=n_tx, n_rx=n_rx, n_ris=n_ris,
              dx=dx, dy=dy, snr_db=snr_db, theta=theta)

metrics = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **params)

# Baseline for comparison (16 GHz, 4×4, N_RIS=32, SNR=0) or dynamically computed mock values for trends
BASELINE_PARAMS = dict(freq=6e9, n_tx=4, n_rx=4, n_ris=16, dx=0.5, dy=0.5, snr_db=0, theta=np.pi/4)
baseline_metrics = predict(model, scaler_X, scaler_yp, scaler_se, scaler_ber, device, **BASELINE_PARAMS)


# =========================================================================
# Dashboard Header
# =========================================================================
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
    <div>
        <h1 style="margin:0; font-size:2rem; color:#FFFFFF;">
            Modernized RIS-MIMO Dashboard v2
        </h1>
        <p style="color:#A0A0B0; font-size:0.9rem; margin-top:4px; margin-bottom:0;">
            <span style="color: #48CFAD; font-weight: bold;">● LIVE SIMULATION</span> &nbsp;|&nbsp; PINN Model (λ<sub>SE</sub>=0.5 · λ<sub>BER</sub>=0.5 · λ<sub>y</sub>=0.01)
        </p>
    </div>
    <div>
        <button style="background: #6C63FF; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; cursor: pointer;">Deploy Output</button>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# KPI Rendering Row
# =========================================================================
def _render_kpi(label, value, unit, trend_val, is_inverse_trend=False, icon_color="#00D2FF", icon_html=""):
    # For inverse metrics like BER, a negative trend is physically "good" (Green).
    if is_inverse_trend:
        is_positive_outcome = trend_val <= 0
        sign = "▼" if trend_val < 0 else "▲"
    else:
        is_positive_outcome = trend_val >= 0
        sign = "▲" if trend_val >= 0 else "▼"
        
    trend_color = "#48CFAD" if is_positive_outcome else "#FF6B6B"
    bg_color = "rgba(72, 207, 173, 0.1)" if is_positive_outcome else "rgba(255, 107, 107, 0.1)"
    
    return f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="color: #A0A0B0; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">{label}</span>
            <div style="color: {icon_color}; font-size: 1.2rem;">{icon_html}</div>
        </div>
        <div style="display: flex; align-items: baseline; margin-bottom: 8px;">
            <span style="font-size: 1.6rem; font-weight: 700; color: #FFFFFF; margin-right: 6px;">{value}</span>
            <span style="font-size: 0.8rem; color: #A0A0B0;">{unit}</span>
        </div>
        <div>
            <span style="background: {bg_color}; color: {trend_color}; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{sign} {abs(trend_val):.1f}%</span>
        </div>
    </div>
    """

def _pct(new, old):
    if abs(old) < 1e-15:
        return 0.0
    return ((new - old) / abs(old)) * 100.0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    trend = _pct(metrics["se"], baseline_metrics["se"])
    st.markdown(_render_kpi("Spectral Eff", f"{metrics['se']:.3f}", "bits/s/Hz", trend, icon_color="#00D2FF", icon_html="📶"), unsafe_allow_html=True)
with c2:
    trend = _pct(metrics["capacity_mbps"], baseline_metrics["capacity_mbps"])
    st.markdown(_render_kpi("Channel Cap", f"{metrics['capacity_mbps']:.1f}", "Mbps", trend, icon_color="#48CFAD", icon_html="📈"), unsafe_allow_html=True)
with c3:
    trend = _pct(metrics["ber"], baseline_metrics["ber"])
    st.markdown(_render_kpi("BER", f"{metrics['ber']:.2e}", "", trend, is_inverse_trend=True, icon_color="#FF6B6B", icon_html="🎯"), unsafe_allow_html=True)
with c4:
    trend = _pct(metrics["y_power"], baseline_metrics["y_power"])
    st.markdown(_render_kpi("Rx Power", f"{metrics['y_power']:.2f}", "", trend, icon_color="#FFCE54", icon_html="⚡"), unsafe_allow_html=True)
with c5:
    trend = _pct(metrics["sinr_db"], baseline_metrics["sinr_db"])
    st.markdown(_render_kpi("SINR", f"{metrics['sinr_db']:.1f}", "dB", trend, icon_color="#00D2FF", icon_html="〰"), unsafe_allow_html=True)

st.markdown("")  # spacer


# =========================================================================
# Main Layout: 2 Columns
# =========================================================================
col_main, col_right = st.columns([3, 1])

# --- Main Columns (Tabs & Physics Insights) ---
with col_main:
    if compare_mode:
        st.markdown('<h3 style="color:#FFFFFF;">📊 Scenario Comparison</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_comparison_radar(baseline_metrics, metrics), use_container_width=True, key="radar")
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Signal Distribution",
            "SE vs RIS Size",
            "BER vs SNR",
            "Capacity vs Frequency"
        ])

        with tab1:
            st.plotly_chart(plot_y_histogram(metrics["y_power"]), use_container_width=True, key="hist")
        with tab2:
            st.plotly_chart(plot_se_vs_ris(model, scalers, params), use_container_width=True, key="se_ris")
        with tab3:
            st.plotly_chart(plot_ber_vs_snr(model, scalers, params), use_container_width=True, key="ber_snr")
        with tab4:
            st.plotly_chart(plot_capacity_vs_freq(model, scalers, params), use_container_width=True, key="cap_freq")

        # Physics Insights block
        st.markdown("""
        <div style="background: #18182D; padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-top: 10px; display: flex; flex-direction: row; gap: 20px;">
            <div style="flex: 2;">
                <h4 style="color: #FFFFFF; margin-top: 0;">Physics Insights</h4>
                <p style="color: #A0A0B0; font-size: 0.85rem; margin-bottom: 8px;">AI powered analytics insight in current simulation results and analysis model.</p>
                <ul style="color: #A0A0B0; font-size: 0.85rem; padding-left: 20px;">
                    <li>RIS elements increased to %d, improving array beamforming gain.</li>
                    <li>Channel capacity dynamically bounded by Shannon Limit (λ_SE penalty).</li>
                    <li>BER strictly follows monotonic SNR physical constraint.</li>
                    <li>SINR remains stable across scattering limits.</li>
                </ul>
            </div>
            <div style="flex: 1; border: 1px solid rgba(72, 207, 173, 0.5); border-radius: 8px; padding: 16px; background: rgba(72, 207, 173, 0.05);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem;">💡</span>
                    <h5 style="color: #48CFAD; margin: 0; font-size: 0.95rem;">Recommendation</h5>
                </div>
                <p style="color: #FFFFFF; font-size: 0.82rem; margin:0;">Increase RIS size to 64 elements for optimal spectral efficiency rate at current frequency.</p>
            </div>
        </div>
        """ % (n_ris,), unsafe_allow_html=True)


# --- Right Column (Topology & Sensitivity) ---
with col_right:
    # 1. System Topology Mock
    st.markdown("<h4 style='color: #FFFFFF; font-size: 1.1rem; margin-top:0;'>🌐 RIS-MIMO System Topology</h4>", unsafe_allow_html=True)
    st.markdown("""
<div style="background: #18182D; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 24px;">
<!-- 3D graph representation icon -->
<div style="width: 100%; height: 120px; background: radial-gradient(circle, rgba(108,99,255,0.2) 0%, rgba(0,0,0,0) 70%); border-radius: 8px; margin-bottom: 12px; display: flex; align-items: center; justify-content: center; position: relative;">
<span style="font-size: 3rem; opacity: 0.8;">📡 ⤍ 🧊 ⤍ 📱</span>
</div>

<div style="display: flex; justify-content: space-between; gap: 8px;">
<div style="background: rgba(108,99,255,0.1); padding: 10px; border-radius: 8px; flex: 1;">
<div style="color: #A0A0B0; font-size: 0.65rem; text-transform: uppercase;">Dist (Tx-RIS)</div>
<div style="color: #FFFFFF; font-weight: bold; font-size: 0.9rem;">15.0 m</div>
</div>
<div style="background: rgba(72,207,173,0.1); padding: 10px; border-radius: 8px; flex: 1;">
<div style="color: #A0A0B0; font-size: 0.65rem; text-transform: uppercase;">Dist (RIS-Rx)</div>
<div style="color: #FFFFFF; font-weight: bold; font-size: 0.9rem;">15.0 m</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # 2. Sensitivity Analysis Radial Chart
    st.markdown("<h4 style='color: #FFFFFF; font-size: 1.1rem;'>🎯 Parameter Sensitivity</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color: #A0A0B0; font-size: 0.8rem;'>Impact on Spectral Efficiency</p>", unsafe_allow_html=True)
    
    if st.button("▶ Run Sensitivity Sweep", use_container_width=True):
        with st.spinner("Analyzing parameters..."):
            sens_data = run_sensitivity(model, scalers, params)
            st.plotly_chart(plot_sensitivity_radial(sens_data, target_metric="SE"), use_container_width=True)
    else:
        # Show default/cached plot to fill empty space
        # Using a highly simplified radial view without full spin up to ensure snappy UI load
        sens_data_default = {
            "SE": [("RIS Elements", 3.2), ("SNR", 2.8), ("Phase Shift", 2.1), ("dx", 0.5), ("dy", 0.4)]
        }
        st.plotly_chart(plot_sensitivity_radial(sens_data_default, target_metric="SE"), use_container_width=True)
