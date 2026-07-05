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

# --- Ensure app/ and project root are on the path for Streamlit Cloud deployment ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------------------------------------------------------------------

import streamlit as st
import json
import streamlit.components.v1 as components
from scipy.stats import gaussian_kde

from model_loader import load_model_and_scalers
from inference import predict, generate_y_distribution, generate_antenna_coordinates
from sensitivity_analysis import run_sensitivity, METRIC_KEYS

# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="Digital Twin Model of RIS-MIMO",
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

/* Dynamic cosmic background gradient */
@keyframes cosmicBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp { 
    background: linear-gradient(-45deg, #05050c, #0d0c1d, #081622, #03080e) !important; 
    background-size: 400% 400% !important;
    animation: cosmicBG 18s ease infinite !important;
}
[data-testid="stHeader"] { background-color: rgba(5,5,12,0.6) !important; backdrop-filter: blur(12px); }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #FFFFFF; }

/* Custom sidebar glassmorphism */
section[data-testid="stSidebar"] > div {
    background: rgba(13, 12, 29, 0.45) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Premium glassmorphic metric cards */
.metric-card {
    background: rgba(24, 24, 45, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(135deg, rgba(0, 210, 255, 0.05), transparent);
    opacity: 0;
    transition: opacity 0.4s ease;
}
.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    border-color: rgba(0, 210, 255, 0.25) !important;
    box-shadow: 0 12px 35px rgba(0, 210, 255, 0.12) !important;
}
.metric-card:hover::before {
    opacity: 1;
}

/* Tab styling with smooth transitions */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    gap: 8px;
}
[data-testid="stTabs"] [data-baseweb="tab"] { 
    color: #A0A0B0; 
    font-weight: 600; 
    border-radius: 8px 8px 0 0;
    padding: 10px 16px;
    transition: all 0.3s ease;
}
[data-testid="stTabs"] [aria-selected="true"] { 
    color: #00D2FF !important; 
    background: rgba(0, 210, 255, 0.06) !important;
    border-bottom: 2px solid #00D2FF !important;
}

/* Sidebar expanders styling */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    font-weight: 700;
    color: #00D9FF !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    margin-bottom: 6px;
}

.info-note {
    background: rgba(0,210,255,0.04);
    border: 1px solid rgba(0,210,255,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    color: #A0C4D0;
    font-size: 0.8rem;
    margin-top: 8px;
    line-height: 1.4;
}

/* Sleek inputs/buttons overrides */
div.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #00D9FF) !important;
    color: white !important;
    border: none !important;
    padding: 8px 20px !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(0, 217, 255, 0.5) !important;
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
        snr_db = st.slider("SNR (dB)", min_value=-10, max_value=20, value=10, step=1, key="snr_db")

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
            st.markdown(f"<div style='color:#A0A0B0;font-size:0.8rem;'>Total Tx Antennas: <span style='color:#00D9FF;font-weight:bold;'>{n_tx}</span></div>", unsafe_allow_html=True)
            
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
            st.markdown(f"<div style='color:#A0A0B0;font-size:0.8rem;'>Total Rx Antennas: <span style='color:#7C3AED;font-weight:bold;'>{n_rx}</span></div>", unsafe_allow_html=True)
            
            col_dx_rx, col_dy_rx = st.columns(2)
            with col_dx_rx:
                dx_rx = st.selectbox("dx (Rx, λ)", options=[0.25, 0.5, 1.0], index=1, key="dx_rx")
            with col_dy_rx:
                dy_rx = st.selectbox("dy (Rx, λ)", options=[0.25, 0.5, 1.0], index=1, key="dy_rx")
        
        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:12px 0;'>", unsafe_allow_html=True)
        
        n_ris = st.select_slider("RIS Elements (N)", options=[8, 16, 32, 64, 128], value=32, key="n_ris")

        theta = st.slider("Phase Shift θ (rad)", min_value=0.0, max_value=2 * np.pi,
                          value=np.pi / 4, step=0.05, format="%.2f", key="theta")



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



    # ── Sensitivity Configuration ───────────────────────────────────
    with st.expander("SENSITIVITY CONFIGURATION", expanded=True):
        st.radio(
            "Sensitivity Mode",
            options=["Differential Sensitivity (recommended)", "Standard Max-Δ Sensitivity"],
            index=0,
            key="sens_mode",
            help="Choose between normalized elasticity-based differential sensitivity or standard max-min delta sweeps."
        )

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


# =========================================================================
# Dashboard Title & Badge
# =========================================================================
st.markdown("""
<div style="padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 25px;">
    <h1 style="margin:0;font-size:2.4rem;font-weight:700;letter-spacing:-0.03em;
               background:linear-gradient(135deg, #FFFFFF 30%, #00D2FF 70%, #48CFAD 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               text-shadow: 0 0 30px rgba(0,210,255,0.15);">
      Digital Twin Model of RIS-MIMO
    </h1>
    <div style="margin-top: 10px; display: inline-flex; align-items: center; gap: 8px; 
                background: rgba(0, 217, 255, 0.1); border: 1px solid rgba(0, 217, 255, 0.3); 
                border-radius: 50px; padding: 4px 14px; font-size: 0.8rem; color: #00D9FF; 
                font-weight: 600; cursor: help;" 
         title="The current Digital Twin models a RIS-assisted Non-Line-of-Sight (NLOS) environment using Rayleigh fading.">
        📡 Propagation Environment: NLOS
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# Redesigned Dashboard Rendering
# =========================================================================
# 1. Calculate KDE data
try:
    samples = generate_y_distribution(metrics["y_power"], n_samples=3000)
    samples_valid = samples[samples > 0]
    kde = gaussian_kde(samples_valid, bw_method="scott")
    x_grid = np.linspace(float(samples_valid.min()), float(samples_valid.max()), 100)
    density = kde(x_grid)
    kde_data = {
        "x": x_grid.tolist(),
        "y": density.tolist()
    }
except Exception as e:
    kde_data = {"x": [], "y": []}

# 2. Run sensitivity analysis for all metrics
sens_mode_internal = "differential" if st.session_state.get("sens_mode", "Differential Sensitivity (recommended)") == "Differential Sensitivity (recommended)" else "legacy"
sens_data = run_sensitivity(model, scalers, params, mode=sens_mode_internal)

# Convert complex formats or numpy numbers to standard JSON types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# 3. Read index.html
html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

# 4. Inject variables
js_inject = f"""
<script>
    window.initialParams = {json.dumps(params, cls=NpEncoder)};
    window.initialMetrics = {json.dumps(metrics, cls=NpEncoder)};
    window.kdeData = {json.dumps(kde_data, cls=NpEncoder)};
    window.sensData = {json.dumps(sens_data, cls=NpEncoder)};
</script>
"""
html_content = html_content.replace("<head>", f"<head>{js_inject}")

# Render as Streamlit HTML component
components.html(html_content, height=950, scrolling=True)


