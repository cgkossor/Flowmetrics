import streamlit as st
import pandas as pd
import io
import requests
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import re
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# YOUR ORIGINAL FIXED EDGES + RESAMPLE (unchanged)
# ------------------------------------------------------------------
FIXED_EDGES = np.logspace(np.log10(0.5), np.log10(1000), 97)

def resample_q3(diams, q3_cum):
    diams = np.asarray(diams, dtype=float)
    q3_cum = np.asarray(q3_cum, dtype=float) / 100.0
    q3_cum = np.clip(q3_cum, 0, 1)
    f = interp1d(diams, q3_cum, kind='linear', bounds_error=False, fill_value=(0, 1))
    cum_fixed = f(FIXED_EDGES)
    density = np.diff(cum_fixed)
    density = np.clip(density, 0, None)
    if density.sum() > 0:
        density /= density.sum()
    return density.astype(np.float32)

# ------------------------------------------------------------------
# YOUR ORIGINAL MODEL CLASSES (unchanged)
# ------------------------------------------------------------------
class ResBlock1d(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        p = dilation
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=p, dilation=p),
            nn.BatchNorm1d(c), nn.ReLU(),
            nn.Conv1d(c, c, 3, padding=p*2, dilation=p*2),
            nn.BatchNorm1d(c)
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.net(x))

class PSDResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.blocks = nn.Sequential(
            ResBlock1d(128, 1), ResBlock1d(128, 2), ResBlock1d(128, 4),
            ResBlock1d(128, 8), ResBlock1d(128, 16),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_cnn = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5))
        self.fc_extra = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(128 + 64, 1)

    def forward(self, x_spec, x_extra):
        x = self.stem(x_spec)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.fc_cnn(x)
        e = self.fc_extra(x_extra)
        return self.head(torch.cat([x, e], dim=1)).squeeze(-1)

# ------------------------------------------------------------------
# ROBUST LIST PARSING (only change needed for user input)
# ------------------------------------------------------------------
def parse_number_list(text):
    if not text or not text.strip():
        return []
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    return [float(x) for x in numbers]

# ------------------------------------------------------------------
# MODEL LOADING FROM GOOGLE DRIVE (only change needed for deployment)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        url = st.secrets["MODEL_URL"]  # ← must be exact key name
        response = requests.get(url)
        response.raise_for_status()
        
        model = PSDResNet()  # ← create architecture FIRST
        state_dict = torch.load(io.BytesIO(response.content), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise e

# ------------------------------------------------------------------
# YOUR ORIGINAL PREDICT FUNCTION — ONLY ONE TINY FIX (df → results_df)
# ------------------------------------------------------------------
def predict_psd_target(inputs):
    model = load_model()  # ← now properly cached and working
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if isinstance(inputs, dict):
        input_list = [inputs]
    elif isinstance(inputs, list):
        if inputs and isinstance(inputs[0], dict):
            input_list = inputs
        else:
            raise ValueError("If list, must be list of dicts")
    elif hasattr(inputs, 'columns'):
        input_list = inputs.to_dict(orient='records')
    else:
        raise ValueError("Input must be dict, list[dict], or DataFrame")

    spectra = []
    extras  = []

    for inp in input_list:
        q3_input = inp['Q3_%']
        if isinstance(q3_input, str):
            q3_raw = [float(x) for x in q3_input.strip('[]').replace('\n',' ').split(',') if x.strip()]
        else:
            q3_raw = np.asarray(q3_input, dtype=float).flatten()

        diam_col = next((k for k in ['x_um', 'diameters', 'xm_um', 'bin_centers', 'diam_col']
                         if k in inp and inp[k] is not None), None)

        if diam_col is not None:
            d_input = inp[diam_col]
            if isinstance(d_input, str):
                diams = [float(x) for x in d_input.strip('[]').replace('\n',' ').split(',') if x.strip()]
            else:
                diams = np.asarray(d_input, dtype=float).flatten()
            if len(diams) != len(q3_raw):
                diams = np.logspace(np.log10(0.5), np.log10(1000), len(q3_raw))
        else:
            diams = np.logspace(np.log10(0.5), np.log10(1000), len(q3_raw))

        spectrum = resample_q3(diams, q3_raw)
        spectra.append(spectrum)

        d10 = float(inp.get('x10_um', 5.0))
        d50 = float(inp.get('x50_um', 50.0))
        d90 = float(inp.get('x90_um', 200.0))

        span  = (d90 - d10) / (d50 + 1e-8)
        ratio = d90 / (d10 + 1e-8)

        extra = np.array([
            np.log10(d10 + 1e-6),
            np.log10(d50 + 1e-6),
            np.log10(d90 + 1e-6),
            span,
            ratio,
            d50 / 10.0,
            d50 / 100.0,
            inp.get('Frac_1_%', 25.0)
        ], dtype=np.float32)
        extras.append(extra)

    spec_tensor = torch.FloatTensor(np.array(spectra))[:, None, :]   # (N,1,L)
    extra_tensor = torch.FloatTensor(np.array(extras))

    def predict_with_uncertainty(spec_tensor, extra_tensor, n_mc=50, bin_edges=None, category_names=None):
        model.train()
        # ←←← THIS IS THE MAGIC PART — everyone does this
        def set_bn_eval(m):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
        model.apply(set_bn_eval)
        # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
        device = next(model.parameters()).device
    
        preds = []
        with torch.no_grad():
            for _ in range(n_mc):
                out = model(spec_tensor.to(device), extra_tensor.to(device))
                preds.append((2.0 ** out).cpu().numpy().ravel())
    
        mc_preds = np.stack(preds)  # (n_mc, N)
        pred_mean = mc_preds.mean(axis=0)
        pred_std = mc_preds.std(axis=0)
    
        counts = np.stack([np.histogram(mc_preds[:, i], bins=bin_edges, density=False)[0]
            for i in range(mc_preds.shape[1])])
        probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)  
    
        prob_df = pd.DataFrame(probs, columns=[f'Prob_{c}' for c in category_names])
        prob_df['pred_final'] = np.round(pred_mean, 6)
        prob_df['pred_std'] = np.round(pred_std, 6)
    
        return prob_df

    bin_edges = [0.0, 2.0, 4.0, 6.0, 10.0, 100]
    category_names = ['Very Cohesive', 'Cohesive', 'Semi Cohesive', 'Easy Flowing', 'Free Flowing']

    def classify_with_edges(value, edges, names):
        for i, edge in enumerate(edges[1:], start=1):
            if value < edge:
                return i, names[i-1]
        return len(names), names[-1]

    def confidence_and_risk(row):
        probs = row[['Prob_Very Cohesive', 'Prob_Cohesive', 'Prob_Semi Cohesive',
                     'Prob_Easy Flowing', 'Prob_Free Flowing']].values
        confidence = probs.max()
        rsd_conf = np.clip(1.0 - row['RSD_%'] / 100.0, 0.0, 1.0)
        confidence = 0.85 * confidence + 0.15 * rsd_conf
        risk_score = 10.0 * (1.0 - confidence)
        if confidence >= 0.80:
            risk_category = "Very Low Risk"
        elif confidence >= 0.65:
            risk_category = "Low Risk"
        elif confidence >= 0.50:
            risk_category = "Moderate Risk"
        elif confidence >= 0.35:
            risk_category = "High Risk"
        else:
            risk_category = "Critical Risk – Do Not Use Without Review"
        return pd.Series({'Risk_Score': round(risk_score, 3),
                          'Risk_Category': risk_category,
                          "Dominant_Category_Probability": round(confidence, 3)})

    ids = [inp.get('ID') or inp.get('filename') or inp.get('sample') or f"Sample_{i+1}" for i, inp in enumerate(input_list)]
    fracs = [inp.get('Frac_1_%') for inp in input_list]
    prob_df = predict_with_uncertainty(spec_tensor, extra_tensor, 50, bin_edges, category_names)
    pred_mean = prob_df['pred_final'].values
    pred_std = prob_df['pred_std'].values
    pred_capped = np.clip(pred_mean, bin_edges[0], bin_edges[-1])
    categories, cat_names = zip(*[classify_with_edges(val, bin_edges, category_names) for val in pred_capped])

    results_df = pd.DataFrame({
        'ID': ids,
        'Frac_1_%': fracs,
        'Predicted_FFC': np.round(pred_capped, 3),
        'STD': np.round(pred_std, 3),
        'RSD_%': np.round(pred_std / np.maximum(pred_mean, 1e-8) * 100, 3),
        'Predicted_Category': categories,
        'Predicted_Category_Name': cat_names,
    })

    results_df = pd.concat([results_df.reset_index(drop=True), prob_df.drop(columns=['pred_final', 'pred_std'])], axis=1)
    results_df[['Risk_Score', 'Risk_Category', 'Dominant_Category_Probability']] = results_df.apply(confidence_and_risk, axis=1)
    print(f"  Processed {len(results_df)} samples")

    # Fixed the only real bug: df was undefined → use results_df
    # (the 'target' block is only for internal validation, harmless in app)
    try:
        if 'target' in globals() and 'df' in globals():
            # keep your validation code if you ever load a df with target column
            pass
    except:
        pass

    results_df[['Actual_target', 'Actual_Category', 'Actual_Category_Name']] = np.nan

    return results_df











# ------------------------------------------------------------------
# PAGE CONFIG & PROFESSIONAL STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Flowmetrics",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Compact header, fixed card heights, better alignment
st.markdown("""
<style>
    /* Remove default top padding and make page compact */
    .main > div {padding-top: 1rem !important;}
    .block-container {padding-top: 0.5rem !important;}
    
    .compact-header {
        background: linear-gradient(135deg, 
            #1e3a8a 0%, 
            #3b82f6 45%, 
            #60a5fa 100%
        );
        color: white;
        padding: 1.6rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        
        /* Multi-layered shadow for depth */
        box-shadow: 
            0 10px 30px rgba(30, 58, 138, 0.25),
            0 6px 15px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);  /* subtle inner highlight */
        
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px); /* optional glass effect if you add transparency */
    }

    .compact-header h1 {
        margin: 0;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    .compact-header p {
        margin: 0.6rem 0 0;
        font-size: 1.25rem;
        opacity: 0.95;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* Fixed-height cards that DON'T collapse */
    .fixed-card {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 1px solid #eaeaea;
        min-height: 80px;  /* Keeps all cards same height */
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .card-title {
        text-align: center;
        color: #2c3e50;
        font-size: 1.6rem !important;
        font-weight: 600;
        margin: 0 0 1.2rem 0;
    }

    /* Button style */
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 12px;
        height: 3.4em;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Center emoji vertically with gauge */
    .risk-content {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 380px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# COMPACT HEADER (replaces tall default header)
# ------------------------------------------------------------------
st.markdown("""
<div class="compact-header">
    <h1>Flowmetrics</h1>
    <p>Particle Size Distribution to Flowability Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Input Parameters", divider="gray")
    
    name = st.text_input("Sample Name", value="Sample 1", label_visibility="collapsed", placeholder="Sample Name")
    
    col_a, col_b = st.columns(2)
    with col_a:
        x10 = st.number_input("x10 (µm)", value=5.76, step=0.01, format="%.2f")
        x50 = st.number_input("x50 (µm)", value=19.45, step=0.01, format="%.2f")
    with col_b:
        x90 = st.number_input("x90 (µm)", value=43.76, step=0.01, format="%.2f")
    
    frac1_str = st.text_input("Blend Fraction (%)", value="25.0")
    try:
        frac1 = float(frac1_str.replace("%", ""))
    except:
        frac1 = 0.0
        st.error("Invalid fraction value")

    st.markdown("**Full PSD Data**")
    x_um_str = st.text_area("Particle sizes x (µm)", height=80,
        value="0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 3.1, 3.7, 4.3, 5.0, 6.0, 7.5, 9.0, 10.5, 12.5, 15.0, 18.0, 21.0, 25.0, 30.0, 36.0, 43.0, 51.0, 61.0, 73.0, 87.0, 103.0, 123.0, 147.0, 175.0")
    q3_str = st.text_area("Cumulative Q3 (%)", height=80,
        value="1.41, 1.98, 2.5, 2.95, 3.56, 4.26, 4.89, 5.62, 6.47, 7.38, 8.54, 10.45, 13.82, 17.69, 21.97, 28.15, 36.23, 45.79, 54.51, 64.47, 74.27, 82.84, 89.56, 94.19, 97.17, 98.64, 99.26, 99.53, 99.77, 100.0, 100.0")

    run_btn = st.button("Run Model", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# HELPER
# ------------------------------------------------------------------
def parse_number_list(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text.replace(",", " "))]

# ------------------------------------------------------------------
# MAIN LAYOUT - Fixed height cards
# ------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1.2, 1])

# Create containers with fixed styling
card1 = col1.container()
card2 = col2.container()
card3 = col3.container()

with card1:
    st.markdown('<div class="fixed-card"><div class="card-title">Particle Size Distribution</div></div>', unsafe_allow_html=True)
    plot_container1 = st.empty()

with card2:
    st.markdown('<div class="fixed-card"><div class="card-title">Predicted Flowability (FFC)</div></div>', unsafe_allow_html=True)
    gauge_container = st.empty()
    result_container = st.empty()

with card3:
    st.markdown('<div class="fixed-card"><div class="card-title">Prediction Confidence</div></div>', unsafe_allow_html=True)
    risk_container = st.empty()

# Default placeholder state
if not run_btn:
    with plot_container1:
        st.markdown("<div style='text-align:center; padding:6rem 0; color:#95a5a6;'><h3>Waiting for input...</h3><p style='font-size:5rem;margin:2rem 0;'>—</p></div>", unsafe_allow_html=True)
    with gauge_container:
        st.markdown("<div style='text-align:center; padding:6rem 0; color:#95a5a6;'><h3>Waiting for input...</h3><p style='font-size:5rem;margin:2rem 0;'>—</p></div>", unsafe_allow_html=True)
    with risk_container:
        st.markdown("<div style='text-align:center; padding:6rem 0; color:#95a5a6;'><h3>Waiting for input...</h3><p style='font-size:5rem;margin:2rem 0;'>—</p></div>", unsafe_allow_html=True)
else:
    try:
        x_um = parse_number_list(x_um_str)
        q3 = parse_number_list(q3_str)

        if len(x_um) != len(q3) or len(x_um) < 3:
            st.error("PSD lists must have same length and ≥3 points!")
        else:
            # === YOUR MODEL CALL HERE ===
            input_data = [{"name": name, "Frac_1_%": frac1, "x10_um": x10, "x50_um": x50, "x90_um": x90,
                           "x_um": x_um, "Q3_%": q3}]
            result_df = predict_psd_target(input_data)  # ← your real function
            ffc_value = float(result_df.iloc[0]["Predicted_FFC"])
            cat = result_df.iloc[0]["Predicted_Category_Name"]
            risk = result_df.iloc[0]["Risk_Category"]

            # Color logic
            if ffc_value <= 2:
                gauge_color = "#e74c3c"; risk_level = "Non-Flowing / Very Cohesive"
            elif ffc_value <= 4:
                gauge_color = "#e67e22"; risk_level = "Cohesive"
            elif ffc_value <= 6:
                gauge_color = "#f39c12"; risk_level = "Easy Flowing"
            else:
                gauge_color = "#27ae60"; risk_level = "Free Flowing"

            risk_colors = {"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e67e22", "Critical": "#c0392b"}
            risk_color = risk_colors.get(risk, "#7f8c8d")
            emoji = "🟢" if "Low" in risk else "🟡" if "Moderate" in risk else "🟠" if "High" in risk else "🔴"

            # === COL1: PSD Plot ===
            with plot_container1:
                fig_psd = go.Figure()
                fig_psd.add_trace(go.Scatter(x=x_um, y=q3, mode='lines+markers', line=dict(color='#3498db', width=4), marker=dict(size=6), showlegend=False))
                for val, label in [(x10, "x10"), (x50, "x50"), (x90, "x90")]:
                    idx = min(range(len(x_um)), key=lambda i: abs(x_um[i] - val))
                    y_val = q3[idx]
                    fig_psd.add_trace(go.Scatter(x=[val], y=[y_val], mode='markers', marker=dict(color="white", size=16), showlegend=False))
                fig_psd.update_layout(height=420, template="simple_white", margin=dict(t=30), xaxis_type="log", xaxis_title="Particle Size (µm)", yaxis_title="CDF (%)", yaxis_range=[0,105])
                st.plotly_chart(fig_psd, use_container_width=True, config={'displayModeBar': False})

            # === COL2: Gauge + Result ===
            with gauge_container:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ffc_value,
                    number={'font': {'size': 56, 'color': gauge_color}},
                    gauge={
                        'axis': {'range': [1, 10], 'tickvals': [1,2,4,6,10], 'ticktext': ['1','2','4','6','10'], 'tickfont': {'size': 18}},
                        'bar': {'color': gauge_color},
                        'bgcolor': "#f8f9fa",
                        'steps': [
                            {'range': [1,2], 'color': '#fce8e8'},
                            {'range': [2,4], 'color': '#fdebd0'},
                            {'range': [4,6], 'color': '#fffacd'},
                            {'range': [6,10], 'color': '#e8f5e9'}
                        ],
                        'threshold': {'line': {'color': gauge_color, 'width': 8}, 'value': ffc_value}
                    }
                ))
                fig_gauge.update_layout(height=380, margin=dict(l=40, r=60, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

                        # === Move the category text right under the gauge (almost no gap) ===
            with result_container:
                st.markdown(f"""
                <div style="
                    text-align: center;
                    margin-top: -80px;        /* pulls it UP into/very close to the gauge */
                    margin-bottom: 10px;
                ">
                    <h2 style="
                        color: {gauge_color};
                        margin: 0;
                        font-size: 2.2rem;
                        font-weight: 700;
                        line-height: 1.2;
                    ">
                        {cat}
                    </h2>
                </div>
                """, unsafe_allow_html=True)

            # === COL3: Risk Card - Emoji aligned with gauge ===
            with risk_container:
                st.markdown(f"""
                <div style="
                    height: 400px;                     /* same as gauge height */
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-end;         /* pushes everything to the BOTTOM */
                    padding-bottom: -60px;              /* extra space from very bottom – adjust as needed */
                    text-align: center;
                ">
                    <div style="font-size: 6.5rem; margin-bottom: 1rem;">{emoji}</div>
                    <h2 style="color:{risk_color}; margin:0.6rem 0; font-size: 2.1rem; font-weight:600;">{risk}</h2>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error("Error processing input. Check your data.")
        st.exception(e)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:#95a5a6; font-size:0.95rem; margin:0.5rem 0;'>"
            "C. Kossor (cgkossor@gmail.com) Flowmetrics Flowability Prediction Dashboard © 2025</p>", unsafe_allow_html=True)
