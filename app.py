import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import lognorm
from scipy.optimize import minimize
import plotly.graph_objects as go
import re
import warnings
warnings.filterwarnings("ignore")

# ===================================================================
# 1. CONSTANTS & BIMODAL HELPERS
# ===================================================================
FIXED_EDGES = np.logspace(np.log10(0.5), np.log10(1000), 97)

def bimodal_cdf(x, w1, med1, sigma1, med2, sigma2):
    return w1 * lognorm.cdf(x, s=sigma1, scale=med1) + (1 - w1) * lognorm.cdf(x, s=sigma2, scale=med2)

def resample_q3_parametric(w1, med1, sigma1, med2, sigma2):
    cum = bimodal_cdf(FIXED_EDGES, w1, med1, sigma1, med2, sigma2)
    cum = np.clip(cum, 0, 1)
    density = np.diff(cum)
    density = np.clip(density, 0, None)
    total = density.sum()
    if total > 0:
        density /= total
    return density.astype(np.float32)

# ===================================================================
# 2. MODEL ARCHITECTURE (13 extra features now!)
# ===================================================================
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
        self.fc_extra = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(128 + 64, 1)

    def forward(self, x_spec, x_extra):
        x = self.stem(x_spec)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.fc_cnn(x)
        e = self.fc_extra(x_extra)
        return self.head(torch.cat([x, e], dim=1)).squeeze(-1)

# ===================================================================
# 3. LOAD MODEL (cached + secrets support)
# ===================================================================
@st.cache_resource
def load_model():
    try:
        url = st.secrets["MODEL_URL"]  # e.g. Google Drive direct download link
        response = requests.get(url)
        response.raise_for_status()
        model = PSDResNet()
        state_dict = torch.load(io.BytesIO(response.content), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        st.success("Bimodal model loaded!")
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        raise

# ===================================================================
# 4. FITTING + PREDICTION LOGIC (your new parametric version)
# ===================================================================
def predict_bimodal(inputs):
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accept single dict or list of dicts
    if isinstance(inputs, dict):
        input_list = [inputs]
    else:
        input_list = inputs

    # Constants
    bin_edges = [0.0, 2.0, 4.0, 6.0, 10.0, 100.0]
    category_names = ['Very Cohesive', 'Cohesive', 'Semi Cohesive', 'Easy Flowing', 'Free Flowing']

    def get_fit_setup(x10, x50, x90):
        x10, x50, x90 = float(x10), float(x50), float(x90)
        med_fine_min = max(0.5, x10 * 0.4)
        med_fine_max = x50 * 0.95
        med_coarse_min = max(x50 * 1.1, 10.0)
        med_coarse_max = x90 * 5.0
        sigma_max = 1.45 if (x90 > 400 or x10 < 2.0) else 1.35
        bounds = [
            (0.03, 0.97),
            (np.log(med_fine_min), np.log(med_fine_max + 1e-8)),
            (0.35, sigma_max),
            (np.log(med_coarse_min), np.log(med_coarse_max + 1e-8)),
            (0.35, sigma_max)
        ]
        initial = [0.45, np.log(x50 * 0.4), 0.75, np.log(x50 * 2.5), 0.65]
        return bounds, initial

    def error_function(params, x10, x50, x90):
        w1, ln_med1, sigma1, ln_med2, sigma2 = params
        med1, med2 = np.exp(ln_med1), np.exp(ln_med2)
        Q10 = bimodal_cdf(x10, w1, med1, sigma1, med2, sigma2)
        Q50 = bimodal_cdf(x50, w1, med1, sigma1, med2, sigma2)
        Q90 = bimodal_cdf(x90, w1, med1, sigma1, med2, sigma2)
        err = (Q10 - 0.1)**2 + (Q50 - 0.5)**2 + (Q90 - 0.9)**2
        penalty = 200 * max(0, w1 - 0.92)**2 + 200 * max(0, 0.08 - w1)**2
        penalty += 30 * max(0, sigma1 - 1.35)**2 + 30 * max(0, sigma2 - 1.35)**2
        return err + penalty

    results = []
    for inp in input_list:
        x10 = inp['x10_um']
        x50 = inp['x50_um']
        x90 = inp['x90_um']
        frac = inp.get('Frac_1_%', 25.0)

        # === Fit bimodal parameters ===
        bounds, init = get_fit_setup(x10, x50, x90)
        res = minimize(error_function, init, args=(x10, x50, x90),
                       bounds=bounds, method='L-BFGS-B', options={'maxiter': 1500})
        w1, ln1, s1, ln2, s2 = res.x
        med1, med2 = np.exp(ln1), np.exp(ln2)

        # === Build model inputs ===
        spectrum = resample_q3_parametric(w1, med1, s1, med2, s2)

        span = (x90 - x10) / (x50 + 1e-8)
        ratio = x90 / (x10 + 1e-8)
        extra = np.array([
            np.log10(x10 + 1e-6),
            np.log10(x50 + 1e-6),
            np.log10(x90 + 1e-6),
            span,
            ratio,
            x50 / 10.0,
            x50 / 100.0,
            frac,
            w1,
            np.log10(med1 + 1e-8),
            s1,
            np.log10(med2 + 1e-8),
            s2
        ], dtype=np.float32)

        spec_tensor = torch.FloatTensor(spectrum)[None, None, :].to(device)  # (1,1,96)
        extra_tensor = torch.FloatTensor(extra)[None, :].to(device)          # (1,13)

        # === MC Dropout inference ===
        model.train()
        model.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm1d) else None)
        preds = []
        with torch.no_grad():
            for _ in range(50):
                out = model(spec_tensor, extra_tensor)
                preds.append((2.0 ** out).cpu().item())
        pred_mean = np.mean(preds)
        pred_std = np.std(preds)
        pred_capped = np.clip(pred_mean, 0.0, 10.0)

        # Category
        for i, edge in enumerate(bin_edges[1:], 1):
            if pred_capped < edge:
                category = category_names[i-1]
                break
        else:
            category = 'Free Flowing'

        # Confidence & Risk
        counts, _ = np.histogram(preds, bins=bin_edges)
        probs = counts / 50
        confidence = probs.max()
        rsd_conf = np.clip(1.0 - pred_std / max(pred_mean, 1e-8) * 100 / 100, 0, 1)
        confidence = 0.85 * confidence + 0.15 * rsd_conf
        risk_score = round(10.0 * (1.0 - confidence), 3)
        if confidence >= 0.80:
            risk_cat = "Very Low Risk"
        elif confidence >= 0.65:
            risk_cat = "Low Risk"
        elif confidence >= 0.50:
            risk_cat = "Moderate Risk"
        elif confidence >= 0.35:
            risk_cat = "High Risk"
        else:
            risk_cat = "Critical Risk – Do Not Use Without Review"

        results.append({
            'name': inp.get('name', 'Sample'),
            'x10_um': round(x10, 3),
            'x50_um': round(x50, 3),
            'x90_um': round(x90, 3),
            'Frac_1_%': frac,
            'Predicted_FFC': round(pred_capped, 3),
            'STD': round(pred_std, 3),
            'RSD_%': round(pred_std / max(pred_mean, 1e-8) * 100, 2),
            'Predicted_Category': category,
            'Confidence': round(confidence, 3),
            'Risk_Score': risk_score,
            'Risk_Category': risk_cat,
            'w_fine': round(w1, 3),
            'med_fine': round(med1, 2),
            'sigma_fine': round(s1, 3),
            'med_coarse': round(med2, 2),
            'sigma_coarse': round(s2, 3),
        })

    return pd.DataFrame(results) if len(results) > 1 else results[0]

# ===================================================================
# 5. PAGE CONFIG & STYLING (your beautiful design kept!)
# ===================================================================
st.set_page_config(page_title="Flowmetrics", page_icon="Flowability", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main > div {padding-top: 1rem !important;}
    .block-container {padding-top: 0.5rem !important;}
    .compact-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 45%, #60a5fa 100%);
        color: white; padding: 1.6rem 2.5rem; border-radius: 20px; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30,58,138,0.25), 0 6px 15px rgba(0,0,0,0.15);
        text-align: center;
    }
    .compact-header h1 {margin:0; font-size:2.6rem; font-weight:800;}
    .fixed-card {background:white; padding:1.2rem; border-radius:16px; box-shadow:0 6px 20px rgba(0,0,0,0.08);
                 border:1px solid #eaeaea; min-height:80px; display:flex; flex-direction:column; justify-content:flex-start;}
    .card-title {text-align:center; color:#2c3e50; font-size:1.6rem; font-weight:600; margin:0 0 1.2rem 0;}
    .stButton>button {background:#0066cc; color:white; border-radius:12px; height:3.4em; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="compact-header">
    <h1>Flowmetrics</h1>
    <p>Bimodal PSD → Flowability Prediction (Next-Gen Model)</p>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# 6. SIDEBAR INPUT
# ===================================================================
with st.sidebar:
    st.header("Input Parameters", divider="gray")
    name = st.text_input("Sample Name", "Sample 1", label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1:
        x10 = st.number_input("x10 (µm)", value=5.76, step=0.01, format="%.3f")
        x50 = st.number_input("x50 (µm)", value=19.45, step=0.01, format="%.3f")
    with col2:
        x90 = st.number_input("x90 (µm)", value=43.76, step=0.01, format="%.3f")
    frac = st.number_input("Blend Fraction 1 (%)", value=25.0, step=0.1)
    run = st.button("Run Prediction", type="primary", use_container_width=True)

# ===================================================================
# 7. MAIN LAYOUT
# ===================================================================
c1, c2, c3 = st.columns([1, 1.2, 1])

card1 = c1.container()
card2 = c2.container()
card3 = c3.container()

with card1:
    st.markdown('<div class="fixed-card"><div class="card-title">Reconstructed PSD (Bimodal Fit)</div></div>', unsafe_allow_html=True)
    psd_plot = st.empty()
with card2:
    st.markdown('<div class="fixed-card"><div class="card-title">Predicted Flowability (FFC)</div></div>', unsafe_allow_html=True)
    gauge = st.empty()
    cat_text = st.empty()
with card3:
    st.markdown('<div class="fixed-card"><div class="card-title">Prediction Confidence & Risk</div></div>', unsafe_allow_html=True)
    risk_card = st.empty()

if not run:
    for placeholder in [psd_plot, gauge, risk_card]:
        placeholder.markdown("<div style='text-align:center;padding:6rem 0;color:#95a5a6;'><h3>Waiting…</h3></div>", unsafe_allow_html=True)
else:
    try:
        result = predict_bimodal({
            "name": name,
            "x10_um": x10,
            "x50_um": x50,
            "x90_um": x90,
            "Frac_1_%": frac
        })

        ffc = result['Predicted_FFC']
        cat = result['Predicted_Category']
        risk = result['Risk_Category']

        # Colors
        gauge_color = "#e74c3c" if ffc <= 2 else "#e67e22" if ffc <= 4 else "#f39c12" if ffc <= 6 else "#27ae60"
        emoji = "Critical" in risk and "Critical" or "High" in risk and "High" or "Moderate" in risk and "Moderate" or "Low" in risk and "Low" or "Very Low"

        risk_color = {"Very Low Risk": "#2ecc71", "Low Risk": "#27ae60", "Moderate Risk": "#f39c12",
                      "High Risk": "#e67e22", "Critical Risk – Do Not Use Without Review": "#c0392b"}.get(risk, "#7f8c8d")

        # PSD Plot (reconstructed bimodal)
        w, mf, sf, mc, sc = result['w_fine'], result['med_fine'], result['sigma_fine'], result['med_coarse'], result['sigma_coarse']
        x_fine = np.logspace(-1, 3, 500)
        pdf_fine = w * lognorm.pdf(x_fine, s=sf, scale=mf)
        pdf_coarse = (1-w) * lognorm.pdf(x_fine, s=sc, scale=mc)
        with psd_plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_fine, y=pdf_fine, name="Fine mode", fill='tozeroy', fillcolor='rgba(52,152,219,0.3)'))
            fig.add_trace(go.Scatter(x=x_fine, y=pdf_coarse, name="Coarse mode", fill='tozeroy', fillcolor='rgba(231,76,60,0.3)'))
            fig.add_trace(go.Scatter(x=x_fine, y=pdf_fine+pdf_coarse, name="Total", line=dict(width=4, color='#2c3e50')))
            fig.update_layout(height=420, template="simple_white", xaxis_type="log", xaxis_title="Particle Size (µm)", yaxis_title="Density")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Gauge
        with gauge:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ffc,
                number={'font': {'size': 56, 'color': gauge_color}},
                gauge={'axis': {'range': [1, 10], 'tickvals': [1,2,4,6,10]},
                       'bar': {'color': gauge_color},
                       'bgcolor': "#f8f9fa",
                       'steps': [{'range': [1,2], 'color': '#fce8e8'},
                                 {'range': [2,4], 'color': '#fdebd0'},
                                 {'range': [4,6], 'color': '#fffacd'},
                                 {'range': [6,10], 'color': '#e8f5e9'}],
                       'threshold': {'line': {'color': gauge_color, 'width': 8}, 'value': ffc}
                }))
            fig_g.update_layout(height=380, margin=dict(l=40,r=60,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True, config={'displayModeBar': False})

        with cat_text:
            st.markdown(f"<h2 style='text-align:center;color:{gauge_color};margin-top:-60px'>{cat}</h2>", unsafe_allow_html=True)

        with risk_card:
            st.markdown(f"""
            <div style='height:400px;display:flex;flex-direction:column;justify-content:center;align-items:center;'>
                <div style='font-size:7rem;margin-bottom:1rem;'>{emoji}</div>
                <h2 style='color:{risk_color};margin:0.6rem 0;font-size:2.1rem;'>{risk}</h2>
                <p style='font-size:1.1rem;color:#555;'>Confidence: {result['Confidence']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Optional: show all fitted params in expander
        with st.expander("Show fitted bimodal parameters"):
            st.json({k: result[k] for k in result if k.startswith(('w_','med_','sigma_'))})

    except Exception as e:
        st.error("Prediction failed. Check inputs.")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:#95a5a6;'>C. Kossor © 2025 – Bimodal Flowability Predictor v2</p>", unsafe_allow_html=True)
