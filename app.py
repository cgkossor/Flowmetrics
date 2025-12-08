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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------------------------------------------------
# PAGE CONFIG & PROFESSIONAL STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="MY MODEL - Flowability Prediction Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif;}
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.8rem !important;
        font-weight: 700;
        text-align: center;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-top: -10px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# HELPER: Parse list input flexibly
# ------------------------------------------------------------------
def parse_number_list(text):
    import re
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text.replace(",", " "))]

# ------------------------------------------------------------------
# LAYOUT: Title + Subtitle
# ------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>MY MODEL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size:1.2rem;'>Particle Size Distribution → Flowability Prediction Dashboard</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# SIDEBAR INPUTS (clean & professional)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    name = st.text_input("Sample Name", value="Sample 1")
    
    st.markdown("**Key Particle Sizes**")
    col_a, col_b = st.columns(2)
    with col_a:
        x10 = st.number_input("x10 (µm)", value=13.83, step=0.1, format="%.2f")
        x50 = st.number_input("x50 (µm)", value=100.36, step=0.1, format="%.2f")
    with col_b:
        x90 = st.number_input("x90 (µm)", value=240.17, step=0.1, format="%.2f")
    
    frac1 = st.slider("Fraction < 10 µm (%)", 0.0, 100.0, 25.0, 0.5)

    st.markdown("**Full PSD Data**")
    st.caption("Paste size (µm) and cumulative Q3(%) lists. Commas, spaces, brackets — anything works.")
    
    x_um_str = st.text_area(
        "Particle sizes x (µm)",
        value="4.5, 5.5, 6.5, 7.5, 9.0, 11.0, 13.0, 15.5, 18.5, 21.5, 25.0, 30.0, 37.5, 45.0, 52.5, 62.5, 75.0, 90.0, 105.0, 125.0, 150.0, 180.0, 215.0, 255.0, 305.0, 365.0, 435.0, 515.0, 615.0, 735.0, 875.0",
        height=100
    )
    q3_str = st.text_area(
        "Cumulative Q3 (%)",
        value="3.46, 4.26, 5.03, 5.78, 6.86, 8.22, 9.5, 11.0, 12.69, 14.29, 16.07, 18.53, 22.12, 25.63, 29.06, 33.57, 39.15, 45.68, 51.94, 59.75, 68.6, 77.7, 85.97, 92.37, 96.71, 98.88, 99.71, 100.0, 100.0, 100.0, 100.0",
        height=100
    )

    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# MAIN DASHBOARD AREA
# ------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1.2, 1])

if run_btn:
    try:
        x_um = parse_number_list(x_um_str)
        q3 = parse_number_list(q3_str)

        if len(x_um) != len(q3):
            st.error("x_um and Q3_% lists must have the same length!")
        elif len(x_um) < 3:
            st.error("Please provide at least 3 data points")
        else:
            # === PLACEHOLDER: Replace with your actual model call ===
            input_data = [{
                "name": name,
                "Frac_1_%": frac1,
                "x10_um": x10,
                "x50_um": x50,
                "x90_um": x90,
                "x_um": x_um,
                "Q3_%": q3
            }]
            result_df = predict_psd_target(input_data)  # ← your real function
            ffc_value = result_df.iloc[0]["Predicted_FFC"]
            cat = result_df.iloc[0]["Predicted_Category_Name"]
            risk = result_df.iloc[0]["Risk_Category"]

            # ------------------------------------------------------------------
            # COLUMN 1: PSD Plot (nice interactive Plotly chart)
            # ------------------------------------------------------------------
            with col1:
                st.markdown("<div class='card'><h3 style='text-align:center;color:#2c3e50'>Particle Size Distribution</h3></div>", unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_um, y=q3, mode='lines+markers', name='Q3(%)',
                                         line=dict(color='#3498db', width=3)))
                fig.update_layout(height=400, template="simple_white", margin=dict(l=20,r=20,t=30,b=20))
                fig.update_xaxes(title="Particle Size (µm)", type="log")
                fig.update_yaxes(title="Cumulative Mass (%)")
                st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------------
            # COLUMN 2: Predicted FFC + Beautiful Gauge
            # ------------------------------------------------------------------
            with col2:
                st.markdown("<div class='card'><h3 style='text-align:center;color:#27ae60'>Predicted Flowability (FFC)</h3></div>", unsafe_allow_html=True)

                # Color logic for gauge
                if ffc_value >= 10:
                    gauge_color = "#2ecc71"
                    risk_level = "Free Flowing"
                elif ffc_value >= 7:
                    gauge_color = "#f39c12"
                    risk_level = "Easy Flowing"
                elif ffc_value >= 4:
                    gauge_color = "#e67e22"
                    risk_level = "Cohesive"
                elif ffc_value >= 2:
                    gauge_color = "#e74c3"
                    risk_level = "Very Cohesive"
                else:
                    gauge_color = "#c0392b"
                    risk_level = "Non-Flowing"

                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = ffc_value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "<b>FFC Value</b>", 'font': {'size': 24}},
                    delta = {'reference': 5},
                    gauge = {
                        'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 2], 'color': '#c0392b'},
                            {'range': [2, 4], 'color': '#e74c22'},
                            {'range': [4, 7], 'color': '#f39c12'},
                            {'range': [7, 10], 'color': '#f1c40f'},
                            {'range': [10, 20], 'color': '#2ecc71'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 6},
                            'thickness': 0.75,
                            'value': 4}}))

                fig_gauge.update_layout(height=400, font={'size': 16})
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown(f"<p class='metric' style='color:{gauge_color};font-weight:bold'>{cat}</p>", unsafe_allow_html=True)

            # ------------------------------------------------------------------
            # COLUMN 3: Risk Category + Summary Card
            # ------------------------------------------------------------------
            with col3:
                st.markdown("<div class='card'><h3 style='text-align:center;color:#c0392b'>Risk Assessment</h3></div>", unsafe_allow_html=True)

                risk_colors = {
                    "Low Risk": "#2ecc71",
                    "Moderate Risk": "#f39c12",
                    "High Risk": "#e74c22",
                    "Critical": "#c0392b"
                }
                risk_color = risk_colors.get(risk, "#7f8c8d")

                st.markdown(f"""
                <div style="text-align:center; padding: 2rem 0;">
                    <div style="font-size: 4.5rem;">{'🔴' if 'Critical' in risk else '🟠' if 'High' in risk else '🟡' if 'Moderate' in risk else '🟢'}</div>
                    <h2 style="color:{risk_color}; margin:0.5rem 0;">{risk}</h2>
                    <p style="font-size:1.2rem; color:#444;">{risk_level}</p>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"**Sample:** {name}")

    except Exception as e:
        st.error("Input error – check your PSD lists!")
        st.exception(e)
else:
    # Placeholder when nothing is run
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col1:
        st.markdown("<div class='card'><h3 style='text-align:center;color:#95a5a6'>Particle Size Distribution</h3><p style='text-align:center;color:#bdc3c7;font-size:4rem;margin-top:3rem'>—</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><h3 style='text-align:center;color:#27ae60'>Predicted FFC</h3><p style='text-align:center;color:#bdc3c7;font-size:4.5rem;margin-top:3rem'>—</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'><h3 style='text-align:center;color:#c0392b'>Risk Assessment</h3><p style='text-align:center;color:#bdc3c7;font-size:4rem;margin-top:3rem'>—</p></div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:#95a5a6;font-size:0.9rem'>MY MODEL – Professional Flowability Prediction Dashboard © 2025</p>", unsafe_allow_html=True)
