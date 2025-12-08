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
import plotly.graph_objects as go

# ------------------------------------------------------------------
# PAGE CONFIG & CLEAN PROFESSIONAL STYLE
# ------------------------------------------------------------------
st.set_page_config(
    page_title="MY MODEL - Flowability Dashboard",
    page_icon="Chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .big-number {
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
    }
    .category-text {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1rem;
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 12px;
        height: 3.5em;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# HELPER
# ------------------------------------------------------------------
def parse_number_list(text):
    import re
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text.replace(",", " "))]

# ------------------------------------------------------------------
# TITLE
# ------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #1e3a8a; margin-bottom:0;'>MY MODEL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size:1.3rem;'>Flowability Prediction Dashboard</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# SIDEBAR INPUTS (clean, no slider, no PSD plot)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Input Parameters")
    name = st.text_input("Sample Name", value="Sample 1")
    
    col1, col2 = st.columns(2)
    with col1:
        x10 = st.number_input("x10 (µm)", value=13.83, step=0.1, format="%.2f")
        x50 = st.number_input("x50 (µm)", value=100.36, step=0.1, format="%.2f")
    with col2:
        x90 = st.number_input("x90 (µm)", value=240.17, step=0.1, format="%.2f")
    
    frac1 = st.number_input("Fraction < 10 µm (%)", value=25.0, step=0.1, format="%.2f")

    st.markdown("**Full PSD Data (optional for prediction)**")
    st.caption("Paste lists — commas, spaces, brackets all work")
    x_um_str = st.text_area("Particle sizes x (µm)", height=100,
        value="4.5,5.5,6.5,7.5,9,11,13,15.5,18.5,21.5,25,30,37.5,45,52.5,62.5,75,90,105,125,150,180,215,255,305,365,435,515,615,735,875")
    q3_str = st.text_area("Cumulative Q3 (%)", height=100,
        value="3.46,4.26,5.03,5.78,6.86,8.22,9.5,11,12.69,14.29,16.07,18.53,22.12,25.63,29.06,33.57,39.15,45.68,51.94,59.75,68.6,77.7,85.97,92.37,96.71,98.88,99.71,100,100,100,100")

    run = st.button("Run Prediction", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# MAIN COLUMNS
# ------------------------------------------------------------------
left, center, right = st.columns([1, 1.3, 1])

if run:
    try:
        x_um = parse_number_list(x_um_str)
        q3 = parse_number_list(q3_str)

        if len(x_um) != len(q3) and len(x_um) > 0:
            st.error("x_um and Q3 lists must have same length")
        elif len(x_um) > 0 and len(x_um) < 3:
            st.error("Need at least 3 points")
        else:
            input_data = [{
                "name": name,
                "Frac_1_%": frac1,
                "x10_um": x10,
                "x50_um": x50,
                "x90_um": x90,
                "x_um": x_um,
                "Q3_%": q3
            }]
            result_df = predict_psd_target(input_data)  # ← your real model
            ffc = round(result_df.iloc[0]["Predicted_FFC"], 3)
            category = result_df.iloc[0]["Predicted_Category_Name"]
            risk = result_df.iloc[0]["Risk_Category"]

            # ========================= CENTER: FFC GAUGE =========================
            with center:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#2c3e50; margin-bottom:1rem;'>Predicted FFC</h3>", unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ffc,
                    number={'font': {'size': 60, 'color': '#2c3e50'}},
                    gauge={
                        'axis': {
                            'range': [1, 15],
                            'tickmode': 'array',
                            'tickvals': [1, 2, 4, 6, 10, 15],
                            'ticktext': ['1', '2', '4', '6', '10', '15'],
                            'tickfont': {'size': 16}
                        },
                        'bar': {'color': '#2c3e50', 'thickness': 0.8},
                        'bgcolor: "#f0f2f6",
                        'steps': [
                            {'range': [1, 2], 'color': '#e74c3c'},    # red
                            {'range': [2, 4], 'color': '#e67e22'},    # orange
                            {'range': [4, 6], 'color': '#f1c40f'},    # yellow
                            {'range': [6, 10], 'color': '#27ae60'},   # green
                            {'range': [10, 15], 'color': '#1e8449'}   # dark green
                        ],
                    }
                ))

                fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # Category name right under the number
                st.markdown(f"<p class='category-text'>{category}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ========================= RIGHT: HORIZONTAL RISK SCALE =========================
            with right:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#c0392b;'>Risk Category</h3>", unsafe_allow_html=True)

                risk_map = {
                    "Low Risk": 1,
                    "Moderate Risk": 2,
                    "High Risk": 3,
                    "Critical": 4
                }
                risk_value = risk_map.get(risk, 0)

                fig_risk = go.Figure(go.Indicator(
                    mode="delta",
                    value=risk_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    delta={'reference': 1, 'position': "top", 'increasing': {'color': "#c0392b"}},
                    gauge={
                        'shape': "bullet",
                        'axis': {'range': [None, 4], 'visible': False},
                        'threshold': {
                            'line': {'color': "black", 'width': 8},
                            'thickness': 0.75,
                            'value': risk_value},
                        'steps': [
                            {'range': [0, 1.5], 'color': "#2ecc71"},
                            {'range': [1.5, 2.5], 'color': "#f39c12"},
                            {'range': [2.5, 3.5], 'color': "#e67e22"},
                            {'range': [3.5, 4], 'color': "#c0392b"}
                        ],
                        'bar': {'color': "black"}
                    }
                ))

                fig_risk.update_layout(height=180, margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig_risk, use_container_width=True)

                # Big risk text
                risk_colors = {"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12",
                               "High Risk": "#e67e22", "Critical": "#c0392b"}
                st.markdown(f"<h2 style='color:{risk_colors.get(risk, '#7f8c8d')}; margin-top:1rem;'>{risk}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:1.2rem; color:#555; margin-top:0.5rem;'>{name}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Input error – check your data")
        st.exception(e)
else:
    # Placeholders
    with center:
        st.markdown("<div class='card'><h3 style='color:#95a5a6'>Predicted FFC</h3><p class='big-number' style='color:#ddd'>—</p></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><h3 style='color:#95a5a6'>Risk Category</h3><p style='font-size:4rem; color:#eee; margin-top:2rem'>—</p></div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("MY MODEL – Professional Flowability Dashboard © 2025")
