import streamlit as st
import pandas as pd
import io
import requests
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
import ast
import warnings
import re  # ← This is part of Python's standard library → NO need to add to requirements.txt

warnings.filterwarnings("ignore")

FIXED_EDGES = np.logspace(np.log10(0.5), np.log10(1000), 97)

# === PAGE CONFIG ===
st.set_page_config(page_title="MY MODEL", layout="centered")
st.title("MY MODEL")

# === MODEL CLASS (must be defined BEFORE loading) ===
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
    def forward(self, x): 
        return self.relu(x + self.net(x))

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

# === MODEL LOADING ===
@st.cache_resource
def load_model():
    model = PSDResNet()  # ← This was missing before!
    url = st.secrets["MODEL_URL"]  # Make sure you set this in secrets.toml
    resp = requests.get(url)
    resp.raise_for_status()
    model_bytes = io.BytesIO(resp.content)
    model.load_state_dict(torch.load(model_bytes, map_location="cpu"))
    model.eval()
    return model

# === ROBUST LIST PARSING ===
def parse_number_list(text):
    """Convert user input like '[1, 2, 3]', '1 2 3', or '1,2,3' → list of floats"""
    if not text or not text.strip():
        return []
    # Extract all numeric tokens (int or float, including scientific notation)
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    return [float(x) for x in numbers]

# === RESAMPLE FUNCTION ===
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

# === PREDICTION FUNCTION (fixed & simplified) ===
@st.cache_data
def predict_psd_target(_model, inputs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(device)

    spectra = []
    extras = []

    for inp in inputs:
        x_um = np.array(inp["x_um"], dtype=float)
        q3 = np.array(inp["Q3_%"], dtype=float)

        if len(x_um) != len(q3):
            raise ValueError("x_um and Q3_% must have same length")

        spectrum = resample_q3(x_um, q3)
        spectra.append(spectrum)

        d10 = float(inp.get("x10_um", 5.0))
        d50 = float(inp.get("x50_um", 50.0))
        d90 = float(inp.get("x90_um", 200.0))
        span = (d90 - d10) / (d50 + 1e-8)
        ratio = d90 / (d10 + 1e-8)

        extra = np.array([
            np.log10(d10 + 1e-6),
            np.log10(d50 + 1e-6),
            np.log10(d90 + 1e-6),
            span,
            ratio,
            d50 / 10.0,
            d50 / 100.0,
            inp.get("Frac_1_%", 25.0)
        ], dtype=np.float32)
        extras.append(extra)

    spec_tensor = torch.FloatTensor(np.array(spectra))[:, None, :]   # (N, 1, 96)
    extra_tensor = torch.FloatTensor(np.array(extras))               # (N, 8)

    # Monte Carlo Dropout for uncertainty
    _model.train()  # Keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(100):
            out = _model(spec_tensor.to(device), extra_tensor.to(device))
            preds.append((2.0 ** out).cpu().numpy().ravel())
    mc_preds = np.stack(preds)
    mean_pred = mc_preds.mean(axis=0)
    std_pred = mc_preds.std(axis=0)

    # Simple category mapping
    bin_edges = [0.0, 2.0, 4.0, 6.0, 10.0, 100.0]
    categories = ['Very Cohesive', 'Cohesive', 'Semi Cohesive', 'Easy Flowing', 'Free Flowing']

    def get_category(val):
        for i, edge in enumerate(bin_edges[1:]):
            if val < edge:
                return categories[i]
        return categories[-1]

    results = []
    for i in range(len(mean_pred)):
        ffc = mean_pred[i]
        cat = get_category(ffc)
        risk = "Low Risk" if std_pred[i] / max(ffc, 1e-8) < 0.1 else "Check Input"
        results.append({
            "Predicted_FFC": round(ffc, 4),
            "Predicted_Category": cat,
            "Risk_Category": risk
        })

    return pd.DataFrame(results)

# === LAYOUT ===
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div style='border: 2px solid #1E90FF; border-radius: 15px; padding: 20px; background-color: #f0f8ff; height: 560px;'>"
                "<h3 style='text-align:center; color:#1E90FF;'>Input Parameters</h3>", unsafe_allow_html=True)

    with st.form("input_form"):
        name = st.text_input("Name", value="Sample 1")
        frac1 = st.number_input("Frac_1_%", value=25.0, step=0.1)
        x10 = st.number_input("x10_um", value=5.0, step=0.1)
        x50 = st.number_input("x50_um", value=25.0, step=0.1)
        x90 = st.number_input("x90_um", value=80.0, step=0.1)

        st.info("You can use spaces, commas, brackets → all work!")
        x_um_str = st.text_input("x_um (diameter points)", value="1, 5, 10, 20, 50, 100, 200")
        q3_str = st.text_input("Q3_% (cumulative)", value="0.01, 0.15, 0.40, 0.75, 0.95, 0.99, 1.0")

        submitted = st.form_submit_button("Run Prediction")

    st.markdown("</div>", unsafe_allow_html=True)

# === Prediction logic ===
result_df = None
if submitted:
    try:
        x_um = parse_number_list(x_um_str)
        q3 = parse_number_list(q3_str)

        if len(x_um) != len(q3):
            st.error(f"Length mismatch: x_um has {len(x_um)} values, Q3_% has {len(q3)}")
        elif len(x_um) < 3:
            st.error("Please enter at least 3 points")
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

            model = load_model()
            result_df = predict_psd_target(model, input_data)

    except Exception as e:
        st.error(f"Input error: {e}")

with col2:
    st.markdown("<div style='border: 2px solid #32CD32; border-radius: 15px; padding: 20px; background-color: #f0fff0; height: 560px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>"
                "<h3 style='text-align:center; color:#228B22;'>Predicted FFC</h3>", unsafe_allow_html=True)

    if result_df is not None:
        ffc = result_df.iloc[0]["Predicted_FFC"]
        st.markdown(f"<h1 style='color:#006400;'>{ffc:.4f}</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='color:#aaa;'>—</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div style='border: 2px solid #FF8C00; border-radius: 15px; padding: 20px; background-color: #fffaf0; height: 560px;'>"
                "<h3 style='text-align:center; color:#FF4500;'>Category Results</h3>", unsafe_allow_html=True)

    if result_df is not None:
        cat = result_df.iloc[0]["Predicted_Category"]
        risk = result_df.iloc[0]["Risk_Category"]
        st.markdown(f"<h2>{cat}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#B22222;'>{risk}</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>—</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


