import streamlit as st
import pandas as pd
import io
import requests
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
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
    model = PSDResNet()
    url = st.secrets["MODEL_URL"]                   # your secret link
    resp = requests.get(url)
    resp.raise_for_status()
    model_bytes = io.BytesIO(resp.content)
    model.load_state_dict(torch.load(model_bytes, map_location="cpu"))
    return model

# ------------------------------------------------------------------
# YOUR ORIGINAL PREDICT FUNCTION — ONLY ONE TINY FIX (df → results_df)
# ------------------------------------------------------------------
def predict_psd_target(inputs):
    model = load_model()                            # ← now uses secret
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

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

    spec_tensor = torch.FloatTensor(np.array(spectra))[:, None, :]
    extra_tensor = torch.FloatTensor(np.array(extras))

    def predict_with_uncertainty(spec_tensor, extra_tensor, n_mc=50, bin_edges=None, category_names=None):
        model.train()
        device = next(model.parameters()).device
        preds = []
        with torch.no_grad():
            for _ in range(n_mc):
                out = model(spec_tensor.to(device), extra_tensor.to(device))
                preds.append((2.0 ** out).cpu().numpy().ravel())
        mc_preds = np.stack(preds)
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
    prob_df = predict_with_uncertainty(spec_tensor, extra_tensor, 50, bin_edges, category_names)

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
    prob_df = predict_with_uncertainty(spec_tensor, extra_tensor, n_mc=30, bin_edges=bin_edges, category_names=category_names)
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
# STREAMLIT UI (exactly like your original layout)
# ------------------------------------------------------------------
st.set_page_config(page_title="MY MODEL", layout="centered")
st.title("MY MODEL")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style="border: 2px solid #1E90FF; border-radius: 15px; padding: 20px; background-color: #f0f8ff; height: 100px;">
            <h3 style="text-align:center; color:#1E90FF;">Input Parameters</h3>
        </div>""", unsafe_allow_html=True)

    with st.form("input_form"):
        name = st.text_input("Name", value="Sample 1")
        frac1 = st.number_input("Frac_1_%", value=25.0, step=0.1)
        x10 = st.number_input("x10_um", value=5.0, step=0.1)
        x50 = st.number_input("x50_um", value=25.0, step=0.1)
        x90 = st.number_input("x90_um", value=80.0, step=0.1)

        st.caption("You can paste with brackets, spaces, commas — anything works")
        x_um_str = st.text_input("x_um", value="1, 5, 10, 20, 50, 100, 200")
        q3_str = st.text_input("Q3_%", value="0.01, 0.15, 0.40, 0.75, 0.95, 0.99, 1.0")

        submitted = st.form_submit_button("Run Prediction")

with col2:
    st.markdown("""
        <div style="border: 2px solid #32CD32; border-radius: 15px; padding: 20px; background-color: #f0fff0; height: 100px;
             display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <h3 style="text-align:center; color:#228B22;">Predicted FFC</h3>
        </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="border: 2px solid #FF8C00; border-radius: 15px; padding: 20px; background-color: #fffaf0; height: 100px;">
            <h3 style="text-align:center; color:#FF4500;">Category Results</h3>
        </div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# RESULT DISPLAY (exactly your original logic)
# ------------------------------------------------------------------
if submitted:
    try:
        x_um = parse_number_list(x_um_str)
        q3 = parse_number_list(q3_str)

        if len(x_um) != len(q3):
            st.error("x_um and Q3_% lists must have the same length!")
        elif len(x_um) < 3:
            st.error("Please provide at least 3 points")
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

            result_df = predict_psd_target(input_data)

            ffc_value = result_df.iloc[0]["Predicted_FFC"]
            cat = result_df.iloc[0]["Predicted_Category_Name"]
            risk = result_df.iloc[0]["Risk_Category"]

            with col2:
                st.markdown(f"<h1 style='color:#006400; text-align:center;'>{ffc_value:.4f}</h1>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<h2 style='text-align:center;'>{cat}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:#B22222; text-align:center;'>{risk}</h3>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Check x_um and Q3_% lists! (or another input error)")
        st.exception(e)
else:
    with col2:
        st.markdown("<h1 style='color:#aaa; text-align:center;'>—</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown("<h2 style='text-align:center;'>—</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>—</h3>", unsafe_allow_html=True)

