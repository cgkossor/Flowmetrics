import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import lognorm
from scipy.optimize import minimize
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.stats import t
from scipy.optimize import brentq
import re
import warnings
import requests
import io
from itertools import product
warnings.filterwarnings("ignore")

# ===================================================================
# 1. CONSTANTS & BIMODAL HELPERS
# ===================================================================
FIXED_EDGES = np.logspace(np.log10(0.5), np.log10(1000), 97)
x_mid = (FIXED_EDGES[:-1] + FIXED_EDGES[1:]) / 2

#CONSTANTS 
Z_0 = 4e-10              # m
D_0 = 1.65e-10           # m
g = 9.81                 # m/s²
Rho_p_default = 1100     # kg/m3
SE_default = 0.40        # J/m2
d_asp_default = 0        # nm


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
        url = st.secrets["MODEL_URL"]  
        response = requests.get(url)
        response.raise_for_status()
        model = PSDResNet()
        state_dict = torch.load(io.BytesIO(response.content), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        st.success("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        raise

# ===================================================================
# 4. FITTING + PREDICTION LOGIC (your new parametric version)
# ===================================================================
def calculate_bond_number(x_mid, vol_den_dist_norm, rho_p, SE, d_asp, Z_0, D_0, g):
    """Calculate Bond number for given size distribution"""
    
    # Basic calculations
    H_0 = Z_0 + (d_asp / 2)
    A = 24 * np.pi * (D_0**2) * SE
    V_p = (4/3) * np.pi * ((x_mid / 2)**3)
    M_p = V_p * rho_p
    W_g = M_p * g
    
    # Distribution calculations
    N_p = vol_den_dist_norm / V_p
    SA_p = 4 * np.pi * ((x_mid / 2)**2)
    SA = N_p * SA_p
    f_SA = SA / np.sum(SA)
    
    # Interaction calculations
    n = len(x_mid)
    Bo_g_ij_Size_Dependent = np.zeros((n, n))
    const1 = A * d_asp / (8 * Z_0**2)                # scalar
    const2 = A / (24 * (2 * d_asp + Z_0)**2)         # scalar
    x_i = x_mid[:, None]
    x_j = x_mid[None, :]
    d_p_ij = 2 * x_i * x_j / (x_i + x_j)
    F_vdw_ij = const1 + const2 * d_p_ij
    W_g_ij = np.sqrt(W_g[:, None] * W_g[None, :])
    Bo_g_ij_Size_Dependent = (f_SA[:, None] * f_SA[None, :]) / (F_vdw_ij / W_g_ij)
    Bo_g = 1 / np.sum(Bo_g_ij_Size_Dependent)
    
    # Additional metrics
    wt_t = N_p * M_p
    Vol = N_p * V_p
    specific_SA = np.sum(SA) / np.sum(wt_t)
    smd = np.sum(Vol) / np.sum(SA)
    
    return Bo_g, specific_SA, smd

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

    results = []
    for inp in input_list:
        x10 = inp['x10_um']
        x50 = inp['x50_um']
        x90 = inp['x90_um']
        frac = inp.get('Frac_1_%', 25.0)
        Rho_p = inp.get('rho_kgm3', Rho_p_default)
        SE = inp.get('SE_jm2', SE_default) 
        d_asp = inp.get('d_asp_nm', d_asp_default) * 1e-9 

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

        # Calcuate Bond Number
        Bo_g, specific_SA, smd_calc = calculate_bond_number(x_mid * 1e-6, spectrum, Rho_p, SE, d_asp, Z_0, D_0, g)    
        pred_capped_Bo = 45.61 * Bo_g ** -0.29

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
            'Predicted_FFC_Bo': round(pred_capped_Bo, 3),
            'Bo_g': round(Bo_g, 3),
            'Specific_SA': round(specific_SA, 3),
            'smd_um': round(smd_calc * 1e6, 3),
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

def predict_psd(data, n_mc=100):

    # === Load model once ===
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === YOUR EXACT CONSTANTS ===
    bin_edges = [0.0, 2.0, 4.0, 6.0, 10.0, 100.0]
    category_names = ['Very Cohesive', 'Cohesive', 'Semi Cohesive', 'Easy Flowing', 'Free Flowing']

    def classify_with_edges(value, edges, names):
            for i, edge in enumerate(edges[1:], start=1):
                if value < edge:
                    return i, names[i-1]
            return len(names), names[-1]

    def confidence_and_risk(row):
        probs = row[[f'Prob_{c.replace(" ", "_")}' for c in category_names]].values
        confidence = probs.max()
        rsd_conf = np.clip(1.0 - row['RSD_%'] / 100.0, 0.0, 1.0)
        confidence = 0.95 * confidence - 0.1 * rsd_conf   # 0.85, 0.15
        risk_score = 10.0 * (1.0 - confidence)
        if confidence >= 0.80:                            # 0.8
            risk_category = "Very Low Risk"
        elif confidence >= 0.65:                          # 0.65
            risk_category = "Low Risk"
        elif confidence >= 0.50:                          # 0.5
            risk_category = "Moderate Risk"
        elif confidence >= 0.40:                          # 0.35
            risk_category = "High Risk"
        else:
            risk_category = "Critical Risk – Do Not Use Without Review"
        return pd.Series({
            'Risk_Score': round(risk_score, 3),
            'Risk_Category': risk_category,
            'Dominant_Category_Probability': round(confidence, 3)
        })

    # === Input to DataFrame ===
    if isinstance(data, dict):
        df = pd.DataFrame([data])
        single = True
    elif isinstance(data, str):
        df = pd.read_csv(data) if data.endswith('.csv') else pd.read_excel(data)
        single = False
    elif hasattr(data, 'columns'):
        df = data.copy()
        single = False
    else:
        raise ValueError("Invalid input")

    N = len(df)
    if 'Frac_1_%' not in df.columns: df['Frac_1_%'] = 25.0
    df['name'] = df['comp_1'] + '_' + df['comp_2']
    # if 'name' not in df.columns: df['name'] = [f"Sample_{i+1}" for i in range(N)]

    # === FAST PATH: use pre-fitted params if exist ===
    if all(c in df.columns for c in ['weight_fine','med_fine','sigma_fine','med_coarse','sigma_coarse']):
        w1 = df['weight_fine'].values
        med_fine = df['med_fine'].values
        s1 = df['sigma_fine'].values
        med_coarse = df['med_coarse'].values
        s2 = df['sigma_coarse'].values
    else:
        def fit_row(r):
            b, i = get_fit_setup(r['x10_um'], r['x50_um'], r['x90_um'])
            res = minimize(error_function, i, args=(r['x10_um'], r['x50_um'], r['x90_um']),
                          bounds=b, method='L-BFGS-B', options={'maxiter': 1500, 'ftol': 1e-15})
            w, l1, s1, l2, s2 = res.x
            return w, np.exp(l1), s1, np.exp(l2), s2
        fitted = df.apply(fit_row, axis=1).tolist()
        w1, med_fine, s1, med_coarse, s2 = map(np.array, zip(*fitted))

    # === Build inputs ===
    spectra = np.stack([resample_q3_parametric(w, mf, sf, mc, sc) 
                       for w,mf,sf,mc,sc in zip(w1, med_fine, s1, med_coarse, s2)])

    d10, d50, d90 = df['x10_um'].values, df['x50_um'].values, df['x90_um'].values
    span = (d90 - d10) / (d50 + 1e-8)
    ratio = d90 / (d10 + 1e-8)

    extras = np.column_stack([
        np.log10(d10 + 1e-6), np.log10(d50 + 1e-6), np.log10(d90 + 1e-6),
        span, ratio, d50/10.0, d50/100.0, df['Frac_1_%'].values,
        w1, np.log10(med_fine + 1e-8), s1, np.log10(med_coarse + 1e-8), s2
    ]).astype(np.float32)

    # === MC Dropout ===
    spec_t = torch.FloatTensor(spectra)[:, None, :].to(device)
    extra_t = torch.FloatTensor(extras).to(device)

    model.train()
    def set_bn_eval(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    model.apply(set_bn_eval)

    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            out = model(spec_t, extra_t)
            preds.append((2.0 ** out).cpu().numpy().ravel())
    mc_preds = np.stack(preds)  # (n_mc, N)

    pred_mean = mc_preds.mean(0)
    pred_std = mc_preds.std(0)
    pred_capped = np.clip(pred_mean, 0.0, 10.0)

    # === Exact same post-processing as your original ===
    results = []
    for i in range(N):
        counts, _ = np.histogram(mc_preds[:, i], bins=bin_edges)
        probs = counts / n_mc
        prob_dict = {f"Prob_{c.replace(' ', '_')}": p for c, p in zip(category_names, probs)}
        dominant_idx = np.argmax(counts)
        cat = category_names[dominant_idx]

        temp_df = pd.DataFrame([{
            'RSD_%': pred_std[i] / max(pred_mean[i], 1e-8) * 100,
            **prob_dict
        }])
        risk = confidence_and_risk(temp_df.iloc[0])

        result = {
            'name': str(df['name'].iloc[i]),
            'x10_um': round(float(d10[i]), 3),
            'x50_um': round(float(d50[i]), 3),
            'x90_um': round(float(d90[i]), 3),
            'Frac_1_%': float(df['Frac_1_%'].iloc[i]),
            'Predicted_FFc': round(pred_capped[i], 4),
            'Predicted_Category': cat,
            'STD': round(pred_std[i], 4),
            'RSD_%': round(pred_std[i] / max(pred_mean[i], 1e-8) * 100, 2),
            'Confidence': round(risk['Dominant_Category_Probability'], 3),
            'Risk_Score': risk['Risk_Score'],
            'Risk_Category': risk['Risk_Category'],
            **prob_dict,
            # 'Fitted_w_fine': round(w1[i], 4),
            # 'Fitted_med_fine': round(med_fine[i], 2),
            # 'Fitted_sigma_fine': round(s1[i], 3),
            # 'Fitted_med_coarse': round(med_coarse[i], 2),
            # 'Fitted_sigma_coarse': round(s2[i], 3),
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    print(f" Processed {len(results_df)} samples")
    if 'target' in df.columns:
        actual_series = df['target'].reindex(results_df.index)
        results_df['Actual_target'] = actual_series.round(3)

        valid = actual_series.notna()
        if valid.any():
            classified = actual_series[valid].apply(lambda x: classify_with_edges(x, bin_edges, category_names))
            cats, names = zip(*classified)
            # results_df.loc[valid, 'Actual_Category_num'] = cats
            results_df.loc[valid, 'Actual_Category'] = names
        else:
            results_df['Actual_Category'] = np.nan
            results_df['Actual_Category_Name'] = np.nan

        matches_pred = (results_df['Actual_Category'] == results_df['Predicted_Category']).sum()
        total = len(results_df)
        print(f"Pred vs Act : {matches_pred}/{total} correct ({matches_pred/total:.1%})")
    else:
        results_df[['Actual_target', 'Actual_Category']] = np.nan

    return results_df if not single else results_df.iloc[0].to_dict()


    if single:
        return results[0]
    else:
        return pd.DataFrame(results)


def load_pure_params():
    try:
        url = st.secrets["PURE_AVG_URL"] 
        response = requests.get(url)
        response.raise_for_status()
        pure_params = pd.read_csv(io.BytesIO(response.content))
        st.success("Material parameters loaded!")
        return pure_params

    except Exception as e:
        st.error(f"Failed to load Excel file: {e}")
        raise

def get_fit_setup(x10, x50, x90):
    x10, x50, x90 = map(float, [x10, x50, x90])
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
    err = (Q10-0.1)**2 + (Q50-0.5)**2 + (Q90-0.9)**2
    penalty = 200*max(0,w1-0.92)**2 + 200*max(0,0.08-w1)**2
    penalty += 30*max(0,sigma1-1.35)**2 + 30*max(0,sigma2-1.35)**2
    return err + penalty

def fit_from_x10x50x90(x10, x50, x90):
    bounds, initial = get_fit_setup(x10, x50, x90)
    res = minimize(error_function, x0=initial, args=(x10,x50,x90),
                bounds=bounds, method='L-BFGS-B',
                options={'ftol':1e-15, 'maxiter':5000})
    w1, ln1, s1, ln2, s2 = res.x
    return [w1, np.exp(ln1), s1, np.exp(ln2), s2]

def predict_blend_exact(frac_A, params_A, params_B):
    alpha = frac_A / 100.0
    def mixed_cdf(x):
        return alpha * bimodal_cdf(x, *params_A) + (1-alpha) * bimodal_cdf(x, *params_B)
    try:
        x10 = brentq(lambda x: mixed_cdf(x) - 0.10, 0.5, 5000)
        x50 = brentq(lambda x: mixed_cdf(x) - 0.50, 1.0, 5000)
        x90 = brentq(lambda x: mixed_cdf(x) - 0.90, 5.0, 5000)
        return fit_from_x10x50x90(x10, x50, x90), [x10, x50, x90]
    except:
        return None

def Blend_Mix(comp1, comp2, n=11):
        
    # if comp1 not in pure_params or comp2 not in pure_params:
    #     raise ValueError(f"One or both materials not found: {comp1}, {comp2}")

    params_A = pure_params[comp1_name]
    params_B = pure_params[comp2_name]

    frac_1 = np.linspace(0, 100, n)
    results = []

    for w in frac_1:
        if w == 0:
            # Pure comp2
            psd_3 = params_B[:3]
            pred_params = params_B[3:]  # [weight_fine, med_fine, sigma_fine, med_coarse, sigma_coarse]
        
        elif w == 100:
            # Pure comp1
            psd_3 = params_A[:3]
            pred_params = params_A[3:]
        
        else:
            # Blend using your exact method
            fitted_result = predict_blend_exact(w, params_A[3:], params_B[3:])
            
            if fitted_result is None or len(fitted_result) != 2:
                st.warning(f"Blend failed at {w}% – using linear interpolation fallback")
                # Linear fallback: interpolate both PSD and params
                alpha = w / 100.0
                psd_3 = [
                    alpha * params_A[0] + (1 - alpha) * params_B[0],
                    alpha * params_A[1] + (1 - alpha) * params_B[1],
                    alpha * params_A[2] + (1 - alpha) * params_B[2],
                ]
                pred_params = [
                    alpha * params_A[3] + (1 - alpha) * params_B[3],
                    alpha * params_A[4] + (1 - alpha) * params_B[4],
                    alpha * params_A[5] + (1 - alpha) * params_B[5],
                    alpha * params_A[6] + (1 - alpha) * params_B[6],
                    alpha * params_A[7] + (1 - alpha) * params_B[7],
                ]
            else:
                pred_params, psd_3 = fitted_result  # This is the correct unpack

        # Unpack predicted params (always 5 values)
        weight_fine_pred, med_fine_pred, sigma_fine_pred, med_coarse_pred, sigma_coarse_pred = pred_params

        results.append({
            'comp_1': comp1,
            'comp_2': comp2,
            'Frac_1_%': w,
            'x10_um': psd_3[0],
            'x50_um': psd_3[1],
            'x90_um': psd_3[2],
            'weight_fine': weight_fine_pred,
            'med_fine': med_fine_pred,
            'sigma_fine': sigma_fine_pred,
            'med_coarse': med_coarse_pred,
            'sigma_coarse': sigma_coarse_pred
        })

    return pd.DataFrame(results)







def calculate_hamaker_constant(gamma, D0=D_0):
    return 24.0 * np.pi * (D0**2) * gamma

def bond_number_single(A, D, rho, d=0.0):
    F_adh = (A*d)/(8*Z_0**2) + (A*D)/(24*(Z_0 + 2*d)**2)
    weight = rho * (np.pi*D**3 / 6.0) * g
    return F_adh / weight if weight != 0 else 0.0

def harmonic_mean(x, y, factor2=False):
    if x + y == 0:
        return 0.0
    if factor2:
        return 2.0 * x * y / (x + y)
    return x * y / (x + y)

def geometric_mean(x, y):
    return np.sqrt(x * y)

def bond_number_cross(comp_i, comp_j):
    A_i, A_j = comp_i['A'], comp_j['A']
    D_i, D_j = comp_i['D'], comp_j['D']
    d_i, d_j = comp_i['d'], comp_j['d']
    rho_i, rho_j = comp_i['rho'], comp_j['rho']

    A_ij   = geometric_mean(A_i, A_j)
    D_ij   = harmonic_mean(D_i, D_j)
    W_i    = rho_i * (np.pi * D_i**3 / 6) * g
    W_j    = rho_j * (np.pi * D_j**3 / 6) * g
    W_ij   = harmonic_mean(W_i, W_j, factor2=True)

    if d_i == 0 and d_j == 0:
        d_small    = 0.0
        sphere_sep = Z_0
    elif d_i > 0 and d_j == 0:
        d_small    = 2.0 * d_i * D_j / (d_i + D_j) if (d_i + D_j) else 0.0
        sphere_sep = Z_0 + d_i
    elif d_i == 0 and d_j > 0:
        d_small    = 2.0 * d_j * D_i / (d_j + D_i) if (d_j + D_i) else 0.0
        sphere_sep = Z_0 + d_j
    else:
        d_small    = harmonic_mean(d_i, d_j)
        sphere_sep = Z_0 + d_i + d_j

    F_adh_ij = (A_ij * d_small) / (8 * Z_0**2) + (A_ij * D_ij) / (24 * sphere_sep**2)
    return F_adh_ij / W_ij if W_ij != 0 else 0.0

# ────────────────────────────────────────────────
#  MULTI-COMPONENT SYSTEM
# ────────────────────────────────────────────────

class MultiComponentSystem:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = []
        self.bond_matrix = None

    def add_component(self, name, rho, D, gamma, d, is_coated=False):
        A = calculate_hamaker_constant(gamma)
        self.components.append({
            'name': name,
            'rho': rho,
            'D': D,
            'gamma': gamma,
            'd': d,
            'A': A,
            'is_coated': is_coated
        })

    def calculate_bond_matrix(self):
        n = len(self.components)
        self.bond_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    c = self.components[i]
                    self.bond_matrix[i,j] = bond_number_single(c['A'], c['D'], c['rho'], c['d'])
                else:
                    self.bond_matrix[i,j] = bond_number_cross(self.components[i], self.components[j])

        # Symmetrize
        for i in range(n):
            for j in range(i+1, n):
                avg = (self.bond_matrix[i,j] + self.bond_matrix[j,i]) / 2
                self.bond_matrix[i,j] = self.bond_matrix[j,i] = avg

        return self.bond_matrix

    def calculate_mixture_bond(self, weight_fractions):
        if self.bond_matrix is None:
            self.calculate_bond_matrix()

        n = len(self.components)
        denom = sum(w / (c['rho'] * c['D']) for w, c in zip(weight_fractions, self.components) if c['D'] > 0)
        if denom == 0:
            return 0.0

        fSA = np.array([
            (w / (c['rho'] * c['D'])) / denom if c['D'] > 0 else 0.0
            for w, c in zip(weight_fractions, self.components)
        ])

        sum_terms = 0.0
        for i in range(n):
            for j in range(n):
                if self.bond_matrix[i,j] > 0 and fSA[i] > 0 and fSA[j] > 0:
                    sum_terms += fSA[i] * fSA[j] / self.bond_matrix[i,j]

        return 1.0 / sum_terms if sum_terms != 0 else 0.0








# ===================================================================
# 5. PAGE CONFIG & STYLING (unchanged – your beautiful design kept!)
# ===================================================================
st.set_page_config(page_title="Flowmetrics", page_icon="Flowability", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Same styles as before */
    .main > div { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0 !important; }

    .compact-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 45%, #60a5fa 100%);
        color: white;
        padding: 1rem 2rem !important;
        border-radius: 16px;
        margin: 0 auto 1.5rem auto;
        max-width: 100%;
        box-shadow: 0 8px 25px rgba(30,58,138,0.3), 0 4px 10px rgba(0,0,0,0.12);
        text-align: center;
    }
    .compact-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .fixed-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.07);
        border: 1px solid #eaeaea;
        min-height: unset !important;
        height: auto;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .card-title {
        text-align: center;
        color: #2c3e50;
        font-size: 1.45rem;
        font-weight: 600;
        margin: 0 0 0.6rem 0;
    }

    .stButton>button {
        background: #0066cc;
        color: white;
        border-radius: 10px;
        height: 2.8em !important;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="compact-header">
    <h1>Flowmetrics</h1>
    <p>Particle Scale Properties to Flowability Prediction</p>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# SIDEBAR: PAGE NAVIGATION
# ===================================================================
with st.sidebar:
    st.header("Prediction Modes", divider="gray")
    
    if st.button("Single", use_container_width=True, type="primary" if st.session_state.get("page", "Single") == "Single" else "secondary"):
        st.session_state.page = "Single"
    
    if st.button("Blends", use_container_width=True, type="primary" if st.session_state.get("page", "Single") == "Blends" else "secondary"):
        st.session_state.page = "Blends"

    if st.button("Dry Coating", use_container_width=True, type="primary" if st.session_state.get("page", "Single") == "Blends" else "secondary"):
        st.session_state.page = "Dry Coating"

# Initialize page
if "page" not in st.session_state:
    st.session_state.page = "Single"

# ===================================================================
# PAGE ROUTING
# ===================================================================
if st.session_state.page == "Single":

# ====================== ULTRA-COMPACT RIBBON - GROUPED COLUMNS ======================
    with st.container():
        st.markdown("""
        <style>
            /* ── Ultra tight global ── */
            .ultra-compact .element-container {
                margin-bottom: 0.12rem !important;
            }
            .ultra-compact label {
                font-size: 0.74rem !important;
                margin-bottom: 0.06rem !important;
                min-height: unset !important;
            }

            /* Model column + run underneath */
            .model-col .stRadio > div {
                flex-direction: column !important;
                gap: 0.18rem !important;
            }
            .model-col label {
                padding: 0.28rem 0.7rem !important;
                font-size: 0.82rem !important;
                min-height: 1.55rem !important;
                line-height: 1.0 !important;
                border-radius: 4px;
                background: white;
                border: 1px solid #d1d5db;
                margin: 0 !important;
            }
            .model-col [data-checked="true"] label {
                background: #dbeafe !important;
                border-color: #2563eb;
                font-weight: 600;
            }

            /* Super narrow inputs */
            .nano-input .stNumberInput > div > div > input,
            .nano-input .stTextInput > div > div > input {
                height: 1.58rem !important;
                font-size: 0.80rem !important;
                padding: 0.12rem 0.38rem !important;
                text-align: center !important;
                width: 68px !important;
                min-width: 60px !important;
                max-width: 80px !important;
            }
            /* Kill +/- */
            .nano-input .stNumberInput [data-testid="stNumberInputStepDown"],
            .nano-input .stNumberInput [data-testid="stNumberInputStepUp"] {
                display: none !important;
            }

            /* Run button - compact & aligned under radios */
            .model-col .run-btn button {
                height: 1.68rem !important;
                font-size: 0.84rem !important;
                padding: 0.1rem 0.8rem !important;
                margin-top: 0.45rem !important;
                width: 100% !important;
                max-width: 110px !important;
            }

            /* Group titles - tiny */
            .group-title {
                font-size: 0.74rem !important;
                color: #6b7280;
                margin: 0.10rem 0 0.22rem 0 !important;
                font-weight: 600;
                letter-spacing: -0.3px;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ultra-compact">', unsafe_allow_html=True)

        # ── Column layout ─────────────────────────────────────────────
        cols = st.columns([22, 1, 14, 14, 1, 14, 14, 1, 18])

        # ── Model + Run + Download ───────────────────────────────
        with cols[0]:
            st.markdown('<div class="model-col">', unsafe_allow_html=True)
            st.markdown('<div class="group-title">Model</div>', unsafe_allow_html=True)
            
            selected_model = st.radio(
                "model",
                ["CNN", "Bond Number", "Both"],
                index=0,
                horizontal=False,
                label_visibility="collapsed",
                key="model_selection"
            )
            
            st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
            run = st.button("Run", type="primary", key="run_prediction")
            st.markdown('</div>', unsafe_allow_html=True)

            # Download button will be placed here conditionally
            download_placeholder = st.empty()

            st.markdown('</div>', unsafe_allow_html=True)

        # ── Material / Conditions ─ split into two narrow columns ──
        with cols[2]:
            st.markdown('<div class="nano-input">', unsafe_allow_html=True)
            st.markdown('<div class="group-title">Material</div>', unsafe_allow_html=True)
            name = st.text_input("Name", "Sample 1", key="sample_name")            
            density     = st.number_input("Density (kg/m³)",     value=1100, step=50,   format="%d")
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[3]:
            st.markdown('<div class="nano-input">', unsafe_allow_html=True)
            st.markdown('<div class="group-title" style="visibility:hidden;">.</div>', unsafe_allow_html=True)  # alignment spacer
            x10  = st.number_input("x10 (µm)", value=5.76,  step=1.0, format="%.2f")
            surf_energy = st.number_input("Surface Energy (J/m²)", value=0.04, step=0.01, min_value=0.0, format="%.3f")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Particle Size ─ split into two narrow columns ────────
        with cols[5]:
            st.markdown('<div class="nano-input">', unsafe_allow_html=True)
            st.markdown('<div class="group-title" style="visibility:hidden;">.</div>', unsafe_allow_html=True)  # alignment
            x50  = st.number_input("x50 (µm)", value=19.45, step=1.0, format="%.2f")
            d_asperity  = st.number_input("Asperity (nm)",    value=0.0,  step=5.0,  min_value=0.0, format="%.1f")
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[6]:
            st.markdown('<div class="nano-input">', unsafe_allow_html=True)
            st.markdown('<div class="group-title" style="visibility:hidden;">.</div>', unsafe_allow_html=True)  # alignment
            x90  = st.number_input("x90 (µm)", value=43.76, step=1.0, format="%.2f")
            frac = st.number_input("Wt. %", value=100, min_value=0, max_value=100, step=1)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close ultra-compact

    # Main 3-column layout
    c1, c2, c3 = st.columns([1, 1.2, 1])

    card1 = c1.container()
    card2 = c2.container()
    card3 = c3.container()

    with card1:
        st.markdown('<div class="fixed-card"><div class="card-title">Reconstructed PSD</div></div>', unsafe_allow_html=True)
        psd_plot = st.empty()
    with card2:
        st.markdown('<div class="fixed-card"><div class="card-title">Predicted Flowability (FFC)</div></div>', unsafe_allow_html=True)
        gauge = st.empty()
        cat_text = st.empty()
    with card3:
        st.markdown('<div class="fixed-card"><div class="card-title">Prediction Confidence</div></div>', unsafe_allow_html=True)
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
                "Frac_1_%": frac,
                "rho_kgm3": density,
                "SE_jm2": surf_energy,
                "d_asp_nm": d_asperity,
            })

            # ── MODEL-BASED FFC SELECTION ────────────────────────────────
            if selected_model == "CNN":
                ffc_value = result['Predicted_FFC']
            elif selected_model == "Bond Number":
                ffc_value = result['Predicted_FFC_Bo']
            elif selected_model == "Both":
                ffc_value = 0.5 * result['Predicted_FFC'] + 0.5 * result['Predicted_FFC_Bo']
            else:
                ffc_value = result['Predicted_FFC']  # fallback

            cat = result['Predicted_Category']
            risk = result['Risk_Category']
            confidence = result['Confidence']

            gauge_color = "#e74c3c" if ffc_value <= 2 else "#e67e22" if ffc_value <= 4 else "#f39c12" if ffc_value <= 6 else "#27ae60"
            risk_colors = {
                "Very Low Risk": "#2ecc71",
                "Low Risk": "#2ecc71",
                "Moderate Risk": "#f39c12",
                "High Risk": "#e67e22",
                "Critical Risk – Do Not Use Without Review": "#c0392b"
            }
            risk_color = risk_colors.get(risk, "#7f8c8d")
            emoji = "🟢" if "Low" in risk else "🟡" if "Moderate" in risk else "🟠" if "High" in risk else "🔴"

            # ── PREPARE EXCEL DOWNLOAD ───────────────────────────────────
            results_df = pd.DataFrame([{
                "Sample_ID": name,
                "Selected_Model": selected_model,
                "d10_um": x10,
                "d50_um": x50,
                "d90_um": x90,
                "Weight_Frac_%": frac,
                "Density_kgm3": density,
                "Surface_Energy_jm2": surf_energy,
                "Asperity_Size_nm": d_asperity,
                'Bond Number': result['Bo_g'],
                'Surface_SA_m2': result['Specific_SA'],
                'SMD_um': result['smd_um'],
                "FFC": round(ffc_value, 2),
                "Predicted_Category" : cat,
                "Risk_Category" : risk,
                "Confidence_%" : int(round(confidence * 100))
            }])

            # Prepare Excel in memory
            output = io.BytesIO()

            # Let pandas choose the engine automatically (will use openpyxl if installed)
            results_df.to_excel(output, index=False, sheet_name="Prediction")

            output.seek(0)

            # Then the download button
            with download_placeholder:
                st.download_button(
                    label="Save to Excel",
                    data=output,
                    file_name=f"{name.replace(' ', '_')}_FFC_Prediction.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_single",
                    help="Download prediction results as Excel file",
                    use_container_width=True,
                    type="secondary"
                )

            # ── PSD Plot ─────────────────────────────────────────────────
            with psd_plot:
                w1 = result['w_fine']
                med1 = result['med_fine']
                sigma1 = result['sigma_fine']
                med2 = result['med_coarse']
                sigma2 = result['sigma_coarse']
                x_um_generated = np.logspace(np.log10(0.5), np.log10(1000), 300)
                q3_generated = bimodal_cdf(x_um_generated, w1, med1, sigma1, med2, sigma2) * 100

                fig_psd = go.Figure()
                fig_psd.add_trace(go.Scatter(x=x_um_generated, y=q3_generated, mode='lines', 
                                           line=dict(color='#3498db', width=4), showlegend=False))
                for val in [x10, x50, x90]:
                    idx = np.argmin(np.abs(x_um_generated - val))
                    y_val = q3_generated[idx]
                    fig_psd.add_trace(go.Scatter(x=[val], y=[y_val], mode='markers',
                                               marker=dict(color="white", size=16, line=dict(color='#3498db', width=3)),
                                               showlegend=False))
                fig_psd.update_layout(height=420, template="simple_white", margin=dict(t=30),
                                    xaxis_type="log", xaxis_title="Particle Size (µm)", 
                                    yaxis_title="CDF (%)", yaxis_range=[0, 105])
                st.plotly_chart(fig_psd, use_container_width=True, config={'displayModeBar': False})

            # ── Gauge + Category ─────────────────────────────────────────
            with gauge:
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

            with cat_text:
                st.markdown(f"""
                <div style="text-align: center; margin-top: -80px;">
                    <h2 style="color: {gauge_color}; margin: 0; font-size: 2.3rem; font-weight: 700;">
                        {cat}
                    </h2>
                </div>
                """, unsafe_allow_html=True)

            # ── Risk Card ────────────────────────────────────────────────
            with risk_card:
                confidence_percent = int(round(confidence * 100))
                st.markdown(f"""
                <div style="height: 450px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                    <div style="font-size: 7.5rem; margin-bottom: 1rem;">{emoji}</div>
                    <div style="color: {risk_color}; font-size: 2.8rem; font-weight: 700; margin: 0.5rem 0; line-height: 1.2;">{risk}</div>
                    <div style="font-size: 2rem; color: #3498db; font-weight: 500; margin-top: 0.3rem;">
                        Confidence: <strong style="font-weight: 800;">{confidence_percent}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error("Prediction failed. Check inputs.")
            st.exception(e)






elif st.session_state.page == "Blends":
    # st.subheader("Blends Prediction")

    # Main layout: 1/3 inputs, 2/3 plot
    col_input, col_plot = st.columns([1, 2])

    with col_input:
        st.markdown("### Input")

        # Load material parameters only once
        if 'pure_params' not in st.session_state:
            with st.spinner("Loading material parameters..."):
                try:
                    pure_avg = load_pure_params()
                    pure_params = {}
                    for _, row in pure_avg.iterrows():
                        name = row['name']
                        pure_params[name] = [
                            float(row['x10_um']),
                            float(row['x50_um']),
                            float(row['x90_um']),
                            float(row['weight_fine']),
                            float(row['med_fine']),
                            float(row['sigma_fine']),
                            float(row['med_coarse']),
                            float(row['sigma_coarse'])
                        ]
                    st.session_state.pure_params = pure_params
                    st.session_state.params_loaded = True  # Flag to show success once
                except Exception as e:
                    st.error("Failed to load material parameters.")
                    st.exception(e)
                    st.stop()
        else:
            pure_params = st.session_state.pure_params

        material_names = sorted(pure_params.keys())

        DEFAULT_A = "Ibuprofen 3"   # Change to your preferred
        DEFAULT_B = "Avicel PH102"       # Change to your preferred

        # Fallback to sorted if not available
        material_names = sorted(pure_params.keys())

        default_a = DEFAULT_A if DEFAULT_A in material_names else material_names[0]
        default_b = DEFAULT_B if DEFAULT_B in material_names and DEFAULT_B != default_a else material_names[1] if len(material_names) > 1 else material_names[0]

        if 'comp1_name' not in st.session_state:
            st.session_state.comp1_name = default_a
        if 'comp2_name' not in st.session_state:
            st.session_state.comp2_name = default_b

        subcol_a, subcol_b = st.columns(2)
        with subcol_a:
            st.markdown("**Part A**")
            comp1_name = st.selectbox(
                "Select Material A",
                options=material_names,
                index=material_names.index(st.session_state.comp1_name),
                key="comp1_select"  # This keeps it in sync
            )
        
        with subcol_b:
            st.markdown("**Part B**")
            comp2_name = st.selectbox(
                "Select Material B",
                options=material_names,
                index=material_names.index(st.session_state.comp2_name),
                key="comp2_select"
            )

        # Run button
        run_blend = st.button("Run Prediction", type="primary", use_container_width=True)

        # === SUCCESS MESSAGES BELOW BUTTON (only once) ===
        if st.session_state.get('params_loaded', False) and not st.session_state.get('params_message_shown', False):
            st.markdown("Material parameters loaded!")  # Optional echo of spinner text
            st.success("Material parameters loaded successfully!")
            st.session_state.params_message_shown = True  # Prevent re-show on rerun

        # === PREDICTION LOGIC ===
        if run_blend:
            if comp1_name == comp2_name:
                st.warning("Please select two different materials.")
            else:
                with st.spinner("Generating blend predictions..."):
                    try:
                        blends_pred = Blend_Mix(comp1_name, comp2_name, n=21)
                        blends_output = predict_psd(blends_pred, n_mc=50)
                        
                        st.session_state.blends_result = blends_output
                        st.session_state.comp1_label = comp1_name.split('_')[0] if '_' in comp1_name else comp1_name
                        st.session_state.comp2_label = comp2_name.split('_')[0] if '_' in comp2_name else comp2_name
                        
                        st.success("Blend prediction complete!")
                        
                    except KeyError as e:
                        st.error(f"Material not found in database: {e}")
                    except Exception as e:
                        st.error("Blend prediction failed.")
                        st.exception(e)

    # === PLOT COLUMN ===
    with col_plot:
        st.markdown("### Predicted Flowability vs Blend Ratio")

        # Vertical centering container for input alignment
        if 'blends_result' not in st.session_state or not run_blend:
            # Placeholder with height to match chart
            st.markdown(
                "<div style='height: 650px; display: flex; align-items: center; justify-content: center;'>"
                "<h3 style='color:#95a5a6; text-align:center;'>"
                "Select two materials and click Run Prediction"
                "</h3></div>",
                unsafe_allow_html=True
            )

        if 'blends_result' in st.session_state and run_blend:
            df = st.session_state.blends_result
            comp1_label = st.session_state.comp1_label
            comp2_label = st.session_state.comp2_label

            fig = go.Figure()

            x = df['Frac_1_%'].values
            y = df['Predicted_FFc'].values
            yerr = df['STD'].values

            # Background bands
            # fig.add_hrect(y0=1, y1=2, fillcolor="#ffcccc", opacity=0.5, line_width=0)
            # fig.add_hrect(y0=2, y1=4, fillcolor="#ffd4b3", opacity=0.5, line_width=0)
            # fig.add_hrect(y0=4, y1=6, fillcolor="#ffffcc", opacity=0.5, line_width=0)
            # fig.add_hrect(y0=6, y1=10, fillcolor="#ccffcc", opacity=0.5, line_width=0)
            # fig.add_hrect(y0=10, y1=16, fillcolor="#ccffcc", opacity=0.8, line_width=0)

            for y_val in [2, 4, 6, 10]:
                fig.add_hline(y=y_val, line=dict(color="gray", width=1, dash="solid"), opacity=0.7)

            # Fit and CI (with clipping as before)
            mask = x > 0
            x_fit = np.linspace(0, 100, 500)

            def power_law(x, a, b):
                return a * np.exp(-b * x)

            if mask.any():
                p0 = (10, 0.05)
                popt, pcov = curve_fit(power_law, x[mask], y[mask], p0=p0, maxfev=1000)
                y_fit = power_law(x_fit, *popt)

                # CI calculation (same as before)
                n = len(x[mask])
                p = len(popt)
                dof = max(n - p, 1)
                t_val = t.ppf(0.975, dof)
                perr = np.sqrt(np.diag(pcov))
                dy = np.zeros_like(x_fit)
                for i in range(len(popt)):
                    dp = np.zeros_like(popt)
                    dp[i] = perr[i]
                    dy += ((power_law(x_fit, *(popt + dp)) - power_law(x_fit, *(popt - dp))) / 2)**2
                dy = np.sqrt(dy) * t_val
                lower_ci = np.clip(y_fit - dy, 0.5, None)
                upper_ci = np.clip(y_fit + dy, None, 20)

                # Confidence band
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_fit, x_fit[::-1]]),
                    y=np.concatenate([upper_ci, lower_ci[::-1]]),
                    fill='toself',
                    fillcolor='gray',
                    opacity=0.3,
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))

                # Trendline
                fig.add_trace(go.Scatter(
                    x=x_fit, y=np.clip(y_fit, None, 20),
                    mode='lines',
                    line=dict(color='black', width=1.5),
                    name='Trendline'
                ))

            # Data points
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(symbol='diamond', size=12, color='black', line=dict(width=1.5)),
                error_y=dict(type='data', array=yerr, color='black', thickness=1.5, width=6),
                name='Predicted FFC'
            ))

            # === FIXED TITLE CENTERING + HEIGHT ===
            fig.update_layout(
                height=650,
                xaxis_title=f"{comp1_label} (wt %)",
                yaxis_title="Predicted FFC",
                title={
                    'text': f"Predicted Flowability: {comp1_label} vs {comp2_label}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                yaxis=dict(
                    type='log',
                    range=[np.log10(1), np.log10(16)],
                    tickvals=[1, 2, 4, 8, 16],
                    ticktext=['1', '2', '4', '8', '16'],
                    fixedrange=True
                ),
                xaxis=dict(range=[-1, 101]),
                showlegend=True,
                template="simple_white",
                margin=dict(l=60, r=40, t=100, b=60),  # Extra top margin for title
                title_font_size=18
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

elif st.session_state.page == "Dry Coating":
    # st.subheader("Dry Coating Applications")
    # ────────────────────────────────────────────────
    #  STREAMLIT APP
    # ────────────────────────────────────────────────

    st.set_page_config(page_title="Dry Coating Applications", layout="wide")
    st.title("Dry Coating Applications")
    st.caption("Compare uncoated / coated scenarios and mixture effective bond numbers")

    # ─── Narrow top section ───────────────────────────────
    left_content, right_empty = st.columns([1.2, 4])

    with left_content:
        n_comp = st.number_input(
            "Number of components in the mixture",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

    # ─── Create tabs for cleaner organization ─────────────────────
    tab_comp, tab_coat, tab_compost, tab_results = st.tabs([
        "Base Properties", "Coating Properties", "Composition", "Results & Download"
    ])

    # ─── Collect base properties ──────────────────────────────────
    with tab_comp:
        st.subheader("Base (uncoated) properties")

        base_data = []
        for i in range(n_comp):
            with st.expander(f"Component {i+1}", expanded=True):
                col1, col2 = st.columns([2,3])
                with col1:
                    name = st.text_input(f"Name", value=f"Cmp {i+1}", key=f"name_base_{i}")
                with col2:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: rho  = st.number_input("Density (kg/m³)", min_value=100.0, value=1200., step=100.0, key=f"rho_{i}")
                    with c2: D    = st.number_input("Diameter (µm)",    min_value=1e-9, value=50.0,  step=5.0, key=f"D_{i}")
                    with c3: gamma= st.number_input("γ (J/m²)",         min_value=0.001, value=0.050, step=0.001, key=f"gamma_{i}")
                    with c4: d    = st.number_input("Asperity d (nm)",   min_value=0.0,   value=0.0,   step=1.0, key=f"d_{i}")

                base_data.append((name, rho, D * 1e-6, gamma, d * 1e-9))

    # ─── Collect coated properties ────────────────────────────────
    with tab_coat:
        st.subheader("Coating properties (applied when coating is active)")

        # ── Reference table ───────────────────────────────────────
        st.markdown("### Typical dry coating reference values")
        ref_data = {
            "Coating Material": ["Silica (SiO₂)", "Silica (SiO₂)"],
            "Type" : ['R972P', 'A200'],
            "Coated γ (J/m²)": [0.0364, 0.044],
            "Coated asperity d (nm)": [20, 12]
        }
        ref_df = pd.DataFrame(ref_data)
        st.dataframe(
            ref_df.style.format({
                "Coated γ (J/m²)": "{:.3f}",
                "Coated asperity d (nm)": "{:.1f}"
            }),
            hide_index=True,
            use_container_width=False   # narrower table
        )

        st.caption("These are example/reference values — feel free to use them or enter your own below.")

        # ── User input for each component's coating ───────────────
        coated_data = []
        for i in range(n_comp):
            name_base = base_data[i][0] if base_data else f"Component {i+1}"
            with st.expander(f"Coating for {name_base}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    gamma_c = st.number_input(
                        "Coated γ (J/m²)",
                        min_value=0.001,
                        value=0.030,
                        step=0.001,
                        key=f"gamma_c_{i}"
                    )
                with c2:
                    d_c_m = st.number_input(
                        "Coated asperity d (nm)",
                        min_value=0.0,
                        value=5.0,
                        step=1.0,
                        key=f"d_c_{i}"
                    )
                    # Optional: show in nm too
                    # st.caption(f"≈ {d_c_m * 1e9:.1f} nm")

                coated_data.append((name_base, gamma_c, d_c_m * 1e-9))


    # ─── Composition ──────────────────────────────────────────────
    with tab_compost:
        st.subheader("Mixture composition (weight %)")
        st.caption("Values are automatically normalized to 100%")

        wt_pcts = []
        cols = st.columns(min(n_comp, 4))
        for i in range(n_comp):
            with cols[i % len(cols)]:
                wt = st.number_input(f"{base_data[i][0]}  wt%", min_value=0.0, max_value=100.0, value=100.0/n_comp, step=0.5, key=f"wt_{i}")
                wt_pcts.append(wt)

        total = sum(wt_pcts)
        if abs(total - 100) > 0.01:
            st.info(f"Weight percentages sum to {total:.1f}% → will be normalized")

        weight_fractions = np.array(wt_pcts) / total if total > 0 else np.ones(n_comp)/n_comp

    # ─── Results & Download ───────────────────────────────────────
    with tab_results:
        if st.button("Calculate all coating scenarios", type="primary", use_container_width=True):
            with st.spinner("Generating all 2ⁿ coating combinations …"):
                scenarios = []
                for pattern in product([False, True], repeat=n_comp):
                    # Build scenario name
                    coated_idx = [k for k, coated in enumerate(pattern) if coated]
                    if not coated_idx:
                        scen_name = "All uncoated"
                    elif len(coated_idx) == n_comp:
                        scen_name = "All coated"
                    else:
                        names = [base_data[k][0] for k in coated_idx]
                        scen_name = f"Coated: {', '.join(names)}"

                    # Build system
                    sys = MultiComponentSystem(n_comp)
                    for k in range(n_comp):
                        if pattern[k]:
                            nm, g_c, d_c = coated_data[k]
                            rho, D = base_data[k][1:3]
                            sys.add_component(f"{nm}_coated", rho, D, g_c, d_c, is_coated=True)
                        else:
                            nm, rho, D, g, d = base_data[k]
                            sys.add_component(nm, rho, D, g, d, is_coated=False)

                    sys.calculate_bond_matrix()
                    B_mix = sys.calculate_mixture_bond(weight_fractions)

                    row = {"Scenario": scen_name, "B_mixture": B_mix}

                    # Add pairwise bond numbers
                    n = n_comp
                    for i in range(n):
                        for j in range(i, n):
                            key = f"B_{i+1}{j+1}" if i==j else f"B_{i+1}-{j+1}"
                            row[key] = sys.bond_matrix[i,j]

                    scenarios.append(row)

                df = pd.DataFrame(scenarios)

                # ─── Display ───────────────────────────────────────
                st.success(f"Calculated {len(df)} scenarios")

                st.subheader("Mixture Bond Number (B_mix)")
                st.dataframe(
                    df[["Scenario", "B_mixture"]].sort_values("B_mixture", ascending=False)
                    .style.format({"B_mixture": "{:.4e}"})
                    .highlight_max(subset="B_mixture", color="#d4f4dd")
                    .highlight_min(subset="B_mixture", color="#ffe6e6"),
                    use_container_width=True
                )

                with st.expander("All pairwise Bond numbers", expanded=False):
                    pair_cols = [c for c in df.columns if c.startswith("B_") and c != "B_mixture"]
                    st.dataframe(
                        df[["Scenario"] + pair_cols]
                        .style.format("{:.3e}", subset=pair_cols),
                        use_container_width=True
                    )

                # ─── Download ──────────────────────────────────────
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv,
                    file_name="bond_number_scenarios.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Quick stats
                st.caption("Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Lowest B_mix", f"{df['B_mixture'].min():.3e}")
                col2.metric("Highest B_mix", f"{df['B_mixture'].max():.3e}")
                col3.metric("Mean B_mix", f"{df['B_mixture'].mean():.3e}")

        else:
            st.info("Press the button above to calculate all coating scenarios.")




# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:#95a5a6; font-size:0.95rem; margin:0.5rem 0;'>"
            "C. Kossor (cgkossor@gmail.com) Flowmetrics Flowability Prediction Dashboard © 2025</p>", unsafe_allow_html=True)
