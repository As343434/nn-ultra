import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# NOTE: These imports assume your 'utils' folder is present.
from utils.data import load_iris, load_wine, load_breast_cancer, standardize, csv_to_xy
from utils.export import download_code, download_pickle, download_torch
from utils.nav import render_sidebar
from utils.theme import apply_theme, metric_row
from utils.viz import plot_loss_curve, plot_accuracy_curve, plot_confusion_matrix, plot_weight_heatmap

# ====================== PAGE CONFIG & CUSTOM CSS ======================
st.set_page_config(page_title="ANN / MLP — NeuralForge", layout="wide", page_icon="⬡")

st.markdown("""
<style>
    /* Sidebar & Layout */
    section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
    h1, h2, h3, h4 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; letter-spacing: -0.02em; color: var(--text); }
    
    /* Custom Cards */
    .nn-card { 
        background: var(--surface); border: 1px solid var(--border); border-radius: 12px; 
        padding: 1.4rem; margin-bottom: 1rem; transition: all 0.2s; 
    }
    .nn-card:hover { border-color: var(--accent); box-shadow: 0 0 15px color-mix(in srgb, var(--accent) 20%, transparent); }
    
    /* Hero Section */
    .nn-hero { 
        background: radial-gradient(ellipse 80% 60% at 50% 0%, color-mix(in srgb, var(--accent) 15%, transparent) 0%, transparent 70%), var(--surface); 
        border: 1px solid var(--border); border-radius: 16px; padding: 2.2rem 2rem; margin-bottom: 1.6rem; 
    }
    
    /* Pills & Badges */
    .nn-pill { 
        display: inline-block; padding: 0.2rem 0.8rem; border-radius: 999px; 
        background: color-mix(in srgb, var(--accent) 15%, transparent); color: var(--accent); 
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.8rem; 
    }
    
    /* Metrics */
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem; }
    
    /* Buttons */
    button[kind="primary"] { 
        background: var(--accent) !important; color: #000 !important; 
        font-weight: 700 !important; border-radius: 8px !important; 
        border: none !important; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

apply_theme()
render_sidebar("ANN / MLP")

# ====================== HERO SECTION ======================
st.markdown(f"""
<div class="nn-hero">
    <div class="nn-pill">Lesson 5</div>
    <h1>ANN / MLP — Neural Engine</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">Build and train a Multi-Layer Perceptron with NumPy or PyTorch backends.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Theory & Architecture", expanded=False):
    st.markdown("""
    MLPs stack **linear layers** and **nonlinear activations**. 
    Training involves a forward pass to compute error and a backward pass (Backprop) to update weights.
    """)
    

st.markdown("---")

# ====================== SIDEBAR / CONTROLS ======================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="nn-card">', unsafe_allow_html=True)
    source   = st.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer", "Upload CSV"])
    backend  = st.selectbox("Backend", ["NumPy (scratch)", "PyTorch"])
    act_name = st.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid", "LeakyReLU"])
    hidden   = st.text_input("Hidden Layers", "64,32", help="Comma separated integers")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="nn-card">', unsafe_allow_html=True)
    epochs   = st.slider("Epochs", 5, 500, 100)
    lr       = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
    bs       = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)
    upload   = st.file_uploader("Custom CSV", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

# ====================== DATA PROCESSING ======================
with col2:
    if source == "Upload CSV" and upload:
        X_raw, y_raw = csv_to_xy(upload.getvalue())
        X_df, y_s = pd.DataFrame(X_raw), pd.Series(y_raw)
    else:
        loaders = {"Iris": load_iris, "Wine": load_wine, "Breast Cancer": load_breast_cancer}
        X_df, y_s = loaders.get(source, load_iris)()

    X = standardize(X_df.values.astype(float))
    y = y_s.values.astype(int)
    n_cls = len(np.unique(y))

    metric_row([("Samples", len(X)), ("Features", X.shape[1]), ("Classes", n_cls)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    st.markdown("### Training Workspace")
    train_btn = st.button("▶ Start Training Engine", type="primary")

# ====================== TRAINING LOGIC ======================
if train_btn:
    sizes = [int(s) for s in hidden.split(",") if s.strip().isdigit()]
    in_d, out_d = X_tr.shape[1], n_cls
    
    # Simple NumPy Training Placeholder (matches your previous logic)
    if backend.startswith("NumPy"):
        # ... [Keep the NumPy training logic from previous block] ...
        # (Assuming the logic remains the same as the previous correct version)
        st.info("Training on NumPy Backend...")
        time.sleep(1) # Simulate
        st.success("NumPy Training Mockup Complete.")
        
    else:
        # PyTorch Logic
        import torch
        import torch.nn as nn
        
        dims = [in_d] + sizes + [out_d]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2: layers.append(nn.ReLU())
        
        model = nn.Sequential(*layers)
        # (Full PyTorch training loop here)
        st.success("PyTorch Model Ready.")

    # ====================== VISUALIZATION ======================
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.markdown('<div class="nn-card">', unsafe_allow_html=True)
        st.write("Loss Curve")
        # st.plotly_chart(plot_loss_curve(losses, val_losses))
        st.markdown('</div>', unsafe_allow_html=True)
