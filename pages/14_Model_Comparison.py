import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Model Comparison",
    layout="wide",
    page_icon="📊"
)

# ====================== SAFE CSS ======================
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    h1, h2, h3, h4 {
        font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    .nn-card {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1.4rem !important;
        margin-bottom: 1rem !important;
    }
    .nn-hero {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 2.2rem 2rem !important;
        margin-bottom: 1.6rem !important;
    }
    .nn-pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        background: rgba(249, 115, 22, 0.15) !important;
        color: #f97316 !important;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .status-running { background: rgba(249,115,22,0.2); color: #f97316; padding: 0.4rem 0.8rem; border-radius: 999px; font-weight: 600; }
    .status-done   { background: rgba(34,197,151,0.2); color: #22c597; padding: 0.4rem 0.8rem; border-radius: 999px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("📊 Model Comparison")
    st.markdown("Benchmark Dashboard")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 14</div>
    <h1>Model Comparison Dashboard</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Side-by-side benchmark of multiple neural architectures.<br>
        Accuracy, F1, parameters, training time, memory — all in one radar chart.
    </p>
</div>
""", unsafe_allow_html=True)

# ====================== HELPERS ======================
def relu(x):
    return np.maximum(0, x)

def train_and_evaluate(X_tr, y_tr, X_te, y_te, layers, lr=0.05, epochs=200, name="model"):
    np.random.seed(42)
    n_classes = len(np.unique(y_tr))
    weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) 
               for i in range(len(layers)-1)]
    biases = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]
    
    params = sum(w.size + b.size for w, b in zip(weights, biases))
    t0 = time.time()
    
    for _ in range(epochs):
        a = X_tr
        acts = [a]
        for i, (W, b) in enumerate(zip(weights, biases)):
            z = a @ W + b
            a = relu(z) if i < len(weights)-1 else z
            acts.append(a)
        
        ex = np.exp(a - a.max(1, keepdims=True))
        probs = ex / ex.sum(1, keepdims=True)
        yoh = np.eye(n_classes)[y_tr]
        
        dz = (probs - yoh) / len(X_tr)
        for i in range(len(weights)-1, -1, -1):
            dW = acts[i].T @ dz
            db = dz.sum(0)
            weights[i] -= lr * dW
            biases[i] -= lr * db
            if i > 0:
                da = dz @ weights[i].T
                da[acts[i] <= 0] = 0
                dz = da
    
    elapsed = time.time() - t0
    
    # Test evaluation
    a = X_te
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        a = relu(z) if i < len(weights)-1 else z
    preds = np.argmax(a, axis=1)
    
    acc = np.mean(preds == y_te)
    f1 = f1_score(y_te, preds, average="macro", zero_division=0)
    mem_kb = params * 4 / 1024  # float32
    
    return {
        "name": name,
        "accuracy": acc,
        "f1": f1,
        "params": params,
        "train_time": elapsed,
        "memory_kb": mem_kb
    }

# ====================== CONFIG ======================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Benchmark Config")
    dataset = st.selectbox("Dataset", ["moons", "2-class", "5-class"])
    n_samples = st.slider("Samples", 200, 2000, 600, 100)
    epochs = st.slider("Epochs per model", 50, 500, 180, 50)
    run_btn = st.button("📊 Run Full Benchmark", type="primary", use_container_width=True)

with col2:
    st.markdown("### 🏗️ Models to Compare")
    st.markdown("_(predefined architectures — you can edit them in the code)_")
    models_cfg = [
        ("Tiny MLP", [8]),
        ("Shallow Wide", [64]),
        ("Deep Narrow", [16, 16, 16]),
        ("Balanced", [32, 32]),
        ("Deep Wide", [64, 64, 64]),
    ]
    for name, hidden in models_cfg:
        st.markdown(f"""
        <div style="display:inline-block;margin:4px;padding:6px 12px;border-radius:8px;
                    background:var(--surface);border:1px solid var(--border);font-size:0.85rem">
            <b>{name}</b> — {hidden}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ====================== BENCHMARK ======================
if run_btn:
    np.random.seed(42)
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset == "2-class":
        X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2,
                                   n_informative=5, random_state=42)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=15, n_classes=5,
                                   n_informative=10, random_state=42)
    
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    n_in = X.shape[1]
    n_out = len(np.unique(y))
    
    bar = st.progress(0)
    status = st.empty()
    results = []
    
    for i, (name, hidden) in enumerate(models_cfg):
        status.markdown(f'<span class="status-running">Training {name}...</span>', unsafe_allow_html=True)
        layers = [n_in] + hidden + [n_out]
        res = train_and_evaluate(X_tr, y_tr, X_te, y_te, layers, epochs=epochs, name=name)
        results.append(res)
        bar.progress((i + 1) / len(models_cfg))
    
    status.markdown('<span class="status-done">✓ Benchmark Complete!</span>', unsafe_allow_html=True)

    # ====================== SUMMARY ======================
    st.markdown("### 📋 Results Summary")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        col.markdown(f"""
        <div class="nn-card" style="text-align:center">
            <div style="font-size:1.1rem;font-weight:700;color:#00d4aa">{r['name']}</div>
            <div style="font-size:0.95rem;margin:8px 0">
                Acc: <b>{r['accuracy']:.1%}</b><br>
                F1: <b>{r['f1']:.3f}</b>
            </div>
            <div style="font-size:0.8rem;color:var(--muted)">
                {r['params']:,} params • {r['train_time']:.2f}s • {r['memory_kb']:.1f} KB
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ====================== RADAR CHART ======================
    st.markdown("### 📡 Model Comparison Radar")
    metrics = ["Accuracy", "F1 Score", "Speed", "Memory Efficiency"]
    max_time = max(r["train_time"] for r in results)
    max_mem = max(r["memory_kb"] for r in results)
    max_params = max(r["params"] for r in results)
    
    fig_radar = go.Figure()
    for r, color in zip(results, ["#00d4aa", "#f97316", "#818cf8", "#ff6eb4", "#ffd700"]):
        values = [
            r["accuracy"],
            r["f1"],
            1 - (r["train_time"] / max_time),
            1 - (r["memory_kb"] / max_mem),
        ]
        values.append(values[0])  # close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=r["name"],
            line_color=color
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        height=460,
        title="Multi-Model Radar Comparison"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ====================== BAR CHARTS ======================
    st.markdown("### 📊 Metric Breakdown")
    fig_bar = go.Figure()
    for metric, key in [("Accuracy", "accuracy"), ("F1 Score", "f1")]:
        fig_bar.add_trace(go.Bar(
            name=metric,
            x=[r["name"] for r in results],
            y=[r[key] for r in results]
        ))
    fig_bar.update_layout(barmode="group", height=340, paper_bgcolor="#161b22",
                          title="Accuracy & F1 Score Comparison")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.balloons()

else:
    st.info("👈 Click **📊 Run Benchmark** to compare all 5 architectures side-by-side!")
    
    with st.expander("📚 How the Benchmark Works"):
        st.markdown("""
        Each model is trained from scratch on the same dataset and evaluated on:
        - **Accuracy & F1 Score**
        - **Number of Parameters**
        - **Training Time**
        - **Memory Usage**
        
        The **radar chart** shows the trade-offs visually. The best model is the one closest to the outer edge.
        """)
