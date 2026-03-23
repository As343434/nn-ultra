"""Model Comparison — Side-by-side benchmark with radar + bar charts."""
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row, card

st.set_page_config(page_title="Model Comparison", layout="wide", page_icon="📊")
apply_theme()
render_sidebar("Model Comparison")

hero(
    "Model Comparison Dashboard",
    "Benchmark multiple neural architectures side-by-side — accuracy, F1, parameters, training time, memory — all in one radar chart.",
    pill="Lesson 14", pill_variant="orange",
)

TEAL = "#00d4aa"; ORANGE = "#f97316"; PURPLE = "#818cf8"; PINK = "#ff6eb4"; SURFACE = "#161b22"; BG = "#0d1117"
COLORS = [TEAL, ORANGE, PURPLE, PINK, "#ffd700"]


def relu(x): return np.maximum(0, x)


def train_and_evaluate(X_tr, y_tr, X_te, y_te, layers, lr=0.05, epochs=200, name="model"):
    n_classes = len(np.unique(y_tr))
    np.random.seed(42)
    weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i]) for i in range(len(layers)-1)]
    biases  = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]
    
    params = sum(w.size + b.size for w, b in zip(weights, biases))
    t0 = time.time()
    losses = []

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
        loss = -np.mean(np.sum(yoh * np.log(probs+1e-9), 1))
        losses.append(loss)
        dz = (probs - yoh) / len(X_tr)
        for i in range(len(weights)-1, -1, -1):
            dW = acts[i].T @ dz; db = dz.sum(0)
            weights[i] -= lr * dW; biases[i] -= lr * db
            if i > 0:
                da = dz @ weights[i].T; da[acts[i] <= 0] = 0; dz = da

    elapsed = time.time() - t0
    a = X_te
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        a = relu(z) if i < len(weights)-1 else z
    preds = np.argmax(a, axis=1)
    acc = np.mean(preds == y_te)
    f1 = f1_score(y_te, preds, average="macro", zero_division=0)
    mem = params * 4 / 1024  # KB (float32)

    return {
        "name": name,
        "accuracy": acc,
        "f1": f1,
        "params": params,
        "train_time": elapsed,
        "memory_kb": mem,
        "final_loss": losses[-1],
        "losses": losses,
    }


# ── Config ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Benchmark Config")
    dataset = st.selectbox("Dataset", ["moons", "2-class", "5-class"])
    n_samples = st.slider("Samples", 200, 2000, 600, 100)
    epochs = st.slider("Epochs", 50, 500, 200, 50)
    run_btn = st.button("📊 Run Benchmark", type="primary")

with col2:
    st.markdown("### 🏗️ Models to Compare")
    st.markdown("_(predefined architectures — edit in code for custom ones)_")
    models_cfg = [
        ("Tiny MLP", [8]),
        ("Shallow Wide", [64]),
        ("Deep Narrow", [16, 16, 16]),
        ("Balanced", [32, 32]),
        ("Deep Wide", [64, 64, 64]),
    ]
    for name, hidden in models_cfg:
        param_est = sum(a*b for a, b in zip([None]+hidden, hidden+[None])[1:-1])
        st.markdown(f"""
        <div style="display:inline-block;margin:3px;padding:4px 10px;border-radius:8px;
                    background:var(--surface2);border:1px solid var(--border);
                    font-size:0.82rem;color:var(--text)">
            <b>{name}</b> — hidden: {hidden}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

if run_btn:
    np.random.seed(42)
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset == "2-class":
        X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2,
                                   n_informative=5, random_state=42)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=15, n_classes=5,
                                   n_informative=10, n_redundant=0, random_state=42)

    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    n_in = X.shape[1]; n_out = len(np.unique(y))

    bar = st.progress(0)
    status = st.empty()
    results = []

    for i, (name, hidden) in enumerate(models_cfg):
        layers = [n_in] + hidden + [n_out]
        status.markdown(f'<span class="status-badge status-running">Training {name}...</span>', unsafe_allow_html=True)
        res = train_and_evaluate(X_tr, y_tr, X_te, y_te, layers, epochs=epochs, name=name)
        results.append(res)
        bar.progress(int((i+1)/len(models_cfg)*100))

    status.markdown('<span class="status-badge status-done">✓ Benchmark complete!</span>', unsafe_allow_html=True)

    # ── Metric table ──────────────────────────────────────────────────────────
    st.markdown("### 📋 Results Summary")
    best_acc  = max(r["accuracy"] for r in results)
    best_f1   = max(r["f1"] for r in results)
    best_time = min(r["train_time"] for r in results)
    best_mem  = min(r["memory_kb"] for r in results)

    cols = st.columns(len(results))
    for col, r, color in zip(cols, results, COLORS):
        is_best = r["accuracy"] == best_acc
        border  = f"border-top: 3px solid {color}" if not is_best else f"border-top: 3px solid {TEAL}; box-shadow: 0 0 16px {TEAL}44"
        col.markdown(f"""
        <div class="nn-card" style="{border}">
            <div style="color:{color};font-weight:700;font-size:0.9rem;margin-bottom:0.5rem">
                {'🏆 ' if is_best else ''}{r['name']}
            </div>
            <div style="font-size:0.8rem;color:var(--muted)">
                Accuracy: <b style="color:var(--text)">{r['accuracy']:.1%}</b><br>
                F1 Score: <b style="color:var(--text)">{r['f1']:.3f}</b><br>
                Params: <b style="color:var(--text)">{r['params']:,}</b><br>
                Time: <b style="color:var(--text)">{r['train_time']*1000:.0f}ms</b><br>
                Memory: <b style="color:var(--text)">{r['memory_kb']:.1f} KB</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        # Radar chart
        metrics_names = ["Accuracy", "F1 Score", "Speed (norm)", "Memory Eff.", "Simplicity"]
        max_time = max(r["train_time"] for r in results)
        max_mem  = max(r["memory_kb"] for r in results)
        max_params = max(r["params"] for r in results)

        fig = go.Figure()
        for r, color in zip(results, COLORS):
            values = [
                r["accuracy"],
                r["f1"],
                1 - r["train_time"] / max_time,
                1 - r["memory_kb"] / max_mem,
                1 - r["params"] / max_params,
            ]
            values.append(values[0])  # close radar
            cats = metrics_names + [metrics_names[0]]
            fig.add_trace(go.Scatterpolar(r=values, theta=cats, fill="toself",
                                          name=r["name"], line_color=color,
                                          fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.1)"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1],
                                                     gridcolor="rgba(255,255,255,0.1)",
                                                     tickcolor="rgba(255,255,255,0.3)"),
                                     angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
                          paper_bgcolor=SURFACE, font=dict(color="#8b949e"), height=420,
                          margin=dict(l=60,r=60,t=50,b=40),
                          title=dict(text="Model Comparison Radar", font=dict(color="#e6edf3")),
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Loss curves
        fig2 = go.Figure()
        for r, color in zip(results, COLORS):
            fig2.add_trace(go.Scatter(y=r["losses"], mode="lines", name=r["name"],
                                      line=dict(color=color, width=2)))
        fig2.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=BG,
                            font=dict(color="#8b949e"), height=420,
                            margin=dict(l=10,r=10,t=50,b=10),
                            title=dict(text="Training Loss Curves", font=dict(color="#e6edf3")),
                            legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)", title="Epoch")
        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)", title="Loss")
        st.plotly_chart(fig2, use_container_width=True)

    # Bar chart comparison
    st.markdown("### 📊 Metric Breakdown")
    bar_metrics = ["Accuracy", "F1 Score"]
    fig3 = go.Figure()
    for metric, attr in [("Accuracy", "accuracy"), ("F1 Score", "f1")]:
        fig3.add_trace(go.Bar(
            name=metric,
            x=[r["name"] for r in results],
            y=[r[attr] for r in results],
        ))
    fig3.update_layout(barmode="group", paper_bgcolor=SURFACE, plot_bgcolor=BG,
                        font=dict(color="#8b949e"), height=300,
                        margin=dict(l=10,r=10,t=40,b=10),
                        title=dict(text="Accuracy & F1 by Model", font=dict(color="#e6edf3")),
                        colorway=[TEAL, ORANGE],
                        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)", tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("👈 Click **📊 Run Benchmark** to compare all 5 architectures!")
