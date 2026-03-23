"""NAS Explorer — Neural Architecture Search: compare architectures by params vs accuracy."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row, card
from utils.viz import plot_architecture_graph, _dark_layout

st.set_page_config(page_title="NAS Explorer", layout="wide", page_icon="🧬")
apply_theme()
render_sidebar("NAS Explorer")

hero(
    "Neural Architecture Search",
    "Automatically search for optimal neural network architectures. Compare accuracy vs parameter count, find the efficiency frontier.",
    pill="Lesson 13", pill_variant="purple",
)

TEAL = "#00d4aa"; ORANGE = "#f97316"; PURPLE = "#818cf8"; SURFACE = "#161b22"; BG = "#0d1117"


def count_params(layers):
    total = 0
    for i in range(len(layers)-1):
        total += layers[i] * layers[i+1] + layers[i+1]
    return total


def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def relu(x): return np.maximum(0, x)


def train_mlp(X_tr, y_tr, X_te, y_te, layers, lr=0.05, epochs=200):
    np.random.seed(42)
    n_classes = len(np.unique(y_tr))
    weights = []
    biases = []
    for i in range(len(layers)-1):
        weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i]))
        biases.append(np.zeros(layers[i+1]))

    for _ in range(epochs):
        # Forward
        a = X_tr
        acts = [a]
        for i, (W, b) in enumerate(zip(weights, biases)):
            z = a @ W + b
            a = relu(z) if i < len(weights)-1 else z
            acts.append(a)
        # Softmax
        ex = np.exp(a - a.max(1, keepdims=True))
        probs = ex / ex.sum(1, keepdims=True)
        yoh = np.eye(n_classes)[y_tr]
        # Backward
        dz = (probs - yoh) / len(X_tr)
        for i in range(len(weights)-1, -1, -1):
            dW = acts[i].T @ dz
            db = dz.sum(0)
            weights[i] -= lr * dW
            biases[i]  -= lr * db
            if i > 0:
                da = dz @ weights[i].T
                da[acts[i] <= 0] = 0
                dz = da
    # Test accuracy
    a = X_te
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        a = relu(z) if i < len(weights)-1 else z
    preds = np.argmax(a, axis=1)
    return np.mean(preds == y_te)


# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🔧 Search Space")
    dataset = st.selectbox("Dataset", ["moons", "2-class", "3-class", "5-class"])
    n_samples = st.slider("Samples", 200, 2000, 500, 100)
    n_trials = st.slider("Architectures to evaluate", 10, 80, 30, 5)
    max_layers = st.slider("Max hidden layers", 1, 5, 3)
    min_units = st.select_slider("Min units/layer", [4, 8, 16, 32], 8)
    max_units = st.select_slider("Max units/layer", [16, 32, 64, 128, 256], 64)
    epochs = st.slider("Epochs per trial", 50, 300, 150, 50)
    search_btn = st.button("🧬 Run NAS", type="primary")

with col2:
    st.markdown("### 📚 What is NAS?")
    card("""
    <b style="color:var(--accent)">Neural Architecture Search (NAS)</b> automates the design of neural network architectures.<br><br>
    Instead of hand-designing the number of layers and neurons, NAS explores a <b>search space</b> of possible architectures and 
    evaluates each one on the task.<br><br>
    <b style="color:var(--accent2)">Search strategies:</b><br>
    • <b>Random Search</b> (this demo) — simple but surprisingly effective<br>
    • <b>Evolutionary</b> — mutate best architectures<br>
    • <b>Differentiable (DARTS)</b> — gradient-based architecture search<br>
    • <b>Reinforcement Learning</b> — controller generates architectures<br><br>
    <b style="color:var(--accent3)">Efficiency Frontier:</b> architectures that achieve the best accuracy for a given parameter budget.
    """)

st.markdown("---")

if search_btn:
    # Generate dataset
    np.random.seed(42)
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset == "2-class":
        X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2,
                                   n_informative=5, random_state=42)
    elif dataset == "3-class":
        X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=3,
                                   n_informative=6, n_redundant=0, random_state=42)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=15, n_classes=5,
                                   n_informative=10, n_redundant=0, random_state=42)

    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    n_in = X.shape[1]
    n_out = len(np.unique(y))

    bar = st.progress(0)
    status = st.empty()

    results = []
    for trial in range(n_trials):
        n_hid = np.random.randint(1, max_layers+1)
        hidden = [np.random.choice([min_units, min_units*2, max_units//2, max_units])
                  for _ in range(n_hid)]
        layers = [n_in] + hidden + [n_out]
        params = count_params(layers)

        acc = train_mlp(X_tr, y_tr, X_te, y_te, layers, epochs=epochs)
        results.append({
            "architecture": hidden,
            "arch_str": str(hidden),
            "params": params,
            "accuracy": acc,
            "layers": n_hid,
        })

        bar.progress(int((trial+1)/n_trials * 100))
        status.markdown(f'<span class="status-badge status-running">Trial {trial+1}/{n_trials}: {hidden} → {acc:.1%}</span>',
                        unsafe_allow_html=True)

    status.markdown('<span class="status-badge status-done">✓ NAS complete!</span>', unsafe_allow_html=True)

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    best = results[0]

    metric_row([
        ("Best Accuracy", f"{best['accuracy']:.1%}"),
        ("Best Architecture", str(best['architecture'])),
        ("Best Params", f"{best['params']:,}"),
        ("Trials", n_trials),
    ])

    # ── Pareto / Efficiency frontier ─────────────────────────────────────────
    st.markdown("### 📊 Architecture Search Results")
    col_a, col_b = st.columns(2)

    params_list = [r["params"] for r in results]
    accs_list   = [r["accuracy"] for r in results]
    layers_list = [r["layers"] for r in results]
    arch_strs   = [r["arch_str"] for r in results]

    # Compute Pareto frontier
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, r2 in enumerate(results):
            if i != j and r2["accuracy"] >= r["accuracy"] and r2["params"] <= r["params"]:
                dominated = True; break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: x["params"])

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=params_list, y=accs_list, mode="markers",
            marker=dict(size=10, color=layers_list, colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Layers"),
                        line=dict(width=1, color="rgba(255,255,255,0.3)")),
            text=arch_strs, hovertemplate="Arch: %{text}<br>Params: %{x:,}<br>Acc: %{y:.1%}",
            name="All architectures"
        ))
        # Pareto frontier line
        px_par = [r["params"] for r in pareto]
        py_par = [r["accuracy"] for r in pareto]
        fig.add_trace(go.Scatter(
            x=px_par, y=py_par, mode="lines+markers",
            line=dict(color=ORANGE, width=3, dash="dot"),
            marker=dict(size=14, symbol="star", color=ORANGE),
            name="Efficiency Frontier"
        ))
        fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=BG,
                          font=dict(color="#8b949e"), height=380,
                          margin=dict(l=10,r=10,t=40,b=10),
                          title=dict(text="Accuracy vs Parameters (Pareto Frontier)",
                                     font=dict(color="#e6edf3")),
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", title="Parameters")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", title="Test Accuracy",
                         tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Top 10 table
        st.markdown("**Top 10 Architectures**")
        top10 = results[:10]
        for i, r in enumerate(top10):
            medal = ["🥇","🥈","🥉"] + [""] * 7
            is_pareto = r in pareto
            pareto_tag = ' <span style="color:var(--accent2)">★ Frontier</span>' if is_pareto else ""
            st.markdown(f"""
            <div class="nn-card" style="padding:0.7rem;margin-bottom:0.4rem">
              {medal[i]} <b>{r['arch_str']}</b>{pareto_tag}<br>
              <span style="color:var(--muted);font-size:0.8rem">
                Acc: <b style="color:var(--accent)">{r['accuracy']:.1%}</b> · 
                Params: <b>{r['params']:,}</b> · 
                Layers: {r['layers']}
              </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Best architecture visualization ──────────────────────────────────────
    st.markdown("### 🏆 Best Architecture")
    col_arch, col_info = st.columns([2, 1])
    with col_arch:
        best_layers = [n_in] + best["architecture"] + [n_out]
        best_names  = ["Input"] + [f"H{i+1}({s})" for i,s in enumerate(best["architecture"])] + ["Output"]
        fig_arch = plot_architecture_graph(
            [min(l, 8) for l in best_layers],  # cap display nodes
            best_names
        )
        st.plotly_chart(fig_arch, use_container_width=True)
    with col_info:
        card(f"""
        <b style="color:var(--accent)">🏆 Best Found:</b><br>
        Architecture: <b>{best['arch_str']}</b><br>
        Test Accuracy: <b>{best['accuracy']:.1%}</b><br>
        Parameters: <b>{best['params']:,}</b><br>
        Layers: <b>{best['layers']}</b><br><br>
        <b style="color:var(--accent2)">Efficiency Frontier has {len(pareto)} architectures</b> — 
        these give the best accuracy-per-parameter tradeoff.
        """)

else:
    st.info("👈 Configure search space and click **🧬 Run NAS** to start exploring architectures!")
