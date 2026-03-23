import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="NAS Explorer",
    layout="wide",
    page_icon="🧬"
)

# ====================== SAFE CSS ======================
st.markdown("""
<style>
    section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
    h1, h2, h3, h4 { font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
    .nn-card { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 1.4rem !important; margin-bottom: 1rem !important; }
    .nn-hero { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 16px !important; padding: 2.2rem 2rem !important; margin-bottom: 1.6rem !important; }
    .nn-pill { display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px; background: rgba(129, 140, 248, 0.15) !important; color: #818cf8 !important; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .status-running { background: rgba(249,115,22,0.2); color: #f97316; padding: 0.4rem 0.8rem; border-radius: 999px; font-weight: 600; }
    .status-done   { background: rgba(34,197,151,0.2); color: #22c597; padding: 0.4rem 0.8rem; border-radius: 999px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🧬 NAS Explorer")
    st.markdown("Neural Architecture Search")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 13</div>
    <h1>Neural Architecture Search</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Explore different neural network architectures automatically.<br>
        Find the best accuracy vs parameter count tradeoff (Efficiency Frontier).
    </p>
</div>
""", unsafe_allow_html=True)

# ====================== HELPERS ======================
def count_params(layers):
    total = 0
    for i in range(len(layers)-1):
        total += layers[i] * layers[i+1] + layers[i+1]   # weights + biases
    return total

def train_simple_mlp(X_tr, y_tr, X_te, y_te, layers, lr=0.05, epochs=150):
    np.random.seed(42)
    n_classes = len(np.unique(y_tr))
    # Initialize weights
    weights = []
    biases = []
    for i in range(len(layers)-1):
        w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
        b = np.zeros(layers[i+1])
        weights.append(w)
        biases.append(b)

    for _ in range(epochs):
        # Forward pass
        a = X_tr
        activations = [a]
        for i in range(len(weights)):
            z = a @ weights[i] + biases[i]
            a = relu(z) if i < len(weights)-1 else z
            activations.append(a)

        # Softmax
        ex = np.exp(a - a.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)

        # Backward pass
        y_onehot = np.eye(n_classes)[y_tr]
        dz = (probs - y_onehot) / len(X_tr)

        for i in range(len(weights)-1, -1, -1):
            dW = activations[i].T @ dz
            db = dz.sum(axis=0)
            weights[i] -= lr * dW
            biases[i] -= lr * db
            if i > 0:
                da = dz @ weights[i].T
                da[activations[i] <= 0] = 0
                dz = da

    # Evaluate on test set
    a = X_te
    for i in range(len(weights)):
        z = a @ weights[i] + biases[i]
        a = relu(z) if i < len(weights)-1 else z
    preds = np.argmax(a, axis=1)
    return np.mean(preds == y_te)

def relu(x):
    return np.maximum(0, x)

# ====================== CONTROLS ======================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🔧 Search Space")
    dataset = st.selectbox("Dataset", ["moons", "2-class", "3-class", "5-class"])
    n_samples = st.slider("Number of Samples", 200, 2000, 600, 100)
    n_trials = st.slider("Number of Architectures to Test", 10, 100, 40, 5)
    max_layers = st.slider("Max Hidden Layers", 1, 5, 3)
    min_units = st.select_slider("Min Units per Layer", [4, 8, 16, 32], 8)
    max_units = st.select_slider("Max Units per Layer", [16, 32, 64, 128, 256], 64)
    epochs_per_trial = st.slider("Epochs per Architecture", 50, 300, 120, 30)
    search_btn = st.button("🧬 Run Neural Architecture Search", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📚 What is NAS?")
    st.markdown("""
    <div class="nn-card">
        <b>Neural Architecture Search (NAS)</b> automates the design of neural networks.<br><br>
        Instead of manually choosing number of layers and neurons, we let the computer explore many architectures 
        and keep the best ones.<br><br>
        This demo uses <b>Random Search</b> — surprisingly effective and simple.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ====================== NAS EXECUTION ======================
if search_btn:
    # Generate dataset
    np.random.seed(42)
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    else:
        n_classes = int(dataset[0])
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=10 if n_classes <= 3 else 15,
            n_classes=n_classes,
            n_informative=6 if n_classes <= 3 else 10,
            random_state=42
        )

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    n_input = X.shape[1]
    n_output = len(np.unique(y))

    bar = st.progress(0)
    status = st.empty()
    results = []

    for trial in range(n_trials):
        # Random architecture
        n_hidden_layers = np.random.randint(1, max_layers + 1)
        hidden_sizes = [np.random.randint(min_units, max_units + 1) for _ in range(n_hidden_layers)]
        architecture = [n_input] + hidden_sizes + [n_output]

        params = count_params(architecture)
        accuracy = train_simple_mlp(X_train, y_train, X_test, y_test, architecture, 
                                   lr=0.08, epochs=epochs_per_trial)

        results.append({
            "architecture": hidden_sizes,
            "arch_str": str(hidden_sizes),
            "params": params,
            "accuracy": accuracy,
            "layers": n_hidden_layers
        })

        bar.progress((trial + 1) / n_trials)
        status.markdown(
            f'<span class="status-running">Trial {trial+1}/{n_trials} — {hidden_sizes} → {accuracy:.1%}</span>',
            unsafe_allow_html=True
        )

    status.markdown('<span class="status-done">✓ NAS Search Complete!</span>', unsafe_allow_html=True)

    # Sort by accuracy
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    best = results[0]

    # Metrics
    st.markdown("### 📊 Results")
    metric_row([
        ("Best Accuracy", f"{best['accuracy']:.1%}"),
        ("Best Architecture", str(best['architecture'])),
        ("Parameters", f"{best['params']:,}"),
        ("Trials Run", n_trials)
    ])

    # Pareto Frontier
    st.markdown("### 📈 Accuracy vs Parameters (Efficiency Frontier)")

    params_list = [r["params"] for r in results]
    acc_list = [r["accuracy"] for r in results]
    layer_list = [r["layers"] for r in results]

    # Simple Pareto front
    pareto = []
    for r in results:
        dominated = any(
            r2["accuracy"] >= r["accuracy"] and r2["params"] <= r["params"] and r2 != r
            for r2 in results
        )
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: x["params"])

    fig = go.Figure()

    # All architectures
    fig.add_trace(go.Scatter(
        x=params_list, y=acc_list,
        mode="markers",
        marker=dict(size=9, color=layer_list, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Hidden Layers")),
        text=[str(r["architecture"]) for r in results],
        hovertemplate="Arch: %{text}<br>Params: %{x:,}<br>Acc: %{y:.1%}<extra></extra>",
        name="All Architectures"
    ))

    # Pareto frontier
    fig.add_trace(go.Scatter(
        x=[r["params"] for r in pareto],
        y=[r["accuracy"] for r in pareto],
        mode="lines+markers",
        line=dict(color="#f97316", width=4, dash="dot"),
        marker=dict(size=12, symbol="star", color="#f97316"),
        name="Efficiency Frontier"
    ))

    fig.update_layout(
        title="Accuracy vs Number of Parameters",
        xaxis_title="Number of Parameters",
        yaxis_title="Test Accuracy",
        height=460,
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)

    # Top Architectures
    st.markdown("### 🏆 Top 10 Architectures")
    for i, r in enumerate(results[:10]):
        medal = ["🥇", "🥈", "🥉"] + [""]*7
        is_pareto = r in pareto
        tag = ' <span style="color:#f97316">★ Frontier</span>' if is_pareto else ""
        st.markdown(f"""
        <div class="nn-card" style="padding:0.8rem 1rem; margin-bottom:0.5rem">
            {medal[i]} <b>{r['arch_str']}</b>{tag}<br>
            <span style="color:var(--muted); font-size:0.85rem">
                Accuracy: <b style="color:#00d4aa">{r['accuracy']:.1%}</b> • 
                Params: <b>{r['params']:,}</b> • 
                Layers: {r['layers']}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.balloons()

else:
    st.info("👈 Configure the search space on the left and click **🧬 Run Neural Architecture Search**")

    with st.expander("📚 About Neural Architecture Search"):
        st.markdown("""
        NAS automates the tedious process of designing neural networks.
        
        This demo uses **Random Search** — one of the simplest yet surprisingly effective methods.
        
        The **Efficiency Frontier** shows architectures that give the best accuracy for a given parameter budget.
        """)
