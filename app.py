import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="NeuralForge Ultra",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# ====================== FULL FIXED CSS ======================
st.markdown("""
<style>
    /* Core Theme */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }

    h1, h2, h3, h4 {
        font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text) !important;
    }

    /* Custom Cards */
    .nn-card {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1.4rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
    }
    .nn-card:hover {
        border-color: var(--accent) !important;
        box-shadow: 0 0 20px rgba(0, 212, 170, 0.3) !important;
    }

    .nn-hero {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 2.5rem 2rem !important;
        margin-bottom: 1.6rem !important;
        position: relative;
        overflow: hidden;
    }

    .nn-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background: rgba(0, 212, 170, 0.15);
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .nn-pill-orange { background: rgba(249, 115, 22, 0.15); color: #f97316; }
    .nn-pill-purple { background: rgba(129, 140, 248, 0.15); color: #818cf8; }

    .nn-metric {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 1rem 1.2rem !important;
        text-align: center;
        transition: box-shadow 0.2s !important;
    }
    .nn-metric:hover { box-shadow: 0 0 15px rgba(0, 212, 170, 0.4) !important; }

    .nn-metric-value {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: var(--accent) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .nn-metric-label {
        font-size: 0.75rem !important;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem !important;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .status-running { background: rgba(249, 115, 22, 0.2); color: #f97316; }
    .status-done   { background: rgba(34, 197, 151, 0.2); color: #22c597; }
    .status-idle   { background: rgba(148, 163, 184, 0.2); color: #94a3b8; }

    /* Streamlit Component Overrides */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
    }

    /* Expander Fix */
    div[data-testid="stExpander"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: var(--accent) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 0 12px rgba(0, 212, 170, 0.4) !important;
    }
    button[kind="secondary"] {
        border: 1px solid var(--accent) !important;
        color: var(--accent) !important;
        border-radius: 8px !important;
    }

    [data-testid="stProgress"] > div > div {
        background: var(--accent) !important;
    }

    hr {
        border-color: var(--border) !important;
        margin: 1.8rem 0 !important;
    }

    /* Cyberpunk Scanlines - Safe version */
    .cyber-scanlines::after {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent 0px,
            transparent 2px,
            rgba(0,0,0,0.05) 2px,
            rgba(0,0,0,0.05) 4px
        );
        pointer-events: none;
        z-index: 9999;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

# ====================== HERO & METRIC HELPERS ======================
def hero(title, subtitle, pill="v3.0 ULTRA"):
    st.markdown(f"""
    <div class="nn-hero">
        <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.8rem;">
            <span style="font-size:2.8rem;">🧠</span>
            <h1 style="margin:0; font-size:2.4rem;">{title}</h1>
        </div>
        <span class="nn-pill">{pill}</span>
        <p style="font-size:1.1rem; color:var(--muted); max-width:800px; line-height:1.6;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="nn-metric">
                <div class="nn-metric-value">{value}</div>
                <div class="nn-metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🧠 NeuralForge")
    st.markdown("### Navigation")
    st.page_link("app.py", label="🏠 Home", icon="🏠")
    # Add other page links here when you create them
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== MAIN CONTENT ======================
st.title("NeuralForge Ultra")

hero(
    "NeuralForge Ultra",
    "The most advanced interactive Neural Network Toolbox — 16 modules, 3 themes, AI-powered explanations, GAN Lab, RL Agent, Transformer Attention, and more.",
    pill="v3.0 ULTRA"
)

metric_row([
    ("Modules", "16"),
    ("Architectures", "10"),
    ("Optimizers", "5"),
    ("Datasets", "10+"),
    ("Themes", "3"),
    ("AI Features", "∞"),
])

st.markdown("---")

# ====================== MODULE GRID ======================
st.markdown("### 🗂️ Module Overview")

modules = [
    ("⬡", "Perceptron", "teal", "Single-neuron classifier with animated decision boundary, live weight updates, and data augmentation."),
    ("⟶", "Forward Pass", "purple", "10 activation functions, step-by-step neuron math, derivative viewer with Taylor expansion."),
    ("↺", "Backpropagation", "teal", "Chain-rule visualizer with live gradient flow, custom loss functions, and computation graph."),
    ("↗", "Gradient Descent", "orange", "5 optimizers on 3D loss surfaces — saddle points, narrow valleys, noisy landscapes."),
    ("⬛", "ANN / MLP", "teal", "Configurable MLP with NumPy/PyTorch backends, custom CSV upload, weight histograms, and early stopping."),
    ("◫", "CNN", "purple", "Conv-net on MNIST/Fashion-MNIST — filter inspector, feature maps, class activation maps."),
    ("⇌", "RNN / LSTM / GRU", "teal", "Sequence modeling lab with sine/stock/text prediction, attention overlay, and hidden-state heatmap."),
    ("◎", "Autoencoder / VAE", "orange", "AE + VAE with 2D latent space, denoising mode, interpolation, and reconstruction gallery."),
    ("◉", "OpenCV Vision", "teal", "15 preprocessing ops, pixel histogram, Fourier transform, and direct CNN pipeline feed."),
    ("⚡", "Transformer Attn", "purple", "Multi-head attention visualizer, positional encoding explorer, and QKV decomposition."),
    ("🎮", "GAN Lab", "orange", "Train a DCGAN in-browser — watch fake images evolve, mode collapse detector, loss dynamics."),
    ("🤖", "RL Agent", "teal", "DQN agent on CartPole/GridWorld — reward curves, Q-value heatmap, policy visualization."),
    ("🧬", "NAS Explorer", "purple", "Neural Architecture Search — compare architectures by params vs accuracy, efficiency frontier."),
    ("📊", "Model Comparison", "orange", "Side-by-side benchmark: accuracy, params, inference time, memory — radar + bar charts."),
    ("🧠", "AI Explainer", "teal", "Claude-powered explainer — ask ANY question about neural networks and get instant answers."),
    ("📤", "Export Hub", "purple", "Export any model as PyTorch, ONNX, pickle, or Python script with one click."),
]

cols = st.columns(3)
accent_map = {"teal": "#00d4aa", "orange": "#f97316", "purple": "#818cf8"}

for i, (icon, name, color, desc) in enumerate(modules):
    accent = accent_map[color]
    with cols[i % 3]:
        st.markdown(f"""
        <div class="nn-card" style="min-height: 138px;">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.6rem;">
                <span style="font-size:1.45rem;">{icon}</span>
                <span style="font-size:1rem; font-weight:700; color:{accent};">{name}</span>
            </div>
            <div style="font-size:0.83rem; color:var(--muted); line-height:1.55;">
                {desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ====================== LIVE DEMO ======================
st.markdown("### ⚡ Live Quick Demo — Train a 3-Layer MLP")

col1, col2 = st.columns([1, 2])

with col1:
    lr_demo = st.slider("Learning rate", 0.001, 0.5, 0.05, 0.001, key="home_lr")
    ep_demo = st.slider("Epochs", 10, 300, 100, 10, key="home_ep")
    h_demo = st.slider("Hidden size", 4, 64, 16, 4, key="home_h")
    dataset = st.selectbox("Dataset", ["moons", "circles", "blobs"], key="home_ds")
    run_demo = st.button("▶ Train", type="primary")

with col2:
    if run_demo:
        np.random.seed(42)
        if dataset == "moons":
            X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
        elif dataset == "circles":
            X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
        else:
            X, y = make_blobs(n_samples=200, centers=3, random_state=42)
            y = (y > 0).astype(int)

        X = StandardScaler().fit_transform(X)
        n_classes = len(np.unique(y))

        # Simple 3-layer MLP
        W1 = np.random.randn(2, h_demo) * 0.3
        b1 = np.zeros(h_demo)
        W2 = np.random.randn(h_demo, h_demo) * 0.3
        b2 = np.zeros(h_demo)
        W3 = np.random.randn(h_demo, n_classes) * 0.3
        b3 = np.zeros(n_classes)

        losses, accs = [], []
        bar = st.progress(0)
        chart_ph = st.empty()

        for ep in range(ep_demo):
            # Forward
            a1 = np.maximum(0, X @ W1 + b1)
            a2 = np.maximum(0, a1 @ W2 + b2)
            z3 = a2 @ W3 + b3
            ex = np.exp(z3 - z3.max(1, keepdims=True))
            probs = ex / ex.sum(1, keepdims=True)

            # Loss & accuracy
            yoh = np.eye(n_classes)[y]
            loss = -np.mean(np.sum(yoh * np.log(probs + 1e-9), 1))
            losses.append(loss)
            preds = np.argmax(probs, axis=1)
            accs.append(np.mean(preds == y))

            # Backward
            dz3 = (probs - yoh) / len(X)
            dW3 = a2.T @ dz3
            db3 = dz3.sum(0)
            da2 = dz3 @ W3.T
            da2[a1 @ W2 + b2 <= 0] = 0
            dW2 = a1.T @ da2
            db2 = da2.sum(0)
            da1 = da2 @ W2.T
            da1[X @ W1 + b1 <= 0] = 0
            dW1 = X.T @ da1
            db1 = da1.sum(0)

            # Update
            W1 -= lr_demo * dW1
            b1 -= lr_demo * db1
            W2 -= lr_demo * dW2
            b2 -= lr_demo * db2
            W3 -= lr_demo * dW3
            b3 -= lr_demo * db3

            bar.progress((ep + 1) / ep_demo)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, mode="lines", name="Loss", line=dict(color="#00d4aa", width=2.5)))
        fig.add_trace(go.Scatter(y=accs, mode="lines", name="Accuracy", line=dict(color="#f97316", width=2.5), yaxis="y2"))

        fig.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e"), height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis2=dict(overlaying="y", side="right", tickformat=".0%"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✓ Final accuracy: **{accs[-1]:.1%}** | Final loss: **{losses[-1]:.4f}**")

    else:
        st.info("👈 Configure parameters on the left and click **▶ Train** to run the live demo.")

st.markdown("---")

st.markdown("""
<div style="text-align:center; color:var(--muted); font-size:0.85rem; padding:2rem 0;">
    🧠 NeuralForge Ultra v3.0 • Built with ❤️ using Streamlit, NumPy, Plotly & scikit-learn<br>
    <span style="color:var(--accent)">Use the sidebar to explore all 16 interactive modules</span>
</div>
""", unsafe_allow_html=True)
