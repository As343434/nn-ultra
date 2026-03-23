"""NeuralForge Ultra — Home Dashboard"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.export import download_code, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row, card

st.set_page_config(page_title="NeuralForge Ultra", layout="wide", page_icon="🧠")
apply_theme()
render_sidebar("Home")

# ── Hero ─────────────────────────────────────────────────────────────────────
hero(
    "NeuralForge Ultra",
    "The most advanced interactive Neural Network Toolbox — 16 modules, 3 themes, AI-powered explanations, GAN Lab, RL Agent, Transformer Attention, and more.",
    pill="v3.0 ULTRA",
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

# ── Module grid ───────────────────────────────────────────────────────────────
st.markdown("### 🗂️ Module Overview")

modules = [
    ("⬡", "Perceptron",          "teal",   "Single-neuron classifier with animated decision boundary, live weight updates, and data augmentation."),
    ("⟶", "Forward Pass",        "purple", "10 activation functions, step-by-step neuron math, derivative viewer with Taylor expansion."),
    ("↺", "Backpropagation",     "teal",   "Chain-rule visualizer with live gradient flow, custom loss functions, and computation graph."),
    ("↗", "Gradient Descent",    "orange", "5 optimizers on 3D loss surfaces — saddle points, narrow valleys, noisy landscapes."),
    ("⬛", "ANN / MLP",           "teal",   "Configurable MLP with NumPy/PyTorch backends, custom CSV upload, weight histograms, and early stopping."),
    ("◫", "CNN",                  "purple", "Conv-net on MNIST/Fashion-MNIST — filter inspector, feature maps, class activation maps."),
    ("⇌", "RNN / LSTM / GRU",    "teal",   "Sequence modeling lab with sine/stock/text prediction, attention overlay, and hidden-state heatmap."),
    ("◎", "Autoencoder / VAE",   "orange", "AE + VAE with 2D latent space, denoising mode, interpolation, and reconstruction gallery."),
    ("◉", "OpenCV Vision",       "teal",   "15 preprocessing ops, pixel histogram, Fourier transform, and direct CNN pipeline feed."),
    ("⚡", "Transformer Attn",   "purple", "Multi-head attention visualizer, positional encoding explorer, and QKV decomposition."),
    ("🎮", "GAN Lab",             "orange", "Train a DCGAN in-browser — watch fake images evolve, mode collapse detector, loss dynamics."),
    ("🤖", "RL Agent",            "teal",   "DQN agent on CartPole/GridWorld — reward curves, Q-value heatmap, policy visualization."),
    ("🧬", "NAS Explorer",        "purple", "Neural Architecture Search — compare architectures by params vs accuracy, efficiency frontier."),
    ("📊", "Model Comparison",    "orange", "Side-by-side benchmark: accuracy, params, inference time, memory — radar + bar charts."),
    ("🧠", "AI Explainer",        "teal",   "Claude-powered explainer — ask ANY question about neural networks and get instant answers."),
    ("📤", "Export Hub",          "purple", "Export any model as PyTorch, ONNX, pickle, or Python script with one click."),
]

cols = st.columns(3)
for i, (icon, name, color, desc) in enumerate(modules):
    accent = {"teal": "#00d4aa", "orange": "#f97316", "purple": "#818cf8"}[color]
    with cols[i % 3]:
        st.markdown(f"""
        <div class="nn-card" style="min-height:130px">
          <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
            <span style="font-size:1.3rem">{icon}</span>
            <span style="font-size:0.95rem;font-weight:700;color:{accent};
                         font-family:'IBM Plex Sans',sans-serif">{name}</span>
          </div>
          <div style="font-size:0.82rem;color:var(--muted);font-family:'IBM Plex Sans',sans-serif;
                      line-height:1.55">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Live Quick Demo ───────────────────────────────────────────────────────────
st.markdown("### ⚡ Live Quick Demo — Train a 3-Layer MLP")

col1, col2 = st.columns([1, 2])
with col1:
    lr_demo  = st.slider("Learning rate", 0.001, 0.5, 0.05, 0.001, key="home_lr")
    ep_demo  = st.slider("Epochs", 10, 300, 100, 10, key="home_ep")
    h_demo   = st.slider("Hidden size", 4, 64, 16, 4, key="home_h")
    dataset  = st.selectbox("Dataset", ["moons", "circles", "blobs"], key="home_ds")
    run_demo = st.button("▶ Train", type="primary")

with col2:
    if run_demo:
        from sklearn.datasets import make_moons, make_circles, make_blobs
        from sklearn.preprocessing import StandardScaler

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
            a1 = np.maximum(0, X @ W1 + b1)
            a2 = np.maximum(0, a1 @ W2 + b2)
            z3 = a2 @ W3 + b3
            ex = np.exp(z3 - z3.max(1, keepdims=True))
            probs = ex / ex.sum(1, keepdims=True)
            yoh = np.eye(n_classes)[y]
            loss = -np.mean(np.sum(yoh * np.log(probs + 1e-9), 1))
            losses.append(loss)
            preds = np.argmax(probs, axis=1)
            accs.append(np.mean(preds == y))

            dz3 = (probs - yoh) / len(X)
            dW3 = a2.T @ dz3; db3 = dz3.sum(0)
            da2 = dz3 @ W3.T; da2[a1 @ W2 + b2 <= 0] = 0
            dW2 = a1.T @ da2; db2 = da2.sum(0)
            da1 = da2 @ W2.T; da1[X @ W1 + b1 <= 0] = 0
            dW1 = X.T @ da1; db1 = da1.sum(0)
            W1 -= lr_demo * dW1; b1 -= lr_demo * db1
            W2 -= lr_demo * dW2; b2 -= lr_demo * db2
            W3 -= lr_demo * dW3; b3 -= lr_demo * db3

            bar.progress(int((ep + 1) / ep_demo * 100))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, mode="lines", name="Loss",
                                 line=dict(color="#00d4aa", width=2.5)))
        fig.add_trace(go.Scatter(y=accs, mode="lines", name="Accuracy",
                                 line=dict(color="#f97316", width=2.5),
                                 yaxis="y2"))
        fig.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#8b949e"),
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            yaxis2=dict(overlaying="y", side="right",
                        gridcolor="rgba(255,255,255,0.03)",
                        tickformat=".0%"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✓ Final accuracy: **{accs[-1]:.1%}** | Final loss: **{losses[-1]:.4f}**")

        download_pickle("⬇ Download model", {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}, "demo_mlp.pkl")
    else:
        st.info("👈 Configure your model and click **▶ Train**")

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:var(--muted);font-size:0.8rem;padding:1rem">
  🧠 NeuralForge Ultra v3.0 · Built with Streamlit, PyTorch, NumPy, Plotly · 
  <span style="color:var(--accent)">Use the sidebar to navigate all 16 modules</span>
</div>
""", unsafe_allow_html=True)
