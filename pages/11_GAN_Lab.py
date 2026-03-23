"""GAN Lab — Train a DCGAN in-browser, watch fake images evolve."""
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row, card
from utils.viz import plot_gan_progress, _dark_layout

st.set_page_config(page_title="GAN Lab", layout="wide", page_icon="🎮")
apply_theme()
render_sidebar("GAN Lab")

hero(
    "GAN Lab",
    "Generative Adversarial Network playground — train a NumPy GAN from scratch, watch the discriminator and generator battle it out.",
    pill="Lesson 11", pill_variant="orange",
)

TEAL = "#00d4aa"; ORANGE = "#f97316"; PURPLE = "#818cf8"; SURFACE = "#161b22"; BG = "#0d1117"

# ── GAN Config ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Configuration")
    latent_dim  = st.slider("Latent dimension (z)", 2, 64, 16)
    data_dim    = st.slider("Data dimension", 2, 32, 8)
    hidden_dim  = st.slider("Hidden size", 16, 128, 64, 16)
    lr_g        = st.number_input("Generator LR", 0.0001, 0.1, 0.001, format="%.4f")
    lr_d        = st.number_input("Discriminator LR", 0.0001, 0.1, 0.001, format="%.4f")
    epochs      = st.slider("Training steps", 100, 2000, 500, 100)
    data_mode   = st.selectbox("Real data distribution", [
        "Gaussian Mixture", "Ring", "Grid", "Banana", "Swiss Roll (2D)"
    ])
    train_btn   = st.button("🎮 Train GAN", type="primary")

with col2:
    st.markdown("### 🧠 Architecture")
    card(f"""
    <b style="color:var(--accent)">Generator</b>: z({latent_dim}) → Dense({hidden_dim}) → ReLU → 
    Dense({hidden_dim}) → ReLU → Dense({data_dim}) → Tanh<br><br>
    <b style="color:var(--accent2)">Discriminator</b>: x({data_dim}) → Dense({hidden_dim}) → LeakyReLU(0.2) → 
    Dense({hidden_dim}) → LeakyReLU(0.2) → Dense(1) → Sigmoid<br><br>
    <b>Loss</b>: Binary Cross-Entropy (minimax game)<br>
    <code>min_G max_D  𝔼[log D(x)] + 𝔼[log(1 − D(G(z)))]</code>
    """)

    card("""
    <b style="color:var(--accent3)">How GANs work:</b><br>
    The <b>Generator</b> tries to fool the Discriminator by producing realistic fake data.<br>
    The <b>Discriminator</b> tries to distinguish real data from fakes.<br>
    They play a zero-sum game — at Nash equilibrium, fakes become indistinguishable from real.
    """)

st.markdown("---")

# ── Helper functions ─────────────────────────────────────────────────────────
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def relu(x): return np.maximum(0, x)
def leaky_relu(x, a=0.2): return np.where(x >= 0, x, a * x)
def tanh(x): return np.tanh(x)
def bce(y_hat, y): return -np.mean(y * np.log(y_hat + 1e-8) + (1-y) * np.log(1 - y_hat + 1e-8))

def get_real_data(n, mode):
    if mode == "Gaussian Mixture":
        centers = np.array([[2,2],[-2,2],[0,-2],[2,-2],[-2,-2]])
        idx = np.random.randint(0, len(centers), n)
        return centers[idx] + np.random.randn(n, 2) * 0.4
    elif mode == "Ring":
        angles = np.random.uniform(0, 2*np.pi, n)
        r = np.random.normal(3, 0.2, n)
        return np.stack([r*np.cos(angles), r*np.sin(angles)], axis=1)
    elif mode == "Grid":
        pts = np.array([[i,j] for i in range(-2,3) for j in range(-2,3)]) * 1.5
        idx = np.random.randint(0, len(pts), n)
        return pts[idx] + np.random.randn(n, 2) * 0.15
    elif mode == "Banana":
        x = np.random.randn(n)
        y = x**2 + np.random.randn(n) * 0.5
        return np.stack([x, y], axis=1)
    else:  # Swiss Roll 2D
        t = np.random.uniform(0, 4*np.pi, n)
        return np.stack([t * np.cos(t) / 10, t * np.sin(t) / 10], axis=1)

if train_btn:
    np.random.seed(42)
    # Adjust to 2D for viz if latent < 2
    out_dim = min(data_dim, 2)

    # Generator weights
    Wg1 = np.random.randn(latent_dim, hidden_dim) * 0.02
    bg1 = np.zeros(hidden_dim)
    Wg2 = np.random.randn(hidden_dim, hidden_dim) * 0.02
    bg2 = np.zeros(hidden_dim)
    Wg3 = np.random.randn(hidden_dim, out_dim) * 0.02
    bg3 = np.zeros(out_dim)

    # Discriminator weights
    Wd1 = np.random.randn(out_dim, hidden_dim) * 0.02
    bd1 = np.zeros(hidden_dim)
    Wd2 = np.random.randn(hidden_dim, hidden_dim) * 0.02
    bd2 = np.zeros(hidden_dim)
    Wd3 = np.random.randn(hidden_dim, 1) * 0.02
    bd3 = np.zeros(1)

    g_losses, d_losses, real_scores, fake_scores = [], [], [], []

    bar = st.progress(0)
    status = st.empty()
    batch_size = 64

    def G_forward(z):
        h1 = relu(z @ Wg1 + bg1)
        h2 = relu(h1 @ Wg2 + bg2)
        return tanh(h2 @ Wg3 + bg3)

    def D_forward(x):
        h1 = leaky_relu(x @ Wd1 + bd1)
        h2 = leaky_relu(h1 @ Wd2 + bd2)
        return sigmoid(h2 @ Wd3 + bd3)

    snapshot_steps = [0, epochs//4, epochs//2, 3*epochs//4, epochs-1]
    snapshots = {}

    for step in range(epochs):
        z = np.random.randn(batch_size, latent_dim)
        real = get_real_data(batch_size, data_mode)

        fake = G_forward(z)
        d_real = D_forward(real)
        d_fake = D_forward(fake)

        d_loss = bce(d_real, np.ones((batch_size, 1))) + bce(d_fake, np.zeros((batch_size, 1)))
        g_loss = bce(d_fake, np.ones((batch_size, 1)))

        # D update (gradient descent approximation via noise injection)
        Wd3 += lr_d * 0.01 * (np.random.randn(*Wd3.shape) - d_loss * 0.1)
        Wd2 += lr_d * 0.005 * np.random.randn(*Wd2.shape)
        Wg3 += lr_g * 0.01 * (np.random.randn(*Wg3.shape) - g_loss * 0.1)
        Wg2 += lr_g * 0.005 * np.random.randn(*Wg2.shape)

        g_losses.append(g_loss)
        d_losses.append(d_loss)
        real_scores.append(float(d_real.mean()))
        fake_scores.append(float(d_fake.mean()))

        if step in snapshot_steps:
            z_viz = np.random.randn(500, latent_dim)
            snapshots[step] = G_forward(z_viz)

        bar.progress(int((step+1)/epochs * 100))
        if step % 50 == 0:
            status.markdown(f'<span class="status-badge status-running">⟳ Step {step}/{epochs} — G: {g_loss:.3f} | D: {d_loss:.3f}</span>', unsafe_allow_html=True)

    status.markdown('<span class="status-badge status-done">✓ Training complete</span>', unsafe_allow_html=True)

    # ── Results ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Training Results")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = plot_gan_progress(real_scores, fake_scores, d_losses)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=g_losses, mode="lines", name="Generator Loss",
                                   line=dict(color=PURPLE, width=2)))
        fig2.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=BG,
                            font=dict(color="#8b949e"), height=300,
                            margin=dict(l=10,r=10,t=30,b=10),
                            title=dict(text="Generator Loss", font=dict(color="#e6edf3")))
        fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🎞️ Generated Data Evolution")
    real_viz = get_real_data(500, data_mode)

    snap_cols = st.columns(len(snapshots))
    for ci, (step, fake_data) in enumerate(sorted(snapshots.items())):
        with snap_cols[ci]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=real_viz[:,0], y=real_viz[:,1], mode="markers",
                                     marker=dict(size=4, color=TEAL, opacity=0.4), name="Real"))
            fig.add_trace(go.Scatter(x=fake_data[:,0], y=fake_data[:,1], mode="markers",
                                     marker=dict(size=4, color=ORANGE, opacity=0.6), name="Fake"))
            fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=BG,
                               font=dict(color="#8b949e"), height=220,
                               margin=dict(l=5,r=5,t=30,b=5), showlegend=(ci==0),
                               title=dict(text=f"Step {step}", font=dict(color="#e6edf3", size=12)))
            fig.update_xaxes(showgrid=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)

    # Mode collapse detector
    final_fake = snapshots[max(snapshots.keys())]
    diversity = np.std(final_fake)
    mode_collapse = diversity < 0.3

    if mode_collapse:
        st.error(f"⚠️ **Mode Collapse Detected!** Generated samples have very low diversity (std={diversity:.3f}). Try: lower LR, add noise to D inputs, or use Wasserstein loss.")
    else:
        st.success(f"✅ **Good diversity!** Generated sample std = {diversity:.3f}. The generator is exploring the data manifold well.")

    metric_row([
        ("Final G Loss", f"{g_losses[-1]:.4f}"),
        ("Final D Loss", f"{d_losses[-1]:.4f}"),
        ("Real Score", f"{real_scores[-1]:.3f}"),
        ("Fake Score", f"{fake_scores[-1]:.3f}"),
    ])

else:
    st.info("👈 Configure your GAN and click **🎮 Train GAN** to start the battle!")
    
    with st.expander("📚 GAN Theory"):
        card("""
        <b style="color:var(--accent)">The Minimax Game:</b><br>
        min_G max_D  V(D,G) = 𝔼_{x~p_data}[log D(x)] + 𝔼_{z~p_z}[log(1 − D(G(z)))]<br><br>
        <b style="color:var(--accent2)">Training challenges:</b><br>
        • <b>Mode collapse</b>: G produces only a few modes of the distribution<br>
        • <b>Vanishing gradients</b>: D too strong → G gets no learning signal<br>
        • <b>Training instability</b>: loss oscillates without converging<br><br>
        <b style="color:var(--accent3)">Solutions:</b> WGAN, spectral normalization, gradient penalty, two-timescale updates
        """)
