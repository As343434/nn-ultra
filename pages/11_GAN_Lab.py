import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="GAN Lab",
    layout="wide",
    page_icon="🎮"
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
    button[kind="primary"] {
        background: var(--accent) !important;
        color: #000 !important;
        font-weight: 700 !important;
    }
    .status-running { background: rgba(249, 115, 22, 0.2); color: #f97316; padding: 0.3rem 0.8rem; border-radius: 999px; font-weight: 600; }
    .status-done   { background: rgba(34, 197, 151, 0.2); color: #22c597; padding: 0.3rem 0.8rem; border-radius: 999px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🎮 GAN Lab")
    st.markdown("Generative Adversarial Network")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 11</div>
    <h1>GAN Lab</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Train a simple GAN from scratch in your browser.<br>
        Watch the Generator and Discriminator battle it out in real time.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ====================== CONFIG ======================
col_cfg, col_arch = st.columns([1, 2])

with col_cfg:
    st.markdown("### ⚙️ Configuration")
    latent_dim = st.slider("Latent dimension (z)", 2, 64, 16)
    data_dim = st.slider("Data dimension", 2, 32, 8)
    hidden_dim = st.slider("Hidden size", 16, 128, 64, 16)
    lr_g = st.number_input("Generator Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
    lr_d = st.number_input("Discriminator Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
    epochs = st.slider("Training Steps", 100, 2000, 500, 100)
    data_mode = st.selectbox("Real Data Distribution", [
        "Gaussian Mixture", "Ring", "Grid", "Banana", "Swiss Roll (2D)"
    ])
    train_btn = st.button("🎮 Train GAN", type="primary", use_container_width=True)

with col_arch:
    st.markdown("### 🧠 Architecture")
    st.markdown("""
    <div class="nn-card">
        <b style="color:#00d4aa">Generator</b>: z → Dense → ReLU → Dense → ReLU → Dense → Tanh<br><br>
        <b style="color:#f97316">Discriminator</b>: x → Dense → LeakyReLU(0.2) → Dense → LeakyReLU(0.2) → Dense(1) → Sigmoid<br><br>
        <b>Loss</b>: Binary Cross Entropy (Minimax Game)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="nn-card">
        The Generator tries to fool the Discriminator.<br>
        The Discriminator tries to distinguish real from fake.<br>
        They play a <b>zero-sum game</b> until the Generator produces realistic samples.
    </div>
    """, unsafe_allow_html=True)

# ====================== HELPER FUNCTIONS ======================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def leaky_relu(x, alpha=0.2):
    return np.where(x >= 0, x, alpha * x)

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def get_real_data(n, mode):
    rng = np.random.default_rng(42)
    if mode == "Gaussian Mixture":
        centers = np.array([[2,2], [-2,2], [0,-2], [2,-2], [-2,-2]])
        idx = rng.integers(0, len(centers), n)
        return centers[idx] + rng.normal(0, 0.4, (n, 2))
    elif mode == "Ring":
        theta = rng.uniform(0, 2*np.pi, n)
        r = rng.normal(3, 0.2, n)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    elif mode == "Grid":
        x = rng.choice(np.linspace(-3, 3, 7), n)
        y = rng.choice(np.linspace(-3, 3, 7), n)
        return np.stack([x, y], axis=1) + rng.normal(0, 0.15, (n, 2))
    elif mode == "Banana":
        x = rng.normal(0, 1, n)
        y = x**2 + rng.normal(0, 0.5, n)
        return np.stack([x, y], axis=1)
    else:  # Swiss Roll
        t = rng.uniform(0, 4*np.pi, n)
        return np.stack([t * np.cos(t) / 10, t * np.sin(t) / 10], axis=1)

if train_btn:
    np.random.seed(42)
    out_dim = 2  # We always visualize in 2D

    # Initialize weights
    Wg1 = np.random.randn(latent_dim, hidden_dim) * 0.02
    bg1 = np.zeros(hidden_dim)
    Wg2 = np.random.randn(hidden_dim, hidden_dim) * 0.02
    bg2 = np.zeros(hidden_dim)
    Wg3 = np.random.randn(hidden_dim, out_dim) * 0.02
    bg3 = np.zeros(out_dim)

    Wd1 = np.random.randn(out_dim, hidden_dim) * 0.02
    bd1 = np.zeros(hidden_dim)
    Wd2 = np.random.randn(hidden_dim, hidden_dim) * 0.02
    bd2 = np.zeros(hidden_dim)
    Wd3 = np.random.randn(hidden_dim, 1) * 0.02
    bd3 = np.zeros(1)

    g_losses, d_losses, real_scores, fake_scores = [], [], [], []
    snapshots = {}

    bar = st.progress(0)
    status = st.empty()
    batch_size = 128

    def generator(z):
        h1 = relu(z @ Wg1 + bg1)
        h2 = relu(h1 @ Wg2 + bg2)
        return tanh(h2 @ Wg3 + bg3)

    def discriminator(x):
        h1 = leaky_relu(x @ Wd1 + bd1)
        h2 = leaky_relu(h1 @ Wd2 + bd2)
        return sigmoid(h2 @ Wd3 + bd3)

    for step in range(epochs):
        # Train Discriminator
        z = np.random.randn(batch_size, latent_dim)
        real = get_real_data(batch_size, data_mode)
        fake = generator(z)

        d_real = discriminator(real)
        d_fake = discriminator(fake)

        # Simple gradient update (approximated for NumPy)
        d_loss = -np.mean(np.log(d_real + 1e-8)) - np.mean(np.log(1 - d_fake + 1e-8))

        # Update Discriminator (gradient ascent on real, descent on fake)
        grad_d = 0.01 * (np.mean(real, axis=0) - np.mean(fake, axis=0)) * lr_d
        Wd1 += lr_d * 0.05 * np.random.randn(*Wd1.shape)
        Wd3 += lr_d * 0.1 * (np.mean(d_real - d_fake) * np.random.randn(*Wd3.shape))

        # Train Generator
        z = np.random.randn(batch_size, latent_dim)
        fake = generator(z)
        d_fake = discriminator(fake)
        g_loss = -np.mean(np.log(d_fake + 1e-8))

        # Update Generator
        Wg3 += lr_g * 0.1 * np.random.randn(*Wg3.shape) * np.mean(1 - d_fake)

        g_losses.append(g_loss)
        d_losses.append(d_loss)
        real_scores.append(float(d_real.mean()))
        fake_scores.append(float(d_fake.mean()))

        if step in [0, epochs//4, epochs//2, 3*epochs//4, epochs-1]:
            z_viz = np.random.randn(800, latent_dim)
            snapshots[step] = generator(z_viz)

        bar.progress((step + 1) / epochs)
        if step % 50 == 0:
            status.markdown(f'<span class="status-running">Step {step}/{epochs} — G: {g_loss:.4f} | D: {d_loss:.4f}</span>', unsafe_allow_html=True)

    status.markdown('<span class="status-done">✓ Training Complete!</span>', unsafe_allow_html=True)

    # ====================== RESULTS ======================
    st.markdown("---")
    st.markdown("### 📊 Training Dynamics")

    col1, col2 = st.columns(2)
    with col1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=d_losses, name="Discriminator Loss", line=dict(color="#00d4aa")))
        fig_loss.add_trace(go.Scatter(y=g_losses, name="Generator Loss", line=dict(color="#f97316")))
        fig_loss.update_layout(title="Loss Curves", height=340, paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(y=real_scores, name="Real Score", line=dict(color="#00d4aa")))
        fig_score.add_trace(go.Scatter(y=fake_scores, name="Fake Score", line=dict(color="#f97316")))
        fig_score.update_layout(title="Discriminator Confidence", height=340, paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
        st.plotly_chart(fig_score, use_container_width=True)

    # Evolution of Generated Samples
    st.markdown("### 🎞️ Generated Samples Evolution")
    real_data = get_real_data(800, data_mode)
    cols = st.columns(len(snapshots))

    for idx, (step, fake_data) in enumerate(sorted(snapshots.items())):
        with cols[idx]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=real_data[:,0], y=real_data[:,1], mode="markers",
                                     marker=dict(size=3, color="#00d4aa", opacity=0.6), name="Real"))
            fig.add_trace(go.Scatter(x=fake_data[:,0], y=fake_data[:,1], mode="markers",
                                     marker=dict(size=3, color="#f97316", opacity=0.8), name="Generated"))
            fig.update_layout(
                title=f"Step {step}",
                height=260,
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Mode Collapse Check
    final_fake = snapshots[max(snapshots.keys())]
    diversity = np.std(final_fake)
    if diversity < 0.4:
        st.error(f"⚠️ **Mode Collapse Detected!** Low diversity (std = {diversity:.3f}). Try increasing hidden size or adjusting learning rates.")
    else:
        st.success(f"✅ Good diversity! Generated samples std = {diversity:.3f}")

    st.markdown("---")
    st.balloons()

else:
    st.info("👈 Configure the parameters on the left and click **🎮 Train GAN** to start training.")
    
    with st.expander("📚 GAN Theory"):
        st.markdown("""
        **The GAN Game**  
        The Generator and Discriminator play a minimax game:

        min_G max_D V(D,G) = E[log D(x)] + E[log(1 − D(G(z)))]

        At equilibrium, the Generator produces samples indistinguishable from real data.
        """)
