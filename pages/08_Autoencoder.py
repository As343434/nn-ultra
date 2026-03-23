import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Autoencoder / VAE",
    layout="wide",
    page_icon="◎"
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
        background: rgba(129, 140, 248, 0.15) !important;
        color: #818cf8 !important;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    button[kind="primary"] {
        background: var(--accent) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("◎ Autoencoder")
    st.markdown("Unsupervised Learning")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 8</div>
    <h1>Autoencoder / VAE</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Compress data into a low-dimensional latent space and learn to reconstruct it.<br>
        Visualize the learned 2D manifold without using class labels.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
An **Autoencoder** learns compression and reconstruction:

$$z = \text{Encoder}(x) \quad \hat{x} = \text{Decoder}(z)$$

Trained with **reconstruction loss**:
$$L = \|x - \hat{x}\|^2$$

**Variational Autoencoder (VAE)** adds regularization:
$$L_{VAE} = \|x - \hat{x}\|^2 + \beta \, D_{KL}(q(z|x) \parallel \mathcal{N}(0,I))$$

A **2D latent space** lets us visualize how the model organizes data **unsupervised**.
""")

st.markdown("---")

# ====================== CONTROLS ======================
col1, col2 = st.columns([1, 2])

with col1:
    dataset_name = st.selectbox("Dataset", ["Iris", "Wine"])
    model_type = st.selectbox("Model Type", ["Autoencoder", "VAE"])
    enc_arch = st.text_input("Encoder Hidden Layers", "32,16")
    latent_dim = st.slider("Latent Dimension", 2, 8, 2)
    epochs = st.slider("Epochs", 20, 500, 150, 20)
    lr = st.slider("Learning Rate", 0.0005, 0.05, 0.005, format="%.4f")
    
    if model_type == "VAE":
        beta_kl = st.slider("β (KL Divergence Weight)", 0.0, 5.0, 1.0, 0.1)
    else:
        beta_kl = 0.0
    
    noise_level = st.slider("Denoising Noise Level", 0.0, 0.5, 0.0, 0.05)

with col2:
    # Load data
    if dataset_name == "Iris":
        data = load_iris()
    else:
        data = load_wine()
    
    X = data.data.astype(np.float32)
    y = data.target.astype(int)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(X))
    c2.metric("Features", X.shape[1])
    c3.metric("Classes", len(np.unique(y)))
    
    st.info("The model learns a compressed representation **without using class labels**.")

# ====================== TRAINING ======================
train_btn = st.button("▶ Train Autoencoder", type="primary", use_container_width=True)

if train_btn:
    torch.manual_seed(42)
    X_t = torch.from_numpy(X)
    input_dim = X.shape[1]
    
    # Parse encoder architecture
    hidden_sizes = [int(x.strip()) for x in enc_arch.split(",") if x.strip().isdigit()]
    
    # Build MLP helper
    def make_mlp(dims):
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    if model_type == "Autoencoder":
        encoder = make_mlp([input_dim] + hidden_sizes + [latent_dim])
        decoder = make_mlp([latent_dim] + hidden_sizes[::-1] + [input_dim])
        
        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
            def forward(self, x):
                z = self.encoder(x)
                x_hat = self.decoder(z)
                return x_hat, z, None, None
    else:  # VAE
        encoder_body = make_mlp([input_dim] + hidden_sizes)
        mu_layer = nn.Linear(hidden_sizes[-1] if hidden_sizes else input_dim, latent_dim)
        logvar_layer = nn.Linear(hidden_sizes[-1] if hidden_sizes else input_dim, latent_dim)
        decoder = make_mlp([latent_dim] + hidden_sizes[::-1] + [input_dim])
        
        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_body = encoder_body
                self.mu_layer = mu_layer
                self.logvar_layer = logvar_layer
                self.decoder = decoder
            def forward(self, x):
                h = self.encoder_body(x)
                mu = self.mu_layer(h)
                logvar = self.logvar_layer(h)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                x_hat = self.decoder(z)
                return x_hat, z, mu, logvar
    
    model = AE()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    recon_losses = []
    kl_losses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Add noise for denoising
        X_input = X_t + noise_level * torch.randn_like(X_t) if noise_level > 0 else X_t
        
        x_hat, z, mu, logvar = model(X_input)
        
        recon_loss = nn.MSELoss()(x_hat, X_t)
        
        if model_type == "VAE" and mu is not None:
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta_kl * kl_loss
            kl_losses.append(kl_loss.item())
        else:
            loss = recon_loss
            kl_losses.append(0.0)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        recon_losses.append(recon_loss.item())
        
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f}")
    
    st.success(f"✅ Training Complete! Final Reconstruction Loss: **{recon_losses[-1]:.5f}**")

    # Loss Curves
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=recon_losses, name="Reconstruction Loss", line=dict(color="#00d4aa")))
    if model_type == "VAE":
        fig_loss.add_trace(go.Scatter(y=kl_losses, name="KL Divergence", line=dict(color="#f97316")))
    fig_loss.update_layout(title="Training Losses", height=340, paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
    st.plotly_chart(fig_loss, use_container_width=True)

    # Latent Space Visualization
    st.markdown("#### 2D Latent Space Visualization")
    with torch.no_grad():
        _, z_all, mu_all, _ = model(X_t)
        z_np = (mu_all if mu_all is not None else z_all).numpy()
    
    # Always show 2D (take first two dimensions if latent > 2)
    if z_np.shape[1] > 2:
        z_plot = z_np[:, :2]
    else:
        z_plot = z_np
    
    fig_latent = px.scatter(
        x=z_plot[:, 0], y=z_plot[:, 1] if z_plot.shape[1] > 1 else np.zeros(len(z_plot)),
        color=y.astype(str),
        color_discrete_sequence=["#00d4aa", "#f97316", "#818cf8", "#f85149"],
        title=f"Latent Space ({model_type}) — Colored by True Class",
        labels={"x": "z₁", "y": "z₂", "color": "Class"}
    )
    fig_latent.update_layout(height=420, paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
    st.plotly_chart(fig_latent, use_container_width=True)

    # Reconstruction Quality
    st.markdown("#### Reconstruction Quality — First 8 Samples")
    with torch.no_grad():
        x_hat_np = model(X_t)[0].numpy()
    
    # Create bar plots for reconstruction
    cols = st.columns(4)
    for i in range(8):
        with cols[i % 4]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(input_dim)), y=X[i], name="Original", marker_color="#00d4aa"))
            fig.add_trace(go.Bar(x=list(range(input_dim)), y=x_hat_np[i], name="Reconstructed", marker_color="#f97316"))
            fig.update_layout(
                title=f"Sample {i+1}",
                height=280,
                barmode='group',
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9"),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Download Options
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        torch.save(model.state_dict(), "autoencoder.pt")
        with open("autoencoder.pt", "rb") as f:
            st.download_button(
                label="⬇ Download Trained Model",
                data=f,
                file_name="autoencoder.pt",
                mime="application/octet-stream"
            )
    
    with col_d2:
        code_str = f"""import torch
import torch.nn as nn

# Simple Autoencoder / VAE skeleton
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear({input_dim}, {hidden_sizes[0] if hidden_sizes else latent_dim}), nn.ReLU(),
            nn.Linear({hidden_sizes[0] if hidden_sizes else latent_dim}, {latent_dim})
        )
        self.decoder = nn.Sequential(
            nn.Linear({latent_dim}, {hidden_sizes[0] if hidden_sizes else input_dim}), nn.ReLU(),
            nn.Linear({hidden_sizes[0] if hidden_sizes else input_dim}, {input_dim})
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = Autoencoder()
"""
        st.download_button(
            label="⬇ Export Model Code",
            data=code_str,
            file_name="autoencoder.py",
            mime="text/plain"
        )

    st.balloons()
