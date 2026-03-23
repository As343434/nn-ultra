"""Autoencoder — unsupervised dimensionality reduction with 2-D latent space."""
import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.data import load_iris, load_wine, standardize
from utils.export import download_code, download_torch
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row
from utils.viz import plot_loss_curve

st.set_page_config(page_title="Autoencoder", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("Autoencoder")

hero(
    "Autoencoder",
    "Unsupervised representation learning — compress data to a 2-D latent space and reconstruct it.",
    pill="Lesson 8", pill_variant="purple",
)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
An **autoencoder** learns to compress then reconstruct:

$$z = \text{Encoder}(x) \quad \hat{x} = \text{Decoder}(z)$$

Trained to minimize **reconstruction loss**:

$$L = \|x - \hat{x}\|^2$$

With a **2-D bottleneck** we can visualise the learned manifold.
Applications include anomaly detection, denoising, and pre-training.

**Variational Autoencoder (VAE)** adds a KL-divergence term to regularise the latent space:

$$L_{\text{VAE}} = \|x - \hat{x}\|^2 + \beta\,D_{KL}(q(z|x)\,\|\,\mathcal{N}(0,I))$$
""")

try:
    import torch, torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

if not TORCH_OK:
    st.error("PyTorch not installed."); st.stop()

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    source    = st.selectbox("Dataset", ["Iris", "Wine"])
    model_t   = st.selectbox("Model type", ["Autoencoder", "VAE"])
    enc_arch  = st.text_input("Encoder hidden sizes", "32,16")
    latent_d  = st.slider("Latent dimension", 2, 8, 2, 1)
    epochs    = st.slider("Epochs", 20, 500, 150, 20)
    lr        = st.slider("Learning rate", 0.0005, 0.05, 0.005, 0.0005, format="%.4f")
    beta_kl   = st.slider("β (KL weight, VAE only)", 0.0, 5.0, 1.0, 0.1,
                          disabled=(model_t != "VAE"))
    noise_lvl = st.slider("Input noise (denoising)", 0.0, 0.5, 0.0, 0.05)

with col2:
    if source == "Iris":
        X_df, y_s = load_iris()
    else:
        X_df, y_s = load_wine()

    X = standardize(X_df.values.astype(float)).astype(np.float32)
    y = y_s.values.astype(int)

    metric_row([
        ("Samples",   len(X)),
        ("Features",  X.shape[1]),
        ("Classes",   len(np.unique(y))),
        ("Latent dim", latent_d),
    ])
    st.info("The encoder maps each sample to the 2-D (or N-D) latent space. "
            "The scatter plot shows class clusters that emerge **without labels**.")

train_btn = st.button("▶ Train Autoencoder", type="primary")

if train_btn:
    torch.manual_seed(42)
    in_d = X.shape[1]
    enc_sizes = [int(s) for s in enc_arch.split(",") if s.strip().isdigit()]

    # ── Build encoder / decoder ────────────────────────────────────────
    def make_mlp(dims, act=nn.ReLU):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(act())
        return nn.Sequential(*layers)

    if model_t == "Autoencoder":
        enc = make_mlp([in_d] + enc_sizes + [latent_d])
        dec = make_mlp([latent_d] + enc_sizes[::-1] + [in_d])

        class AE(nn.Module):
            def __init__(self): super().__init__(); self.enc = enc; self.dec = dec
            def forward(self, x): z = self.enc(x); return self.dec(z), z, None, None

    else:  # VAE
        enc_body = make_mlp([in_d] + enc_sizes)
        mu_layer  = nn.Linear(enc_sizes[-1] if enc_sizes else in_d, latent_d)
        lv_layer  = nn.Linear(enc_sizes[-1] if enc_sizes else in_d, latent_d)
        dec = make_mlp([latent_d] + enc_sizes[::-1] + [in_d])

        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc_body = enc_body
                self.mu_l  = mu_layer
                self.lv_l  = lv_layer
                self.dec   = dec
            def forward(self, x):
                h  = self.enc_body(x)
                mu = self.mu_l(h); lv = self.lv_l(h)
                z  = mu + torch.randn_like(mu) * (0.5 * lv).exp()
                return self.dec(z), z, mu, lv

    model = AE()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    X_t   = torch.from_numpy(X)

    losses, recon_losses, kl_losses = [], [], []
    bar = st.progress(0)

    for ep in range(epochs):
        # Add noise for denoising AE
        X_noisy = X_t + noise_lvl * torch.randn_like(X_t) if noise_lvl > 0 else X_t
        opt.zero_grad()
        x_hat, z, mu, lv = model(X_noisy)
        recon = nn.MSELoss()(x_hat, X_t)
        if model_t == "VAE" and mu is not None:
            kl = -0.5 * torch.mean(1 + lv - mu**2 - lv.exp())
            loss = recon + beta_kl * kl
            kl_losses.append(kl.item())
        else:
            loss = recon
            kl_losses.append(0.0)
        loss.backward(); opt.step()
        losses.append(loss.item())
        recon_losses.append(recon.item())
        bar.progress(int((ep+1)/epochs*100))
        time.sleep(0.005)

    st.success(f"✓ Done — final reconstruction loss: **{recon_losses[-1]:.5f}**")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_loss_curve(losses), use_container_width=True)
    with c2:
        if model_t == "VAE":
            fig_kl = go.Figure()
            fig_kl.add_trace(go.Scatter(y=kl_losses, mode="lines",
                                        line=dict(color="#f97316", width=2)))
            fig_kl.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                                 font=dict(color="#8b949e"), height=300,
                                 margin=dict(l=10,r=10,t=30,b=10),
                                 title=dict(text="KL Loss", font=dict(color="#e6edf3")))
            st.plotly_chart(fig_kl, use_container_width=True)
        else:
            metric_row([("Final loss", f"{losses[-1]:.5f}"),
                        ("Recon loss", f"{recon_losses[-1]:.5f}")])

    # ── Latent space scatter ────────────────────────────────────────────
    st.markdown("#### Latent space (first 2 dims)")
    with torch.no_grad():
        _, z_all, mu_all, _ = model(X_t)
        z_np = (mu_all if mu_all is not None else z_all).numpy()

    fig_z = px.scatter(
        x=z_np[:, 0], y=z_np[:, 1] if latent_d > 1 else np.zeros(len(z_np)),
        color=y.astype(str),
        color_discrete_sequence=["#00d4aa", "#f97316", "#818cf8", "#f85149", "#3fb950"],
        template="plotly_dark",
        labels={"x": "z₁", "y": "z₂", "color": "class"},
        title=f"Latent space — {source}",
    )
    fig_z.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                        font=dict(color="#8b949e"), height=380)
    st.plotly_chart(fig_z, use_container_width=True)

    # ── Reconstruction quality ──────────────────────────────────────────
    st.markdown("#### Reconstruction quality — first 6 samples")
    with torch.no_grad():
        x_hat_np = model(X_t)[0].numpy()

    fig_rc, axes = plt.subplots(2, 6, figsize=(12, 3.5), facecolor="#0d1117")
    for i in range(6):
        ax_o, ax_r = axes[0, i], axes[1, i]
        ax_o.bar(range(in_d), X[i],      color="#00d4aa", width=0.8)
        ax_r.bar(range(in_d), x_hat_np[i], color="#f97316", width=0.8)
        for ax in (ax_o, ax_r):
            ax.set_facecolor("#0d1117"); ax.tick_params(colors="#8b949e", labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor("rgba(255,255,255,0.08)")
        if i == 0:
            ax_o.set_ylabel("Original", color="#8b949e", fontsize=8)
            ax_r.set_ylabel("Reconstructed", color="#8b949e", fontsize=8)
    plt.suptitle("Feature-wise reconstruction", color="#e6edf3", fontsize=10)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig_rc)

    download_torch("⬇ Download model", model.state_dict(), "autoencoder.pt")

    code = f"""\
import torch, torch.nn as nn

# {model_t} — latent_dim={latent_d}
enc = nn.Sequential(
    nn.Linear({in_d}, {enc_sizes[0] if enc_sizes else latent_d}), nn.ReLU(),
    nn.Linear({enc_sizes[0] if enc_sizes else latent_d}, {latent_d}),
)
dec = nn.Sequential(
    nn.Linear({latent_d}, {enc_sizes[0] if enc_sizes else in_d}), nn.ReLU(),
    nn.Linear({enc_sizes[0] if enc_sizes else in_d}, {in_d}),
)

opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr={lr})
for epoch in range({epochs}):
    z    = enc(X_t)
    xhat = dec(z)
    loss = nn.MSELoss()(xhat, X_t)
    loss.backward(); opt.step(); opt.zero_grad()
"""
    download_code("⬇ Export Python", code, "autoencoder.py")
