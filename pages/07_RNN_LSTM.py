"""RNN / LSTM — sequence modeling lab with hidden-state heatmap."""
import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.export import download_code, download_torch
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero
from utils.viz import plot_loss_curve

st.set_page_config(page_title="RNN / LSTM", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("RNN / LSTM")

hero(
    "RNN / LSTM",
    "Predict sequences with RNN or LSTM. Visualize hidden-state evolution over time.",
    pill="Lesson 7", pill_variant="orange",
)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
**RNN** maintains a hidden state $h_t$ across time steps:

$$h_t = \tanh(W_x\,x_t + W_h\,h_{t-1} + b_h)$$

**LSTM** adds gating to avoid vanishing gradients:

$$f_t = \sigma(W_f[h_{t-1},x_t]+b_f), \quad
  i_t = \sigma(W_i[h_{t-1},x_t]+b_i)$$
$$\tilde{c}_t = \tanh(W_c[h_{t-1},x_t]+b_c), \quad
  c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o[h_{t-1},x_t]+b_o), \quad h_t = o_t \odot \tanh(c_t)$$

**Use LSTM** when the task needs memory over long horizons (>50 steps).
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
    model_type = st.selectbox("Model", ["RNN", "LSTM", "GRU"])
    task       = st.selectbox("Task", ["Sine wave", "Noisy sine", "Sum of harmonics"])
    seq_len    = st.slider("Sequence length", 10, 80, 40, 5)
    hidden     = st.slider("Hidden size", 8, 128, 32, 8)
    n_layers   = st.slider("Layers (stacked)", 1, 3, 1, 1)
    epochs     = st.slider("Epochs", 20, 400, 100, 20)
    lr         = st.slider("Learning rate", 0.0005, 0.05, 0.005, 0.0005, format="%.4f")
    dropout    = st.slider("Dropout", 0.0, 0.5, 0.0, 0.05)

with col2:
    t = np.linspace(0, 10 * np.pi, 800)
    if task == "Sine wave":
        series = np.sin(t).astype(np.float32)
    elif task == "Noisy sine":
        series = (np.sin(t) + 0.25 * np.random.default_rng(0).normal(size=len(t))).astype(np.float32)
    else:
        series = (np.sin(t) + 0.5*np.sin(3*t) + 0.25*np.cos(5*t)).astype(np.float32)

    fig_preview = go.Figure()
    fig_preview.add_trace(go.Scatter(y=series[:200], mode="lines",
                                     line=dict(color="#00d4aa", width=1.5)))
    fig_preview.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                              font=dict(color="#8b949e"),
                              height=200, margin=dict(l=10,r=10,t=10,b=10),
                              title=dict(text="Input series (first 200 pts)", font=dict(color="#e6edf3")))
    st.plotly_chart(fig_preview, use_container_width=True)

train_btn = st.button("▶ Train", type="primary")

if train_btn:
    torch.manual_seed(42)
    X_seq = np.array([series[i:i+seq_len]   for i in range(len(series)-seq_len-1)], dtype=np.float32)
    y_seq = np.array([series[i+seq_len]      for i in range(len(series)-seq_len-1)], dtype=np.float32)
    X_t = torch.from_numpy(X_seq).unsqueeze(-1)  # (N, seq, 1)
    y_t = torch.from_numpy(y_seq).unsqueeze(-1)

    model_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[model_type]
    rnn = model_cls(1, hidden, num_layers=n_layers, batch_first=True,
                    dropout=dropout if n_layers > 1 else 0.0)
    head = nn.Linear(hidden, 1)
    params = list(rnn.parameters()) + list(head.parameters())
    opt  = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = nn.MSELoss()

    bar    = st.progress(0)
    losses = []
    for ep in range(epochs):
        opt.zero_grad()
        out = rnn(X_t)
        h_last = out[0][:, -1, :]     # works for RNN/GRU/LSTM
        pred = head(h_last)
        loss = crit(pred, y_t)
        loss.backward(); opt.step()
        sched.step(loss.item())
        losses.append(loss.item())
        bar.progress(int((ep+1)/epochs*100))
        time.sleep(0.005)

    st.success(f"✓ Done — final MSE: **{losses[-1]:.5f}**")
    st.plotly_chart(plot_loss_curve(losses), use_container_width=True)

    # Predictions vs targets
    with torch.no_grad():
        out = rnn(X_t)
        preds_np = head(out[0][:, -1, :]).squeeze().numpy()

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=y_seq[:300], mode="lines", name="Target",
                                  line=dict(color="#00d4aa")))
    fig_pred.add_trace(go.Scatter(y=preds_np[:300], mode="lines", name="Pred",
                                  line=dict(color="#f97316", dash="dash")))
    fig_pred.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                           font=dict(color="#8b949e"), height=280,
                           margin=dict(l=10,r=10,t=30,b=10),
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           title=dict(text="Target vs Predicted (300 steps)", font=dict(color="#e6edf3")))
    st.plotly_chart(fig_pred, use_container_width=True)

    # Hidden state heatmap
    st.markdown("#### Hidden-state heatmap (first sample)")
    with torch.no_grad():
        raw = rnn(X_t[:1])[0][0].numpy()  # (seq, hidden)
    n_h = min(32, raw.shape[1])
    raw_crop = raw[:, :n_h]
    fig3, ax3 = plt.subplots(figsize=(10, 2.5), facecolor="#0d1117")
    im = ax3.imshow(raw_crop.T, aspect="auto", cmap="magma",
                    origin="lower", interpolation="nearest")
    ax3.set_xlabel("Time step", color="#8b949e")
    ax3.set_ylabel("Unit", color="#8b949e")
    ax3.set_title(f"{model_type} hidden state (first {n_h} units)", color="#e6edf3")
    ax3.tick_params(colors="#8b949e")
    for sp in ax3.spines.values(): sp.set_edgecolor("rgba(255,255,255,0.08)")
    plt.colorbar(im, ax=ax3)
    plt.tight_layout()
    st.pyplot(fig3)

    download_torch("⬇ Download model",
                   {"rnn": rnn.state_dict(), "head": head.state_dict()},
                   f"{model_type.lower()}_model.pt")

    code = f"""\
import torch, torch.nn as nn

rnn  = nn.{model_type}(1, {hidden}, num_layers={n_layers}, batch_first=True)
head = nn.Linear({hidden}, 1)
opt  = torch.optim.Adam(list(rnn.parameters())+list(head.parameters()), lr={lr})

for epoch in range({epochs}):
    out     = rnn(X_t)
    h_last  = out[0][:, -1, :]
    pred    = head(h_last)
    loss    = nn.MSELoss()(pred, y_t)
    loss.backward(); opt.step(); opt.zero_grad()
"""
    download_code("⬇ Export Python", code, f"{model_type.lower()}.py")
