import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import streamlit as st

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="RNN / LSTM / GRU",
    layout="wide",
    page_icon="⇌"
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
        border-radius: 8px !important;
    }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("⇌ RNN / LSTM / GRU")
    st.markdown("Sequence Modeling Lab")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 7</div>
    <h1>RNN / LSTM / GRU</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Predict sequences with Recurrent Neural Networks.<br>
        Visualize hidden state evolution over time.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
**RNN**: Simple recurrent unit with hidden state  
$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

**LSTM**: Adds memory cell and three gates to solve vanishing gradient problem  
**GRU**: Simplified LSTM with fewer parameters

**Best Practice**: Use **LSTM** or **GRU** for long sequences (>30 steps).
""")

st.markdown("---")

# ====================== CONTROLS ======================
col1, col2 = st.columns([1, 2])

with col1:
    model_type = st.selectbox("Model Type", ["RNN", "LSTM", "GRU"])
    task = st.selectbox("Task", ["Sine wave", "Noisy sine", "Sum of harmonics"])
    seq_len = st.slider("Sequence Length", 10, 100, 50, 5)
    hidden_size = st.slider("Hidden Size", 8, 128, 32, 8)
    num_layers = st.slider("Number of Layers", 1, 3, 1)
    epochs = st.slider("Epochs", 20, 400, 100, 20)
    lr = st.slider("Learning Rate", 0.0005, 0.05, 0.005, format="%.4f")
    dropout = st.slider("Dropout", 0.0, 0.5, 0.0, 0.05)

with col2:
    # Generate sequence preview
    t = np.linspace(0, 10 * np.pi, 800)
    if task == "Sine wave":
        series = np.sin(t).astype(np.float32)
    elif task == "Noisy sine":
        np.random.seed(42)
        series = (np.sin(t) + 0.25 * np.random.randn(len(t))).astype(np.float32)
    else:
        series = (np.sin(t) + 0.5 * np.sin(3 * t) + 0.25 * np.cos(5 * t)).astype(np.float32)

    fig_preview = go.Figure()
    fig_preview.add_trace(go.Scatter(y=series[:250], mode="lines",
                                     line=dict(color="#00d4aa", width=2)))
    fig_preview.update_layout(
        title="Input Sequence Preview (first 250 points)",
        height=240,
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9")
    )
    st.plotly_chart(fig_preview, use_container_width=True)

# ====================== TRAINING ======================
train_btn = st.button("▶ Train Sequence Model", type="primary", use_container_width=True)

if train_btn:
    torch.manual_seed(42)
    np.random.seed(42)

    # Prepare data
    X_seq = np.array([series[i:i+seq_len] for i in range(len(series) - seq_len - 1)], dtype=np.float32)
    y_seq = np.array([series[i + seq_len] for i in range(len(series) - seq_len - 1)], dtype=np.float32)

    X_t = torch.from_numpy(X_seq).unsqueeze(-1)   # (N, seq_len, 1)
    y_t = torch.from_numpy(y_seq).unsqueeze(-1)   # (N, 1)

    # Model
    if model_type == "RNN":
        rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                     batch_first=True, dropout=dropout if num_layers > 1 else 0)
    elif model_type == "LSTM":
        rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                      batch_first=True, dropout=dropout if num_layers > 1 else 0)
    else:  # GRU
        rnn = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                     batch_first=True, dropout=dropout if num_layers > 1 else 0)

    head = nn.Linear(hidden_size, 1)

    optimizer = optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for ep in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == "LSTM":
            out, (h_n, c_n) = rnn(X_t)
        else:
            out, h_n = rnn(X_t)
        
        last_hidden = out[:, -1, :]          # (batch, hidden)
        pred = head(last_hidden)
        
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.progress((ep + 1) / epochs)
        status_text.text(f"Epoch {ep+1}/{epochs} | Loss: {loss.item():.6f}")

    st.success(f"✅ Training Completed! Final MSE: **{losses[-1]:.6f}**")

    # Loss Curve
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=losses, mode='lines', name='MSE Loss',
                                  line=dict(color='#00d4aa', width=3)))
    fig_loss.update_layout(title="Training Loss Curve", height=320,
                           paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
    st.plotly_chart(fig_loss, use_container_width=True)

    # Predictions vs Ground Truth
    with torch.no_grad():
        if model_type == "LSTM":
            out, _ = rnn(X_t)
        else:
            out, _ = rnn(X_t)
        preds = head(out[:, -1, :]).squeeze().numpy()

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=y_seq[:400], mode="lines", name="Target",
                                  line=dict(color="#00d4aa", width=2)))
    fig_pred.add_trace(go.Scatter(y=preds[:400], mode="lines", name="Prediction",
                                  line=dict(color="#f97316", dash="dash", width=2)))
    fig_pred.update_layout(
        title="Target vs Prediction (first 400 steps)",
        height=340,
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Hidden State Heatmap (first sample)
    st.markdown("#### Hidden State Evolution (First Sequence)")
    with torch.no_grad():
        if model_type == "LSTM":
            out, _ = rnn(X_t[:1])
        else:
            out, _ = rnn(X_t[:1])
        hidden_states = out[0].numpy()   # (seq_len, hidden_size)

    # Show first min(32, hidden) units
    n_show = min(32, hidden_states.shape[1])
    hidden_crop = hidden_states[:, :n_show].T

    fig_hm = go.Figure(data=go.Heatmap(
        z=hidden_crop,
        colorscale="Magma",
        showscale=True
    ))
    fig_hm.update_layout(
        title=f"{model_type} Hidden States over Time (first {n_show} units)",
        xaxis_title="Time Step",
        yaxis_title="Hidden Unit",
        height=420,
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Download Options
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        model_dict = {"rnn": rnn.state_dict(), "head": head.state_dict()}
        torch.save(model_dict, f"{model_type.lower()}_seq_model.pt")
        with open(f"{model_type.lower()}_seq_model.pt", "rb") as f:
            st.download_button(
                label="⬇ Download Trained Model",
                data=f,
                file_name=f"{model_type.lower()}_seq_model.pt",
                mime="application/octet-stream"
            )
    
    with col_d2:
        code_str = f"""import torch
import torch.nn as nn

rnn = nn.{model_type}(1, {hidden_size}, num_layers={num_layers}, batch_first=True)
head = nn.Linear({hidden_size}, 1)

# Training loop example:
for epoch in range({epochs}):
    out = rnn(X)
    pred = head(out[:, -1, :])
    loss = nn.MSELoss()(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""
        st.download_button(
            label="⬇ Export Training Code",
            data=code_str,
            file_name=f"{model_type.lower()}_sequence.py",
            mime="text/plain"
        )

    st.balloons()
