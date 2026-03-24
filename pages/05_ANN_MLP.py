import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import io
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="ANN / MLP — NeuralForge",
    layout="wide",
    page_icon="⬛"
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
        background: rgba(0, 212, 170, 0.15) !important;
        color: var(--accent) !important;
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
    st.title("⬛ ANN / MLP")
    st.markdown("Multi-Layer Perceptron")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 5</div>
    <h1>ANN / MLP — Neural Engine</h1>
    <p style="color: var(--muted); font-size: 1.1rem; margin-top: 0.5rem;">
        Build and train a fully connected neural network with NumPy (from scratch) or PyTorch backend.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Theory & Architecture", expanded=False):
    st.markdown("""
    A **Multi-Layer Perceptron (MLP)** is a feedforward neural network with one or more hidden layers.
    - **Forward Pass**: Input → Linear → Activation → ... → Output
    - **Loss**: Cross-Entropy (classification)
    - **Backward Pass**: Backpropagation + Gradient Descent
    **Activation functions**: ReLU, Tanh, Sigmoid, Leaky ReLU
    """)

st.markdown("---")

# ====================== CONTROLS ======================
col_ctrl, col_main = st.columns([1, 2])

with col_ctrl:
    st.markdown('<div class="nn-card">', unsafe_allow_html=True)
    source = st.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer", "Upload CSV"])
    backend = st.selectbox("Backend", ["NumPy (from scratch)", "PyTorch"])
    activation = st.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid", "LeakyReLU"])
    hidden_input = st.text_input("Hidden Layers (comma separated)", "64,32")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="nn-card">', unsafe_allow_html=True)
    epochs = st.slider("Epochs", 10, 500, 100)
    lr = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.4f")
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)
    upload = st.file_uploader("Upload Custom CSV (features + label)", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

# ====================== DATA LOADING ======================
with col_main:
    if source == "Upload CSV" and upload is not None:
        df = pd.read_csv(upload)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(int)
    else:
        if source == "Iris":
            data = load_iris()
        elif source == "Wine":
            data = load_wine()
        else:
            data = load_breast_cancer()
        X = data.data
        y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_classes = len(np.unique(y))
    n_features = X.shape[1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(X))
    c2.metric("Features", n_features)
    c3.metric("Classes", n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.markdown("### Training Workspace")
    train_btn = st.button("▶ Start Training Engine", type="primary", use_container_width=True)

# ====================== TRAINING ======================
if train_btn:
    hidden_sizes = [int(x.strip()) for x in hidden_input.split(",") if x.strip().isdigit()]
    input_dim = X_train.shape[1]
    output_dim = n_classes

    losses = []
    accuracies = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    if backend == "NumPy (from scratch)":
        st.info("Training with **NumPy from scratch**")
        np.random.seed(42)
        h = hidden_sizes[0] if hidden_sizes else 32

        W1 = np.random.randn(input_dim, h) * np.sqrt(2.0 / input_dim)
        b1 = np.zeros(h)
        W2 = np.random.randn(h, output_dim) * np.sqrt(2.0 / h)
        b2 = np.zeros(output_dim)

        for epoch in range(epochs):
            z1 = X_train @ W1 + b1
            a1 = np.maximum(0, z1) if activation == "ReLU" else np.tanh(z1)
            z2 = a1 @ W2 + b2
            ex = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
            probs = ex / ex.sum(axis=1, keepdims=True)

            y_onehot = np.eye(output_dim)[y_train]
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))
            losses.append(loss)

            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y_train)
            accuracies.append(acc)

            dz2 = (probs - y_onehot) / len(X_train)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0) if activation == "ReLU" else da1 * (1 - np.tanh(z1)**2)
            dW1 = X_train.T @ dz1
            db1 = dz1.sum(axis=0)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.1%}")

        st.success("✅ NumPy Training Completed!")

    else:
        # PyTorch Backend
        import torch
        import torch.nn as nn
        import torch.optim as optim

        st.info("Training with **PyTorch** backend")

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden_sizes:
                    layers.append(nn.Linear(prev, h))
                    if activation == "ReLU":
                        layers.append(nn.ReLU())
                    elif activation == "Tanh":
                        layers.append(nn.Tanh())
                    elif activation == "Sigmoid":
                        layers.append(nn.Sigmoid())
                    elif activation == "LeakyReLU":
                        layers.append(nn.LeakyReLU(0.01))
                    prev = h
                layers.append(nn.Linear(prev, output_dim))
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        model = MLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.long)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tr_t)
            loss = criterion(outputs, y_tr_t)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_tr_t).float().mean().item()
            accuracies.append(acc)

            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc:.1%}")

        st.success("✅ PyTorch Training Completed!")

    # ====================== RESULTS ======================
    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, name="Loss", line=dict(color="#00d4aa", width=2.5)))
    fig.add_trace(go.Scatter(y=accuracies, name="Accuracy", line=dict(color="#f97316", width=2.5), yaxis="y2"))
    fig.update_layout(
        title="Training Progress",
        height=380,
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        yaxis2=dict(overlaying="y", side="right", tickformat=".0%"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Final Test Evaluation
    if backend == "NumPy (from scratch)":
        z1 = X_test @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        probs = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        y_pred = np.argmax(probs, axis=1)
    else:
        model.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_test, dtype=torch.float32)
            outputs = model(X_te_t)
            y_pred = torch.argmax(outputs, dim=1).numpy()

    acc_test = np.mean(y_pred == y_test)

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Test Accuracy", f"{acc_test:.1%}")
    with col_res2:
        st.metric("Final Training Loss", f"{losses[-1]:.4f}")

    # ====================== DOWNLOAD (FIXED) ======================
    st.markdown("---")
    if backend == "NumPy (from scratch)":
        # Save NumPy weights to .npz
        buffer = io.BytesIO()
        np.savez_compressed(buffer, W1=W1, b1=b1, W2=W2, b2=b2)
        buffer.seek(0)

        st.download_button(
            label="⬇ Download Trained Weights (NumPy .npz)",
            data=buffer,
            file_name="mlp_weights.npz",
            mime="application/octet-stream"
        )
    else:
        # PyTorch weights
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        st.download_button(
            label="⬇ Download Trained Model (PyTorch .pt)",
            data=buffer,
            file_name="mlp_model.pt",
            mime="application/octet-stream"
        )

else:
    st.info("👈 Configure parameters and click **▶ Start Training Engine**")
