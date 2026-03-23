import time
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Note: These imports assume your local project structure exists. 
# If running standalone, replace these with your actual helper logic.
from utils.data import load_iris, load_wine, load_breast_cancer, standardize, csv_to_xy
from utils.export import download_code, download_pickle, download_torch
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row
from utils.viz import plot_loss_curve, plot_accuracy_curve, plot_confusion_matrix, plot_weight_heatmap

# ====================== CONFIG & THEME ======================
st.set_page_config(page_title="ANN / MLP", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("ANN / MLP")

hero(
    "ANN / MLP",
    "Build and train a Multi-Layer Perceptron on real datasets — NumPy or PyTorch backend, confusion matrix, weight heatmaps.",
    pill="Lesson 5",
)

# ====================== THEORY ======================
with st.expander("📖 Theory", expanded=False):
    st.markdown("""
MLPs stack **linear layers** and **nonlinear activations**:

$$\\mathbf{a}^{(l)} = \\sigma(W^{(l)}\\,\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)})$$



Training minimizes cross-entropy via **mini-batch gradient descent** and backprop.

**Key hyperparameters:**
- **Layer sizes:** Wider layers = more capacity; deeper layers = better feature composition.
- **Learning rate:** Crucial for stability; too high causes divergence, too low causes stagnation.
- **Activation:** **ReLU** is the modern standard; **Sigmoid/Softmax** is used for the output probabilities.
""")

st.markdown("---")

# ====================== CONTROLS ======================
col1, col2 = st.columns([1, 2])

with col1:
    source   = st.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer", "Upload CSV"])
    backend  = st.selectbox("Backend", ["NumPy (scratch)", "PyTorch"])
    act_name = st.selectbox("Hidden activation", ["ReLU", "Tanh", "Sigmoid", "LeakyReLU"])
    hidden   = st.text_input("Hidden layer sizes", "64,32", help="e.g. 64,32 → two hidden layers")
    
    st.markdown("---")
    epochs   = st.slider("Epochs", 5, 500, 100, 5)
    lr       = st.slider("Learning rate", 0.0001, 0.5, 0.01, 0.0001, format="%.4f")
    bs       = st.slider("Batch size", 8, 256, 32, 8)
    test_sz  = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    
    upload   = st.file_uploader("Upload CSV (last col = label)", type=["csv"])

# ====================== DATA LOADING ======================
with col2:
    if source == "Iris":
        X_df, y_s = load_iris()
    elif source == "Wine":
        X_df, y_s = load_wine()
    elif source == "Breast Cancer":
        X_df, y_s = load_breast_cancer()
    else:
        if upload:
            X_raw, y_raw = csv_to_xy(upload.getvalue())
            X_df = pd.DataFrame(X_raw)
            y_s  = pd.Series(y_raw)
        else:
            st.warning("Please upload a CSV file to proceed with custom data.")
            st.stop()

    X = standardize(X_df.values.astype(float))
    y = y_s.values.astype(int)
    classes = np.unique(y)
    n_cls = len(classes)

    metric_row([
        ("Samples", len(X)),
        ("Features", X.shape[1]),
        ("Classes", n_cls),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_sz, 
                                            random_state=42, stratify=y)
    st.info(f"📊 **Train set:** {len(X_tr)} samples  |  **Test set:** {len(X_te)} samples")

st.markdown("---")

# ====================== TRAINING ======================
st.markdown("### Train")
if st.button("▶ Train MLP", type="primary"):
    # Parse hidden layers
    try:
        sizes = [int(s.strip()) for s in hidden.split(",") if s.strip()]
    except ValueError:
        st.error("Invalid hidden layer format. Use numbers separated by commas (e.g., 64,32).")
        st.stop()

    in_d, out_d = X_tr.shape[1], n_cls
    losses, val_losses, accs = [], [], []

    # ── NumPy backend ──────────────────────────────────────────────────────
    if backend.startswith("NumPy"):
        # Activations and derivatives
        act_fns = {
            "ReLU":      (lambda z: np.maximum(0, z),           lambda z, a: (z > 0).astype(float)),
            "Tanh":      (np.tanh,                              lambda z, a: 1 - a**2),
            "Sigmoid":   (lambda z: 1/(1+np.exp(-z)),            lambda z, a: a*(1-a)),
            "LeakyReLU": (lambda z: np.where(z>=0, z, 0.01*z),  lambda z, a: np.where(z>=0, 1.0, 0.01)),
        }
        act_f, act_d = act_fns[act_name]

        dims = [in_d] + sizes + [out_d]
        Ws = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2/dims[i]) for i in range(len(dims)-1)]
        bs_list = [np.zeros((1, dims[i+1])) for i in range(len(dims)-1)]

        y_oh_tr = np.eye(out_d)[y_tr]
        y_oh_te = np.eye(out_d)[y_te]

        def softmax(z):
            e = np.exp(z - z.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)

        def fwd(X_in):
            a, zs, acts = X_in, [], [X_in]
            for i, (W, b) in enumerate(zip(Ws, bs_list)):
                z = a @ W + b
                zs.append(z)
                a = act_f(z) if i < len(Ws)-1 else z
                acts.append(a)
            return acts, zs

        bar = st.progress(0)
        for ep in range(epochs):
            idx = np.random.permutation(len(X_tr))
            ep_loss = 0.0
            for i in range(0, len(X_tr), bs):
                bi = idx[i:i+bs]
                Xb, Yb = X_tr[bi], y_oh_tr[bi]
                
                # Forward
                acts, zs = fwd(Xb)
                probs = softmax(acts[-1])
                ep_loss += -np.mean(np.sum(Yb * np.log(probs + 1e-9), 1))
                
                # Backward
                grad = (probs - Yb) / len(Xb)
                for j in reversed(range(len(Ws))):
                    dW = acts[j].T @ grad
                    db = grad.sum(0, keepdims=True)
                    # Update
                    Ws[j] -= lr * dW
                    bs_list[j] -= lr * db
                    if j > 0:
                        grad = (grad @ Ws[j].T) * act_d(zs[j-1], acts[j])

            losses.append(ep_loss / (len(X_tr)//bs + 1))
            
            # Validation
            acts_te, _ = fwd(X_te)
            probs_te = softmax(acts_te[-1])
            val_l = -np.mean(np.sum(y_oh_te * np.log(probs_te + 1e-9), 1))
            val_losses.append(val_l)
            accs.append(np.mean(np.argmax(probs_te, 1) == y_te))
            bar.progress((ep + 1) / epochs)

        preds = np.argmax(softmax(fwd(X_te)[0][-1]), 1)
        download_pickle("⬇ NumPy model", {"Ws": Ws, "bs": bs_list}, "mlp_numpy.pkl")

    # ── PyTorch backend ────────────────────────────────────────────────────
    else:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            st.error("PyTorch not found in the current environment.")
            st.stop()

        torch.manual_seed(42)
        act_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid, "LeakyReLU": nn.LeakyReLU}
        
        dims = [in_d] + sizes + [out_d]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(act_map[act_name]())
        
        model = nn.Sequential(*layers)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)

        bar = st.progress(0)
        for ep in range(epochs):
            idx = torch.randperm(len(X_tr_t))
            ep_loss = 0.0
            model.train()
            for i in range(0, len(X_tr_t), bs):
                bi = idx[i:i+bs]
                opt.zero_grad()
                out = model(X_tr_t[bi])
                loss = crit(out, y_tr_t[bi])
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            
            losses.append(ep_loss / (len(X_tr_t)//bs + 1))
            
            model.eval()
            with torch.no_grad():
                val_out = model(X_te_t)
                val_l = crit(val_out, torch.tensor(y_te, dtype=torch.long)).item()
                val_losses.append(val_l)
                accs.append((val_out.argmax(1).numpy() == y_te).mean())
            bar.progress((ep + 1) / epochs)

        preds = model(X_te_t).argmax(1).numpy()
        download_torch("⬇ PyTorch model", model.state_dict(), "mlp_torch.pt")

    # ====================== RESULTS ======================
    final_acc = np.mean(preds == y_te)
    st.success(f"✓ Training complete — Test accuracy: **{final_acc:.2%}**")

    metric_row([
        ("Test Acc", f"{final_acc:.2%}"),
        ("Final Loss", f"{losses[-1]:.4f}"),
        ("Val Loss", f"{val_losses[-1]:.4f}"),
        ("Epochs", epochs),
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_loss_curve(losses, val_losses), use_container_width=True)
    with c2:
        st.plotly_chart(plot_accuracy_curve(accs), use_container_width=True)

    cm = confusion_matrix(y_te, preds)
    st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

    if backend.startswith("NumPy"):
        st.plotly_chart(plot_weight_heatmap(Ws[0], "Input → Hidden Layer 1 Weights"), use_container_width=True)

    with st.expander("📋 Classification report"):
        st.code(classification_report(y_te, preds))

    # Export Code Generation
    export_code = f"""import numpy as np

# Config: {sizes} hidden, {act_name}, lr={lr}, epochs={epochs}
def sigmoid(z): return 1/(1+np.exp(-z))
def relu(z): return np.maximum(0, z)

# Dims: {[in_d] + sizes + [out_d]}
# Load your model Ws and bs from the pickle file to run inference.
"""
    download_code("⬇ Export Python Snippet", export_code, "mlp_inference.py")
