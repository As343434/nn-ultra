"""Perceptron — single-neuron classifier with live decision boundary."""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data import load_moons, load_circles, load_blobs
from utils.export import download_code, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero
from utils.viz import plot_decision_boundary

st.set_page_config(page_title="Perceptron", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("Perceptron")

hero(
    "Perceptron",
    "Single-neuron linear classifier. Adjust weights and watch the boundary move in real time.",
    pill="Lesson 1", pill_variant="orange",
)

# ── Theory ────────────────────────────────────────────────────────────────────
with st.expander("📖 Theory", expanded=False):
    st.markdown("""
The perceptron computes a **linear score** and applies a **step function**:

$$\\hat{y} = \\mathbb{1}(\\mathbf{w}^\\top \\mathbf{x} + b \\geq 0)$$

Weight updates move the boundary toward misclassified points:

$$\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta (y - \\hat{y})\\,\\mathbf{x}, \\quad
  b \\leftarrow b + \\eta (y - \\hat{y})$$

**Key insight:** The perceptron converges *only* on linearly separable data.
For non-linearly separable data use moons / circles to see it struggle.
""")

st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])
with col1:
    dataset  = st.selectbox("Dataset", ["make_moons", "make_circles", "make_blobs"])
    n        = st.slider("Samples", 100, 800, 300, 50)
    noise    = st.slider("Noise", 0.0, 0.5, 0.2, 0.05)
    factor   = st.slider("Circle factor", 0.1, 0.9, 0.5, 0.05,
                         disabled=(dataset != "make_circles"))
    centers  = st.slider("Blob centers", 2, 5, 2, 1,
                         disabled=(dataset != "make_blobs"))

    st.markdown("---")
    w1   = st.slider("Weight w₁", -3.0, 3.0, 0.3, 0.05)
    w2   = st.slider("Weight w₂", -3.0, 3.0, -0.2, 0.05)
    bias = st.slider("Bias b",    -2.0,  2.0, 0.0, 0.05)
    lr   = st.slider("Learning rate η", 0.01, 1.0, 0.1, 0.01)
    epochs = st.slider("Epochs", 1, 100, 20, 1)

    upload = st.file_uploader("Or upload 2-column CSV", type=["csv"])

with col2:
    # Load data
    if dataset == "make_moons":
        X, y = load_moons(n, noise)
    elif dataset == "make_circles":
        X, y = load_circles(n, noise, factor)
    else:
        X, y = load_blobs(n, centers)
        y = (y > 0).astype(int)   # binarise

    if upload:
        df_up = pd.read_csv(upload)
        X, y = df_up.iloc[:, :2].values, df_up.iloc[:, 2].values.astype(int)

    w = np.array([w1, w2], dtype=float)

    st.pyplot(plot_decision_boundary(X, y, w, bias, "Live boundary"), use_container_width=True)

    scatter = px.scatter(x=X[:, 0], y=X[:, 1], color=y.astype(str),
                         color_discrete_sequence=["#00d4aa", "#f97316"],
                         template="plotly_dark",
                         labels={"x": "x₁", "y": "x₂", "color": "class"},
                         title="Point cloud (interactive)")
    scatter.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", height=260)
    st.plotly_chart(scatter, use_container_width=True)

# ── Training ──────────────────────────────────────────────────────────────────
st.markdown("### Train from Scratch")
train_btn = st.button("▶ Train Perceptron", type="primary")

if train_btn:
    w_t = np.array([w1, w2], dtype=float)
    b_t = bias
    history = []
    bar = st.progress(0)
    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            y_hat = 1 if np.dot(w_t, X[i]) + b_t >= 0 else 0
            delta = lr * (y[i] - y_hat)
            w_t += delta * X[i]
            b_t += delta
            errors += int(delta != 0)
            history.append({"epoch": epoch + 1, "errors": errors,
                             "w1": round(w_t[0], 4), "w2": round(w_t[1], 4),
                             "b": round(b_t, 4)})
        bar.progress(int((epoch + 1) / epochs * 100))
        time.sleep(0.02)

    st.success("Training complete!")
    st.pyplot(plot_decision_boundary(X, y, w_t, b_t, "Trained boundary"),
              use_container_width=True)

    with st.expander("📋 Weight update history (first 40 steps)"):
        st.dataframe(pd.DataFrame(history).head(40), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        download_pickle("⬇ Download model", {"w": w_t, "b": b_t}, "perceptron.pkl")
    with col_b:
        code = f"""\
import numpy as np

w = np.array({list(w_t)})
b = {b_t:.4f}
lr = {lr}

# Perceptron prediction
def predict(X):
    return (X @ w + b >= 0).astype(int)

# Training loop
for epoch in range({epochs}):
    for x, label in zip(X, y):
        y_hat = 1 if w @ x + b >= 0 else 0
        delta = lr * (label - y_hat)
        w += delta * x
        b += delta
"""
        download_code("⬇ Export Python", code, "perceptron.py")
