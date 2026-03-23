import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Perceptron",
    layout="wide",
    page_icon="⬡"
)

# ====================== SAFE THEME CSS ======================
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
    .nn-card, .stPlotlyChart, .element-container {
        border-radius: 12px !important;
    }
    button[kind="primary"] {
        background: var(--accent) !important;
        color: #000 !important;
        font-weight: 700 !important;
    }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ====================== HELPERS ======================
def plot_decision_boundary(X, y, w, b, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = (Z >= 0).astype(int).reshape(xx.shape)

    fig = px.imshow(Z, x=np.linspace(x_min, x_max, 300), y=np.linspace(y_min, y_max, 300),
                    color_continuous_scale=["#f97316", "#00d4aa"], origin="lower",
                    labels={"x": "x₁", "y": "x₂"}, title=title)
    
    # Overlay scatter points
    fig.add_scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                    marker=dict(color=y, colorscale=["#f97316", "#00d4aa"], size=8, line=dict(width=1, color="white")),
                    name="Data Points")
    
    fig.update_layout(
        height=520,
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("⬡ Perceptron")
    st.markdown("Single-neuron classifier")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== MAIN UI ======================
st.title("Perceptron")
st.markdown("**Single-neuron linear classifier.** Adjust weights and watch the boundary move in real time.")

# Hero-like banner
st.markdown("""
<div style="background: linear-gradient(90deg, #1a1f2e, #0f1626); padding: 1.8rem; border-radius: 12px; border: 1px solid #00d4aa33; margin-bottom: 1.5rem;">
    <h2 style="margin:0; color:#00d4aa;">Lesson 1: The Perceptron</h2>
    <p style="color:#8b949e; margin-top:0.5rem;">The foundation of all neural networks</p>
</div>
""", unsafe_allow_html=True)

# Theory
with st.expander("📖 Theory", expanded=True):
    st.markdown("""
The perceptron computes a **linear score** and applies a **step function**:

$$\\hat{y} = \\mathbb{1}(\\mathbf{w}^\\top \\mathbf{x} + b \\geq 0)$$

**Weight update rule** (when misclassified):
$$\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta (y - \\hat{y})\\,\\mathbf{x}, \\quad b \\leftarrow b + \\eta (y - \\hat{y})$$

**Important:** The perceptron can only learn **linearly separable** problems.
""")

st.markdown("---")

# Controls + Visualization
col1, col2 = st.columns([1, 2])

with col1:
    dataset = st.selectbox("Dataset", ["make_moons", "make_circles", "make_blobs"])
    n_samples = st.slider("Number of samples", 100, 800, 300, 50)
    noise = st.slider("Noise level", 0.0, 0.5, 0.2, 0.05)

    if dataset == "make_circles":
        factor = st.slider("Circle factor", 0.1, 0.9, 0.5, 0.05)
    else:
        factor = 0.5

    if dataset == "make_blobs":
        centers = st.slider("Number of centers", 2, 5, 2, 1)
    else:
        centers = 2

    st.markdown("---")
    st.subheader("Manual Weights")
    w1 = st.slider("Weight w₁", -3.0, 3.0, 0.3, 0.05)
    w2 = st.slider("Weight w₂", -3.0, 3.0, -0.2, 0.05)
    bias = st.slider("Bias b", -2.0, 2.0, 0.0, 0.05)

    st.markdown("---")
    st.subheader("Training")
    lr = st.slider("Learning rate η", 0.01, 1.0, 0.1, 0.01)
    epochs = st.slider("Epochs", 1, 100, 20, 1)

    uploaded_file = st.file_uploader("Upload your own 2D CSV (columns: x1, x2, label)", type=["csv"])

with col2:
    # Generate or load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] < 3:
            st.error("CSV must have at least 3 columns: x1, x2, label")
            X, y = np.random.rand(100, 2), np.random.randint(0, 2, 100)
        else:
            X = df.iloc[:, :2].values
            y = df.iloc[:, 2].astype(int).values
    else:
        if dataset == "make_moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        elif dataset == "make_circles":
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
            y = y.astype(int)
        else:
            X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)
            y = (y > 0).astype(int)

    w = np.array([w1, w2])

    # Live boundary plot
    fig_boundary = plot_decision_boundary(X, y, w, bias, "Live Decision Boundary")
    st.plotly_chart(fig_boundary, use_container_width=True)

    # Data scatter
    scatter = px.scatter(
        x=X[:, 0], y=X[:, 1], color=y.astype(str),
        color_discrete_sequence=["#f97316", "#00d4aa"],
        template="plotly_dark",
        labels={"x": "Feature x₁", "y": "Feature x₂"},
        title="Data Points"
    )
    scatter.update_layout(height=280, paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
    st.plotly_chart(scatter, use_container_width=True)

# ====================== TRAINING SECTION ======================
st.markdown("---")
st.subheader("Train Perceptron from Scratch")

if st.button("▶ Train Perceptron", type="primary"):
    w_t = np.array([w1, w2], dtype=float)
    b_t = bias
    history = []

    bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            y_hat = 1 if np.dot(w_t, X[i]) + b_t >= 0 else 0
            delta = lr * (y[i] - y_hat)
            if delta != 0:
                w_t += delta * X[i]
                b_t += delta
                errors += 1

        history.append({
            "epoch": epoch + 1,
            "errors": errors,
            "w1": round(w_t[0], 4),
            "w2": round(w_t[1], 4),
            "b": round(b_t, 4)
        })

        bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} — Errors: {errors}")

    st.success("✅ Training Complete!")

    # Final boundary
    final_fig = plot_decision_boundary(X, y, w_t, b_t, "Trained Decision Boundary")
    st.plotly_chart(final_fig, use_container_width=True)

    # History
    with st.expander("📋 Training History (Weight Updates)"):
        st.dataframe(pd.DataFrame(history), use_container_width=True)

    # Export options
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            label="⬇ Download Model (Pickle)",
            data=pd.to_pickle({"w": w_t, "b": b_t}),
            file_name="perceptron_model.pkl",
            mime="application/octet-stream"
        )
    with col_b:
        code_str = f"""import numpy as np

w = np.array([{w_t[0]:.4f}, {w_t[1]:.4f}])
b = {b_t:.4f}
lr = {lr}

def predict(X):
    return (X @ w + b >= 0).astype(int)

# Training loop example
for epoch in range({epochs}):
    for x, label in zip(X, y):
        y_hat = 1 if (w @ x + b >= 0) else 0
        delta = lr * (label - y_hat)
        w += delta * x
        b += delta
"""
        st.download_button(
            label="⬇ Export Training Code",
            data=code_str,
            file_name="perceptron.py",
            mime="text/plain"
        )
