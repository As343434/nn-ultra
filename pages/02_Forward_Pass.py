"""Forward Pass — step-by-step neuron calculator (self-contained)."""
import io
import pickle

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Forward Pass", layout="wide", page_icon="⟶")

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
  --accent:     #00d4aa;
  --accent2:    #f97316;
  --accent3:    #a78bfa;
  --surface:    #0f1117;
  --surface2:   #1a1d27;
  --border:     #2a2d3a;
  --text:       #e8eaf0;
  --muted:      #6b7280;
  --success:    #22c55e;
  --radius:     10px;
  --radius-lg:  14px;
  --card-shadow:0 2px 12px rgba(0,0,0,0.4);
  --glow:       0 0 16px rgba(0,212,170,0.25);
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
  background: #080a10 !important;
  color: var(--text) !important;
}

h1, h2, h3, h4 {
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
}

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}

.nn-hero {
  background: radial-gradient(ellipse 80% 60% at 50% 0%,
    rgba(167,139,250,0.12) 0%, rgba(0,212,170,0.07) 40%, transparent 70%),
    var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2.2rem 2rem;
  margin-bottom: 1.6rem;
}

.nn-pill-purple {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  background: rgba(167,139,250,0.15);
  color: var(--accent3);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}

.trace-box {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
  color: var(--accent);
  white-space: pre;
  overflow-x: auto;
  margin-bottom: 1rem;
}

.deriv-box {
  background: var(--surface2);
  border: 1px solid rgba(0,212,170,0.3);
  border-radius: var(--radius);
  padding: 0.7rem 1rem;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 1rem;
}

[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.8rem 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; }

button[kind="primary"] {
  background: var(--accent) !important;
  color: #000 !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
  box-shadow: 0 0 12px rgba(0,212,170,0.4) !important;
}
button[kind="secondary"] {
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  border-radius: 8px !important;
}

div[data-testid="stExpander"] > details {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

[data-testid="stProgress"] > div > div { background: var(--accent) !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NeuralForge Ultra")
    st.markdown("**v3.0 · 16 MODULES**")
    st.markdown("---")
    pages = [
        ("🏠", "Home"), ("⬡", "Perceptron"), ("⟶", "Forward Pass"),
        ("↺", "Backpropagation"), ("↗", "Gradient Descent"), ("⬛", "ANN / MLP"),
        ("◫", "CNN"), ("⇌", "RNN / LSTM"), ("◎", "Autoencoder / VAE"),
        ("◉", "OpenCV Vision"), ("⚡", "Transformer Attn"), ("🎮", "GAN Lab"),
        ("🤖", "RL Agent"), ("🧬", "NAS Explorer"), ("📊", "Model Comparison"),
        ("🧠", "AI Explainer"),
    ]
    for icon, name in pages:
        style = "color: var(--accent); font-weight: 700;" if name == "Forward Pass" else ""
        st.markdown(f"<div style='{style}'>{icon} {name}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with ❤️ · NeuralForge Ultra")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nn-hero">
  <div class="nn-pill-purple">Lesson 2</div>
  <h1 style="margin:0 0 0.4rem 0; font-size:2rem;">⟶ Forward Propagation</h1>
  <p style="color:#9ca3af; margin:0; font-size:1rem;">
    Step-by-step single-layer calculator. Change inputs, weights, and activation live.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Theory ────────────────────────────────────────────────────────────────────
with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
The forward pass applies a **linear transformation** then a **nonlinearity**:

$$z = W\mathbf{x} + \mathbf{b}$$
$$\mathbf{a} = \sigma(z)$$

Popular activations and their derivatives:

| Activation | Formula | Derivative |
|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $\sigma(1-\sigma)$ |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ |
| ReLU | $\max(0,z)$ | $\mathbb{1}(z>0)$ |
| GELU | approx | smooth ReLU |
| Swish | $z \cdot \sigma(z)$ | $\sigma + z\sigma(1-\sigma)$ |
""")

st.markdown("---")

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### ⚙️ Parameters")
    activation = st.selectbox("Activation function",
                              ["Sigmoid", "ReLU", "LeakyReLU", "Tanh",
                               "GELU", "Swish", "Softplus", "Softmax"])
    st.markdown("**Inputs**")
    x1 = st.number_input("x₁", value=1.0,  step=0.1, format="%.2f")
    x2 = st.number_input("x₂", value=-0.5, step=0.1, format="%.2f")

    st.markdown("**Weights (2→2 layer)**")
    c1, c2 = st.columns(2)
    with c1:
        w11 = st.number_input("W₁₁", value=0.5,  step=0.1, format="%.2f")
        w21 = st.number_input("W₂₁", value=0.3,  step=0.1, format="%.2f")
    with c2:
        w12 = st.number_input("W₁₂", value=-0.4, step=0.1, format="%.2f")
        w22 = st.number_input("W₂₂", value=0.2,  step=0.1, format="%.2f")

    b1 = st.number_input("b₁", value=0.1,  step=0.1, format="%.2f")
    b2 = st.number_input("b₂", value=-0.1, step=0.1, format="%.2f")

# ── Math ──────────────────────────────────────────────────────────────────────
sigmoid_fn = lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500)))

act_fns = {
    "Sigmoid":   sigmoid_fn,
    "ReLU":      lambda z: np.maximum(0, z),
    "LeakyReLU": lambda z: np.where(z >= 0, z, 0.01 * z),
    "Tanh":      np.tanh,
    "Softplus":  lambda z: np.log1p(np.exp(np.clip(z, -500, 80))),
    "GELU":      lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))),
    "Swish":     lambda z: z * sigmoid_fn(z),
    "Softmax":   lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum(),
}

deriv_fns = {
    "Sigmoid":   lambda z, a: a * (1 - a),
    "ReLU":      lambda z, a: (z > 0).astype(float),
    "LeakyReLU": lambda z, a: np.where(z >= 0, 1.0, 0.01),
    "Tanh":      lambda z, a: 1 - a**2,
    "Softplus":  lambda z, a: sigmoid_fn(z),
    "GELU":      lambda z, a: (
        0.5 * (1 + np.tanh(0.797885 * (z + 0.044715 * z**3))) +
        0.5 * z * (1 - np.tanh(0.797885 * (z + 0.044715 * z**3))**2) *
        (0.797885 * (1 + 3 * 0.044715 * z**2))
    ),
    "Swish":     lambda z, a: a + sigmoid_fn(z) * (1 - a),
    "Softmax":   lambda z, a: a * (1 - a),
}

x = np.array([x1, x2])
W = np.array([[w11, w12], [w21, w22]])
b = np.array([b1, b2])
z = W @ x + b
a = act_fns[activation](z)
da = deriv_fns[activation](z, a)

# ── Right column ──────────────────────────────────────────────────────────────
with col2:
    st.markdown("#### Computation trace")
    trace = (
        f"x  = {np.round(x, 4).tolist()}\n"
        f"W  =\n     {np.round(W[0], 4).tolist()}\n"
        f"     {np.round(W[1], 4).tolist()}\n"
        f"b  = {np.round(b, 4).tolist()}\n"
        f"z  = W @ x + b = {np.round(z, 4).tolist()}\n"
        f"a  = {activation}(z) = {np.round(a, 4).tolist()}"
    )
    st.markdown(f'<div class="trace-box">{trace}</div>', unsafe_allow_html=True)

    # Derivative info
    st.markdown(
        f'<div class="deriv-box">σ′(z) at current z = <code>{np.round(da, 4).tolist()}</code></div>',
        unsafe_allow_html=True,
    )

    # Activation curve plot
    st.markdown("#### Activation curve")
    z_range = np.linspace(-5, 5, 300)
    a_range = act_fns[activation](z_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z_range, y=a_range,
        mode="lines", name=activation,
        line=dict(color="#00d4aa", width=2.5),
    ))
    # Mark current z values
    for i, (zi, ai) in enumerate(zip(z, a)):
        fig.add_trace(go.Scatter(
            x=[zi], y=[ai],
            mode="markers",
            marker=dict(color="#f97316", size=10, symbol="circle"),
            name=f"z_{i+1}={zi:.3f}",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(title="z", gridcolor="#2a2d3a", zerolinecolor="#4b5563"),
        yaxis=dict(title="a = σ(z)", gridcolor="#2a2d3a", zerolinecolor="#4b5563"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Output metrics
    m1, m2 = st.columns(2)
    m1.metric("Neuron 1 output", f"{float(a[0]):.4f}")
    m2.metric("Neuron 2 output", f"{float(a[1]):.4f}")

# ── Exports ───────────────────────────────────────────────────────────────────
st.markdown("---")
exp1, exp2 = st.columns(2)

with exp1:
    state = {"x": x.tolist(), "W": W.tolist(), "b": b.tolist(),
             "z": z.tolist(), "a": a.tolist()}
    buf = io.BytesIO()
    pickle.dump(state, buf)
    st.download_button(
        "⬇ Save state",
        data=buf.getvalue(),
        file_name="forward_state.pkl",
        mime="application/octet-stream",
    )

with exp2:
    python_export = f"""\
import numpy as np

x  = np.array([{x1}, {x2}])
W  = np.array([[{w11}, {w12}],
               [{w21}, {w22}]])
b  = np.array([{b1}, {b2}])

z  = W @ x + b          # linear step
# activation: {activation}
# a = {np.round(a, 4).tolist()}

print("z =", z)
print("a =", a)
"""
    st.download_button(
        "⬇ Export Python",
        data=python_export,
        file_name="forward_pass.py",
        mime="text/x-python",
    )
