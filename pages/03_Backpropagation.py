"""Backpropagation — chain-rule gradient visualizer (self-contained)."""
import io
import pickle

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Backpropagation", layout="wide", page_icon="↺")

# ── Theme injection ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;700;800&display=swap');

:root {
  --accent:  #00d4aa;
  --accent2: #f97316;
  --accent3: #a78bfa;
  --surface: #0f1117;
  --surface2: #1a1d27;
  --border:  #2a2d3a;
  --text:    #e8eaf0;
  --muted:   #6b7280;
  --success: #22c55e;
  --radius:  10px;
  --radius-lg: 14px;
  --card-shadow: 0 2px 12px rgba(0,0,0,0.4);
  --glow: 0 0 16px rgba(0,212,170,0.25);
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
    rgba(0,212,170,0.12) 0%, rgba(167,139,250,0.07) 40%, transparent 70%),
    var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2.2rem 2rem;
  margin-bottom: 1.6rem;
}

.nn-pill {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  background: rgba(0,212,170,0.15);
  color: var(--accent);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}

.nn-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.4rem;
  margin-bottom: 1rem;
  box-shadow: var(--card-shadow);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.grad-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.45rem 0.8rem;
  border-radius: 8px;
  background: var(--surface2);
  border: 1px solid var(--border);
  margin-bottom: 0.4rem;
  font-family: 'IBM Plex Mono', monospace;
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
}

div[data-testid="stExpander"] > details {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

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
        style = "color: var(--accent); font-weight: 700;" if name == "Backpropagation" else ""
        st.markdown(f"<div style='{style}'>{icon} {name}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with ❤️ · NeuralForge Ultra")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nn-hero">
  <div class="nn-pill">Lesson 3</div>
  <h1 style="margin:0 0 0.4rem 0; font-size:2rem;">↺ Backpropagation</h1>
  <p style="color:#9ca3af; margin:0; font-size:1rem;">
    Visualize the chain rule step by step. See how gradients flow backwards through a neuron.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Theory ────────────────────────────────────────────────────────────────────
with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
Backprop uses the **chain rule** to compute gradients efficiently:

$$\frac{\partial L}{\partial w} =
  \underbrace{\frac{\partial L}{\partial a}}_{\delta_a}
  \cdot \underbrace{\frac{\partial a}{\partial z}}_{\sigma'}
  \cdot \underbrace{\frac{\partial z}{\partial w}}_{x}$$

**MSE loss:**  $L = \frac{1}{2}(a - y)^2$, so $\delta_a = a - y$

The same formula applies to every parameter — backprop just reuses
intermediate products rather than recomputing them.
""")

st.markdown("---")

# ── Controls + Computation ────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### ⚙️ Parameters")
    act      = st.selectbox("Activation", ["Sigmoid", "Tanh", "ReLU"])
    loss_fn  = st.selectbox("Loss",       ["MSE", "Binary Cross-Entropy"])
    x        = st.number_input("Input x",   value=0.80, step=0.05, format="%.2f")
    w        = st.number_input("Weight w",  value=0.50, step=0.05, format="%.2f")
    b        = st.number_input("Bias b",    value=0.10, step=0.05, format="%.2f")
    y_target = st.number_input("Target y",  value=1.00, step=0.05, format="%.2f")

# ── Math ──────────────────────────────────────────────────────────────────────
z = w * x + b

if act == "Sigmoid":
    a      = 1 / (1 + np.exp(-z))
    da_dz  = float(a * (1 - a))
elif act == "Tanh":
    a      = np.tanh(z)
    da_dz  = float(1 - a**2)
else:  # ReLU
    a      = float(max(0.0, z))
    da_dz  = 1.0 if z > 0 else 0.0

a = float(a)

if loss_fn == "MSE":
    loss   = 0.5 * (a - y_target)**2
    dL_da  = a - y_target
else:  # BCE
    a_c    = float(np.clip(a, 1e-7, 1 - 1e-7))
    loss   = -(y_target * np.log(a_c) + (1 - y_target) * np.log(1 - a_c))
    dL_da  = -y_target / a_c + (1 - y_target) / (1 - a_c)

dL_dz = float(dL_da) * da_dz
dL_dw = dL_dz * x
dL_db = dL_dz

# ── Right column output ───────────────────────────────────────────────────────
with col2:
    st.markdown("#### Forward pass")
    st.json({"z": round(z, 5), "a": round(a, 5), "loss": round(float(loss), 5)})

    if st.button("🔄 Compute gradients", type="primary"):

        # Chain-rule breakdown
        st.markdown("#### Chain-rule breakdown")
        steps = {
            "dL/da":      float(dL_da),
            "da/dz (σ′)": float(da_dz),
            "dL/dz":      float(dL_dz),
            "dL/dw":      float(dL_dw),
            "dL/db":      float(dL_db),
        }
        for k, v in steps.items():
            color = "#00d4aa" if v >= 0 else "#f97316"
            st.markdown(
                f"""<div class="grad-row">
                      <span style="color:#9ca3af">{k}</span>
                      <span style="color:{color};font-weight:700">{v:.5f}</span>
                    </div>""",
                unsafe_allow_html=True,
            )

        # Gradient bar chart
        st.markdown("#### Gradient magnitudes")
        labels = ["dL/dw", "dL/db"]
        values = [float(dL_dw), float(dL_db)]
        colors = ["#00d4aa" if v >= 0 else "#f97316" for v in values]

        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v:.5f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0", family="IBM Plex Sans"),
            yaxis=dict(gridcolor="#2a2d3a", zerolinecolor="#2a2d3a"),
            xaxis=dict(gridcolor="#2a2d3a"),
            margin=dict(l=10, r=10, t=20, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

        # SGD update preview
        updated_w = w - 0.1 * dL_dw
        updated_b = b - 0.1 * dL_db
        st.markdown(f"""
        > **After one SGD step (lr = 0.1):**  
        > w → `{updated_w:.5f}` &nbsp;&nbsp; b → `{updated_b:.5f}`
        """)

        # Computation graph
        st.markdown("#### Computation graph")
        st.markdown(f"""
        ```
        x={x:.2f}  w={w:.2f}
             ↘   ↙
          z = wx + b = {z:.4f}  (b={b:.2f})
               ↓
          a = {act}(z) = {a:.4f}
               ↓
          L = {loss_fn}(a, y={y_target}) = {float(loss):.4f}
        ```
        """)

# ── Exports ───────────────────────────────────────────────────────────────────
st.markdown("---")
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    state = {
        "x": x, "w": w, "b": b, "y": y_target,
        "z": float(z), "a": a,
        "dL_dw": float(dL_dw), "dL_db": float(dL_db),
    }
    buf = io.BytesIO()
    pickle.dump(state, buf)
    st.download_button(
        "⬇ Save gradient state",
        data=buf.getvalue(),
        file_name="backprop_state.pkl",
        mime="application/octet-stream",
    )

with exp_col2:
    code = f"""\
import numpy as np

x, w, b, y = {x}, {w}, {b}, {y_target}

z     = w * x + b
a     = 1 / (1 + np.exp(-z))   # sigmoid
loss  = 0.5 * (a - y)**2       # MSE

dL_da = a - y
da_dz = a * (1 - a)            # sigmoid derivative
dL_dz = dL_da * da_dz
dL_dw = dL_dz * x
dL_db = dL_dz

print(f"z={{z:.5f}}, a={{a:.5f}}, loss={{loss:.5f}}")
print(f"dL/dw={{dL_dw:.5f}}, dL/db={{dL_db:.5f}}")

# SGD update (lr=0.1)
w_new = w - 0.1 * dL_dw
b_new = b - 0.1 * dL_db
print(f"Updated: w={{w_new:.5f}}, b={{b_new:.5f}}")
"""
    st.download_button(
        "⬇ Export Python",
        data=code,
        file_name="backprop.py",
        mime="text/x-python",
    )
