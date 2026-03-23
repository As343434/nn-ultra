"""Backpropagation — chain-rule gradient visualizer."""
import numpy as np
import plotly.express as px
import streamlit as st

from utils.export import download_code, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero
from utils.viz import plot_gradients

st.set_page_config(page_title="Backpropagation", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("Backpropagation")

hero(
    "Backpropagation",
    "Visualize the chain rule step by step. See how gradients flow backwards through a neuron.",
    pill="Lesson 3", pill_variant="",
)

with st.expander("📖 Theory", expanded=False):
    st.markdown("""
Backprop uses the **chain rule** to compute gradients efficiently:

$$\\frac{\\partial L}{\\partial w} =
  \\underbrace{\\frac{\\partial L}{\\partial a}}_{\\delta_a}
  \\cdot \\underbrace{\\frac{\\partial a}{\\partial z}}_{\\sigma'}
  \\cdot \\underbrace{\\frac{\\partial z}{\\partial w}}_{x}$$

**MSE loss:**  $L = \\frac{1}{2}(a - y)^2$, so $\\delta_a = a - y$

The same formula applies to every parameter — backprop just reuses
intermediate products rather than recomputing them.
""")

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    act = st.selectbox("Activation", ["Sigmoid", "Tanh", "ReLU"])
    loss_fn = st.selectbox("Loss", ["MSE", "Binary Cross-Entropy"])
    x = st.number_input("Input x",    value=0.8, step=0.05)
    w = st.number_input("Weight w",   value=0.5, step=0.05)
    b = st.number_input("Bias b",     value=0.1, step=0.05)
    y = st.number_input("Target y",   value=1.0, step=0.05)

with col2:
    z = w * x + b
    if act == "Sigmoid":
        a = 1 / (1 + np.exp(-z))
        da_dz = a * (1 - a)
    elif act == "Tanh":
        a = np.tanh(z)
        da_dz = 1 - a**2
    else:
        a = max(0.0, z)
        da_dz = 1.0 if z > 0 else 0.0

    if loss_fn == "MSE":
        loss  = 0.5 * (a - y)**2
        dL_da = a - y
    else:  # BCE
        a_c   = float(np.clip(a, 1e-7, 1 - 1e-7))
        loss  = -(y * np.log(a_c) + (1 - y) * np.log(1 - a_c))
        dL_da = -y / a_c + (1 - y) / (1 - a_c)

    dL_dz = dL_da * da_dz
    dL_dw = dL_dz * x
    dL_db = dL_dz

    # Summary table
    st.markdown("#### Forward pass")
    st.json({"z": round(z, 5), "a": round(float(a), 5), "loss": round(float(loss), 5)})

    # Chain-rule breakdown
    if st.button("🔄 Compute gradients", type="primary"):
        steps = {
            "dL/da": float(dL_da),
            "da/dz (σ′)": float(da_dz),
            "dL/dz": float(dL_dz),
            "dL/dw": float(dL_dw),
            "dL/db": float(dL_db),
        }
        st.markdown("#### Chain-rule breakdown")
        for k, v in steps.items():
            col_k, col_v = st.columns([2, 1])
            col_k.markdown(f"`{k}`")
            color = "#00d4aa" if v >= 0 else "#f97316"
            col_v.markdown(f"<span style='color:{color};font-weight:700'>{v:.5f}</span>",
                           unsafe_allow_html=True)

        st.plotly_chart(
            plot_gradients(["dL/dw", "dL/db"], [float(dL_dw), float(dL_db)]),
            use_container_width=True,
        )

        updated_w = w - 0.1 * dL_dw
        updated_b = b - 0.1 * dL_db
        st.markdown(f"""
        > **After one SGD step (lr=0.1):**  
        > w → `{updated_w:.5f}`,  b → `{updated_b:.5f}`
        """)

    download_pickle("⬇ Save gradient state",
                    {"x": x, "w": w, "b": b, "y": y, "z": z,
                     "a": float(a), "dL_dw": float(dL_dw), "dL_db": float(dL_db)},
                    "backprop_state.pkl")

code = f"""\
import numpy as np

x, w, b, y = {x}, {w}, {b}, {y}
z = w * x + b
a = 1 / (1 + np.exp(-z))        # sigmoid

loss  = 0.5 * (a - y)**2        # MSE
dL_da = a - y
da_dz = a * (1 - a)             # sigmoid derivative
dL_dw = dL_da * da_dz * x
dL_db = dL_da * da_dz
"""
download_code("⬇ Export Python", code, "backprop.py")
