"""Forward Pass — step-by-step neuron calculator."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.export import download_code, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero
from utils.viz import plot_activation

st.set_page_config(page_title="Forward Pass", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("Forward Pass")

hero(
    "Forward Propagation",
    "Step-by-step single-layer calculator. Change inputs, weights, and activation live.",
    pill="Lesson 2", pill_variant="purple",
)

with st.expander("📖 Theory", expanded=False):
    st.markdown("""
The forward pass applies a **linear transformation** then a **nonlinearity**:

$$z = W\\mathbf{x} + \\mathbf{b}$$
$$\\mathbf{a} = \\sigma(z)$$

Popular activations and their derivatives:

| Activation | Formula | Derivative |
|---|---|---|
| Sigmoid | $\\frac{1}{1+e^{-z}}$ | $\\sigma(1-\\sigma)$ |
| Tanh | $\\tanh(z)$ | $1 - \\tanh^2(z)$ |
| ReLU | $\\max(0,z)$ | $\\mathbb{1}(z>0)$ |
| GELU | approx | smooth ReLU |
| Swish | $z \\cdot \\sigma(z)$ | $\\sigma + z\\sigma(1-\\sigma)$ |
""")

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    activation = st.selectbox("Activation function",
                              ["Sigmoid","ReLU","LeakyReLU","Tanh","GELU","Swish","Softplus","Softmax"])
    st.markdown("**Inputs**")
    x1 = st.number_input("x₁", value=1.0,  step=0.1)
    x2 = st.number_input("x₂", value=-0.5, step=0.1)
    st.markdown("**Weights  (2→2 layer)**")
    c1, c2 = st.columns(2)
    with c1:
        w11 = st.number_input("W₁₁", value=0.5,  step=0.1)
        w21 = st.number_input("W₂₁", value=0.3,  step=0.1)
    with c2:
        w12 = st.number_input("W₁₂", value=-0.4, step=0.1)
        w22 = st.number_input("W₂₂", value=0.2,  step=0.1)
    b1 = st.number_input("b₁", value=0.1,  step=0.1)
    b2 = st.number_input("b₂", value=-0.1, step=0.1)

with col2:
    x = np.array([x1, x2])
    W = np.array([[w11, w12], [w21, w22]])
    b = np.array([b1, b2])
    z = W @ x + b

    act_fns = {
        "Sigmoid":   lambda z: 1 / (1 + np.exp(-z)),
        "ReLU":      lambda z: np.maximum(0, z),
        "LeakyReLU": lambda z: np.where(z >= 0, z, 0.01 * z),
        "Tanh":      np.tanh,
        "Softplus":  lambda z: np.log1p(np.exp(z)),
        "GELU":      lambda z: 0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3))),
        "Swish":     lambda z: z / (1 + np.exp(-z)),
        "Softmax":   lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum(),
    }
    a = act_fns[activation](z)

    st.markdown("#### Computation trace")
    st.code(f"x  = {x}\nW  =\n{W}\nb  = {b}\nz  = W @ x + b = {np.round(z,4)}\na  = {activation}(z) = {np.round(a,4)}", language="python")

    st.plotly_chart(plot_activation(activation), use_container_width=True)

    # Derivative at z
    derivs = {
        "Sigmoid":   lambda z, a: a * (1 - a),
        "ReLU":      lambda z, a: (z > 0).astype(float),
        "LeakyReLU": lambda z, a: np.where(z >= 0, 1.0, 0.01),
        "Tanh":      lambda z, a: 1 - a**2,
        "Softplus":  lambda z, a: 1 / (1 + np.exp(-z)),
        "GELU":      lambda z, a: a / z if np.all(z != 0) else np.zeros_like(z),
        "Swish":     lambda z, a: a + (1/(1+np.exp(-z))) * (1 - a),
        "Softmax":   lambda z, a: a * (1 - a),
    }
    da = derivs[activation](z, a)
    st.info(f"σ′(z) at current z = **{np.round(da, 4)}**")

    download_pickle("⬇ Save state", {"x": x, "W": W, "b": b, "z": z, "a": a}, "forward_state.pkl")

code = f"""\
import numpy as np

x = np.array([{x1}, {x2}])
W = np.array([[{w11}, {w12}], [{w21}, {w22}]])
b = np.array([{b1}, {b2}])

z = W @ x + b   # linear step
# {activation} activation
"""
download_code("⬇ Export Python", code, "forward_pass.py")
