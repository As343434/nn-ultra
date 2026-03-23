"""Gradient Descent — optimizer playground with 3D surface and contour path."""
import time

import numpy as np
import streamlit as st

from utils.export import download_code, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero
from utils.viz import plot_contour_path, plot_3d_surface

st.set_page_config(page_title="Gradient Descent", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("Gradient Descent")

hero(
    "Gradient Descent",
    "Compare GD, SGD, Momentum, and Adam on a 3-D loss surface. Watch paths diverge.",
    pill="Lesson 4", pill_variant="orange",
)

with st.expander("📖 Theory", expanded=False):
    st.markdown("""
We minimize a loss surface by stepping opposite to the gradient:

| Optimizer | Update Rule |
|---|---|
| GD | $\\theta \\leftarrow \\theta - \\eta\\,\\nabla L$ |
| SGD | same but on a random mini-batch |
| Momentum | $v \\leftarrow \\beta v + (1-\\beta)\\nabla L$;  $\\theta \\leftarrow \\theta - \\eta v$ |
| Adam | bias-corrected adaptive moment estimation |

**Rule of thumb:** Adam converges faster; SGD with momentum often generalizes better.
""")

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    optimizer = st.selectbox("Optimizer", ["GD", "SGD", "Momentum", "Adam"])
    lr        = st.slider("Learning rate", 0.001, 0.5, 0.05, 0.001, format="%.3f")
    steps     = st.slider("Steps", 10, 200, 60, 5)
    init_x    = st.slider("Start x", -3.0, 3.0, 2.5, 0.1)
    init_y    = st.slider("Start y", -3.0, 3.0,-2.0, 0.1)
    beta1     = st.slider("β₁ (Momentum)", 0.5, 0.99, 0.9, 0.01)
    beta2     = st.slider("β₂ (Adam v)", 0.9, 0.999, 0.999, 0.001)

with col2:
    st.plotly_chart(plot_3d_surface(), use_container_width=True)

if st.button("▶ Animate", type="primary"):
    x, y = float(init_x), float(init_y)
    xs, ys = [x], [y]
    vx = vy = mx = my = vvx = vvy = 0.0

    bar = st.progress(0)
    for t in range(1, steps + 1):
        gx, gy = 2 * x, 2 * y
        if optimizer == "GD":
            x -= lr * gx; y -= lr * gy
        elif optimizer == "SGD":
            nx, ny = np.random.normal(scale=0.15, size=2)
            x -= lr * (gx + nx); y -= lr * (gy + ny)
        elif optimizer == "Momentum":
            vx = beta1 * vx + (1 - beta1) * gx
            vy = beta1 * vy + (1 - beta1) * gy
            x -= lr * vx; y -= lr * vy
        else:  # Adam
            mx  = beta1  * mx  + (1 - beta1)  * gx
            my  = beta1  * my  + (1 - beta1)  * gy
            vvx = beta2  * vvx + (1 - beta2)  * gx**2
            vvy = beta2  * vvy + (1 - beta2)  * gy**2
            mxh = mx  / (1 - beta1**t)
            myh = my  / (1 - beta1**t)
            vxh = vvx / (1 - beta2**t)
            vyh = vvy / (1 - beta2**t)
            x  -= lr * mxh / (vxh**0.5 + 1e-8)
            y  -= lr * myh / (vyh**0.5 + 1e-8)

        xs.append(x); ys.append(y)
        bar.progress(int(t / steps * 100))
        time.sleep(0.01)

    final_loss = x**2 + y**2
    st.success(f"Final loss = **{final_loss:.5f}** at ({x:.4f}, {y:.4f})")
    st.plotly_chart(plot_contour_path(xs, ys), use_container_width=True)

    download_pickle("⬇ Download path", {"xs": xs, "ys": ys}, "optimizer_path.pkl")

    code = f"""\
import numpy as np

x, y = {init_x}, {init_y}
lr = {lr}
vx = vy = 0.0

for _ in range({steps}):
    gx, gy = 2 * x, 2 * y
    # {optimizer}
    vx = {beta1} * vx + {1 - beta1:.3f} * gx
    vy = {beta1} * vy + {1 - beta1:.3f} * gy
    x -= lr * vx
    y -= lr * vy

print(f"min ≈ ({{x:.4f}}, {{y:.4f}}), loss={{x**2+y**2:.5f}}")
"""
    download_code("⬇ Export Python", code, "gradient_descent.py")
