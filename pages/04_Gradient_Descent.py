"""Gradient Descent — optimizer playground with 3D surface and contour path (self-contained)."""
import io
import pickle
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gradient Descent", layout="wide", page_icon="↗")

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
    rgba(249,115,22,0.12) 0%, rgba(0,212,170,0.07) 40%, transparent 70%),
    var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2.2rem 2rem;
  margin-bottom: 1.6rem;
}

.nn-pill-orange {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  background: rgba(249,115,22,0.15);
  color: var(--accent2);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}

[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.8rem 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; }

button[kind="primary"] {
  background: var(--accent2) !important;
  color: #000 !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
  box-shadow: 0 0 12px rgba(249,115,22,0.4) !important;
}

div[data-testid="stExpander"] > details {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

[data-testid="stProgress"] > div > div { background: var(--accent2) !important; }
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
        style = "color: var(--accent2); font-weight: 700;" if name == "Gradient Descent" else ""
        st.markdown(f"<div style='{style}'>{icon} {name}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with ❤️ · NeuralForge Ultra")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nn-hero">
  <div class="nn-pill-orange">Lesson 4</div>
  <h1 style="margin:0 0 0.4rem 0; font-size:2rem;">↗ Gradient Descent</h1>
  <p style="color:#9ca3af; margin:0; font-size:1rem;">
    Compare GD, SGD, Momentum, and Adam on a 3-D loss surface. Watch paths diverge.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Theory ────────────────────────────────────────────────────────────────────
with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
We minimize a loss surface by stepping opposite to the gradient:

| Optimizer | Update Rule |
|---|---|
| GD | $\theta \leftarrow \theta - \eta\,\nabla L$ |
| SGD | same but on a random mini-batch |
| Momentum | $v \leftarrow \beta v + (1-\beta)\nabla L$; $\theta \leftarrow \theta - \eta v$ |
| Adam | bias-corrected adaptive moment estimation |

**Rule of thumb:** Adam converges faster; SGD with momentum often generalizes better.
""")

st.markdown("---")

# ── Viz helpers ───────────────────────────────────────────────────────────────
def make_3d_surface():
    grid = np.linspace(-3, 3, 80)
    X, Y = np.meshgrid(grid, grid)
    Z = X**2 + Y**2          # simple bowl — easy to swap for Rosenbrock etc.
    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Teal",
        opacity=0.82,
        showscale=False,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#2a2d3a", color="#6b7280"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#2a2d3a", color="#6b7280"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#2a2d3a", color="#6b7280"),
        ),
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        margin=dict(l=0, r=0, t=20, b=0),
        height=420,
        title=dict(text="Loss surface: L = x² + y²", font=dict(size=13, color="#9ca3af")),
    )
    return fig


def make_contour_path(xs, ys, label="path"):
    grid = np.linspace(-3.2, 3.2, 200)
    X, Y = np.meshgrid(grid, grid)
    Z = X**2 + Y**2
    losses = [xi**2 + yi**2 for xi, yi in zip(xs, ys)]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=grid, y=grid, z=Z,
        colorscale="Teal",
        showscale=False,
        contours=dict(coloring="heatmap", showlabels=False),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers",
        line=dict(color="#f97316", width=2),
        marker=dict(color="#f97316", size=5),
        name=label,
    ))
    # Start / end markers
    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]],
        mode="markers",
        marker=dict(color="#a78bfa", size=12, symbol="star"),
        name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]],
        mode="markers",
        marker=dict(color="#00d4aa", size=12, symbol="circle"),
        name="End",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,17,23,1)",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(title="x", gridcolor="#2a2d3a", zerolinecolor="#4b5563"),
        yaxis=dict(title="y", gridcolor="#2a2d3a", zerolinecolor="#4b5563"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=400,
        title=dict(text="Contour path", font=dict(size=13, color="#9ca3af")),
    )
    return fig, losses


def make_loss_curve(losses):
    fig = go.Figure(go.Scatter(
        y=losses, mode="lines",
        line=dict(color="#00d4aa", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,17,23,1)",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(title="Step", gridcolor="#2a2d3a"),
        yaxis=dict(title="Loss", gridcolor="#2a2d3a"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
        title=dict(text="Loss over steps", font=dict(size=13, color="#9ca3af")),
    )
    return fig

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### ⚙️ Parameters")
    optimizer = st.selectbox("Optimizer", ["GD", "SGD", "Momentum", "Adam"])
    lr        = st.slider("Learning rate",  0.001, 0.5,   0.05,  0.001, format="%.3f")
    steps     = st.slider("Steps",          10,    200,   60,    5)
    init_x    = st.slider("Start x",       -3.0,   3.0,   2.5,  0.1)
    init_y    = st.slider("Start y",       -3.0,   3.0,  -2.0,  0.1)
    beta1     = st.slider("β₁ (Momentum)", 0.5,   0.99,   0.9,  0.01)
    beta2     = st.slider("β₂ (Adam v)",   0.9,   0.999,  0.999, 0.001)

with col2:
    st.plotly_chart(make_3d_surface(), use_container_width=True)

# ── Animation + Run ───────────────────────────────────────────────────────────
if st.button("▶ Run optimizer", type="primary"):
    x, y  = float(init_x), float(init_y)
    xs, ys = [x], [y]
    vx = vy = mx = my = vvx = vvy = 0.0

    progress_bar = st.progress(0)
    status_text  = st.empty()

    for t in range(1, steps + 1):
        gx, gy = 2 * x, 2 * y          # gradient of x²+y²

        if optimizer == "GD":
            x -= lr * gx
            y -= lr * gy

        elif optimizer == "SGD":
            nx, ny = np.random.normal(scale=0.15, size=2)
            x -= lr * (gx + nx)
            y -= lr * (gy + ny)

        elif optimizer == "Momentum":
            vx = beta1 * vx + (1 - beta1) * gx
            vy = beta1 * vy + (1 - beta1) * gy
            x -= lr * vx
            y -= lr * vy

        else:  # Adam
            mx  = beta1 * mx  + (1 - beta1) * gx
            my  = beta1 * my  + (1 - beta1) * gy
            vvx = beta2 * vvx + (1 - beta2) * gx**2
            vvy = beta2 * vvy + (1 - beta2) * gy**2
            mxh = mx  / (1 - beta1**t)
            myh = my  / (1 - beta1**t)
            vxh = vvx / (1 - beta2**t)
            vyh = vvy / (1 - beta2**t)
            x -= lr * mxh / (vxh**0.5 + 1e-8)
            y -= lr * myh / (vyh**0.5 + 1e-8)

        xs.append(x)
        ys.append(y)
        progress_bar.progress(int(t / steps * 100))
        status_text.markdown(
            f"<span style='color:#6b7280;font-size:0.82rem;'>Step {t}/{steps} — "
            f"loss = {x**2+y**2:.5f}</span>", unsafe_allow_html=True
        )
        time.sleep(0.01)

    status_text.empty()
    final_loss = x**2 + y**2

    # Result metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Final loss",    f"{final_loss:.5f}")
    m2.metric("Final x",       f"{x:.4f}")
    m3.metric("Final y",       f"{y:.4f}")

    # Contour + loss curve
    contour_fig, losses = make_contour_path(xs, ys, label=optimizer)
    st.plotly_chart(contour_fig, use_container_width=True)
    st.plotly_chart(make_loss_curve(losses), use_container_width=True)

    # ── Exports ───────────────────────────────────────────────────────────────
    st.markdown("---")
    dl1, dl2 = st.columns(2)

    with dl1:
        buf = io.BytesIO()
        pickle.dump({"xs": xs, "ys": ys, "losses": losses}, buf)
        st.download_button(
            "⬇ Download path",
            data=buf.getvalue(),
            file_name="optimizer_path.pkl",
            mime="application/octet-stream",
        )

    with dl2:
        code = f"""\
import numpy as np

x, y  = {init_x}, {init_y}
lr    = {lr}
beta1 = {beta1}
beta2 = {beta2}
vx = vy = mx = my = vvx = vvy = 0.0

for t in range(1, {steps} + 1):
    gx, gy = 2 * x, 2 * y   # gradient of L = x^2 + y^2

    # {optimizer}
"""
        if optimizer == "GD":
            code += """\
    x -= lr * gx
    y -= lr * gy
"""
        elif optimizer == "SGD":
            code += """\
    import numpy as np
    nx, ny = np.random.normal(scale=0.15, size=2)
    x -= lr * (gx + nx)
    y -= lr * (gy + ny)
"""
        elif optimizer == "Momentum":
            code += f"""\
    vx = beta1 * vx + (1 - beta1) * gx
    vy = beta1 * vy + (1 - beta1) * gy
    x -= lr * vx
    y -= lr * vy
"""
        else:
            code += f"""\
    mx  = beta1 * mx  + (1 - beta1) * gx
    my  = beta1 * my  + (1 - beta1) * gy
    vvx = beta2 * vvx + (1 - beta2) * gx**2
    vvy = beta2 * vvy + (1 - beta2) * gy**2
    mxh = mx  / (1 - beta1**t)
    myh = my  / (1 - beta1**t)
    vxh = vvx / (1 - beta2**t)
    vyh = vvy / (1 - beta2**t)
    x -= lr * mxh / (vxh**0.5 + 1e-8)
    y -= lr * myh / (vyh**0.5 + 1e-8)
"""
        code += f"""
print(f"min ≈ ({{x:.4f}}, {{y:.4f}}), loss={{x**2+y**2:.5f}}")
"""
        st.download_button(
            "⬇ Export Python",
            data=code,
            file_name="gradient_descent.py",
            mime="text/x-python",
        )
