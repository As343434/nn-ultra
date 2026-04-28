import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="NeuralForge • Shabnam",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"   # clean welcome page
)

# ====================== FUTURISTIC CSS ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,500;14..32,700;14..32,800&family=Space+Mono&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .stApp {
        background: radial-gradient(circle at 20% 30%, #05070a, #000000);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Typography */
    h1, h2, h3, h4, .stMarkdown {
        font-family: 'Inter', 'Helvetica Neue', sans-serif !important;
    }

    /* Glow text effect */
    .neon-glow {
        text-shadow: 0 0 8px rgba(0, 212, 170, 0.6), 0 0 20px rgba(0, 212, 170, 0.3);
    }

    /* Hero container */
    .hero-wrapper {
        background: linear-gradient(135deg, rgba(10, 20, 30, 0.7) 0%, rgba(0, 0, 0, 0.85) 100%);
        backdrop-filter: blur(12px);
        border-radius: 48px;
        border: 1px solid rgba(0, 212, 170, 0.4);
        padding: 2.5rem 3rem;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5), 0 0 30px rgba(0,212,170,0.2);
    }

    .badge {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(0, 212, 170, 0.5);
        border-radius: 100px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #00d4aa;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    .dev-info {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 24px;
        padding: 0.8rem 1.8rem;
        margin-top: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        font-family: 'Space Mono', monospace;
    }

    /* Card grid */
    .module-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }

    .module-card {
        background: rgba(18, 25, 35, 0.6);
        backdrop-filter: blur(8px);
        border-radius: 28px;
        border: 1px solid rgba(0, 212, 170, 0.2);
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.2, 0.9, 0.4, 1.1);
        cursor: default;
    }
    .module-card:hover {
        transform: translateY(-6px);
        border-color: #00d4aa;
        box-shadow: 0 0 25px rgba(0, 212, 170, 0.3);
        background: rgba(18, 25, 45, 0.8);
    }
    .module-icon {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
    }
    .module-title {
        font-weight: 800;
        font-size: 1.3rem;
        background: linear-gradient(135deg, #00d4aa, #818cf8);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.5rem;
    }
    .module-desc {
        font-size: 0.85rem;
        color: #9ca3af;
        line-height: 1.4;
    }

    /* Buttons */
    .launch-btn {
        background: linear-gradient(90deg, #00d4aa, #0f6b5e);
        border: none;
        color: black;
        font-weight: 700;
        padding: 0.7rem 1.8rem;
        border-radius: 40px;
        font-size: 1rem;
        cursor: pointer;
        transition: 0.2s;
        box-shadow: 0 0 12px #00d4aa;
        display: inline-block;
        text-align: center;
    }
    .launch-btn:hover {
        transform: scale(1.03);
        box-shadow: 0 0 25px #00d4aa;
    }

    hr {
        border-color: rgba(0, 212, 170, 0.3);
        margin: 2.5rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0c10;
    }
    ::-webkit-scrollbar-thumb {
        background: #00d4aa;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ====================== 3D NEURAL NETWORK (PURE ABSTRACT) ======================
def create_rotating_neural_net():
    """Glowing, abstract 3D neural network – no baby, pure cyberpunk style"""
    # Generate layers with random offsets for organic look
    np.random.seed(42)
    n_layers = 5
    nodes_per_layer = [5, 8, 12, 8, 4]
    layer_positions = np.linspace(-3.2, 3.2, n_layers)
    
    all_nodes = []
    layer_ranges = []
    for i, (x, n_nodes) in enumerate(zip(layer_positions, nodes_per_layer)):
        y_vals = np.linspace(-2.5, 2.5, n_nodes)
        z_vals = np.random.uniform(-1.5, 1.5, n_nodes) * 0.8
        layer_nodes = np.column_stack([np.full(n_nodes, x), y_vals, z_vals])
        all_nodes.append(layer_nodes)
        layer_ranges.append((len(all_nodes)-1, n_nodes))
    
    all_nodes = np.vstack(all_nodes)
    
    # Build edges between consecutive layers
    edges = []
    cumulative = 0
    for i in range(n_layers - 1):
        start_idx = cumulative
        end_idx = cumulative + nodes_per_layer[i]
        next_start = end_idx
        next_end = next_start + nodes_per_layer[i+1]
        for s in range(start_idx, end_idx):
            for e in range(next_start, next_end):
                # only connect if distance in y is reasonable
                if abs(all_nodes[s,1] - all_nodes[e,1]) < 2.0:
                    edges.append((all_nodes[s], all_nodes[e]))
        cumulative += nodes_per_layer[i]
    
    # Create figure
    fig = go.Figure()
    
    # Edges (glowing lines)
    for (p1, p2) in edges:
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color='rgba(0, 212, 170, 0.5)', width=2),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Nodes (pulsing spheres)
    fig.add_trace(go.Scatter3d(
        x=all_nodes[:,0], y=all_nodes[:,1], z=all_nodes[:,2],
        mode='markers',
        marker=dict(
            size=4,
            color='#00d4aa',
            opacity=0.9,
            symbol='circle',
            line=dict(width=1, color='white')
        ),
        name='Neurons'
    ))
    
    # Additional floating particles for atmosphere
    extra_particles = np.random.randn(150, 3) * 3.5
    fig.add_trace(go.Scatter3d(
        x=extra_particles[:,0], y=extra_particles[:,1], z=extra_particles[:,2],
        mode='markers',
        marker=dict(size=1.5, color='#818cf8', opacity=0.4),
        showlegend=False
    ))
    
    # Camera animation frames
    n_frames = 60
    frames = []
    for i in range(n_frames):
        angle = (i / n_frames) * 2 * np.pi
        radius = 6.5
        eye_x = radius * np.cos(angle)
        eye_z = radius * np.sin(angle)
        eye_y = 1.5 + 0.8 * np.sin(angle * 1.2)
        frames.append(go.Frame(
            layout=go.Layout(
                scene=dict(
                    camera=dict(
                        eye=dict(x=eye_x, y=eye_y, z=eye_z),
                        up=dict(x=0, y=1, z=0),
                        center=dict(x=0, y=0, z=0)
                    )
                )
            )
        ))
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="▶ ROTATE",
                     method="animate",
                     args=[None, {"frame": {"duration": 70, "redraw": True},
                                  "fromcurrent": True, "mode": "immediate",
                                  "transition": {"duration": 0}}]),
                dict(label="⏹ STOP",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ],
            direction="left",
            pad={"r": 10, "t": 10},
            x=0.02,
            y=0.95,
            font=dict(color="white", size=11)
        )],
        scene=dict(
            xaxis=dict(visible=False, showticklabels=False, title=""),
            yaxis=dict(visible=False, showticklabels=False, title=""),
            zaxis=dict(visible=False, showticklabels=False, title=""),
            aspectmode='cube',
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=5.2, y=2.0, z=3.8))
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        showlegend=False
    )
    fig.frames = frames
    return fig

# ====================== MAIN WELCOME PAGE ======================
# Hero Section
st.markdown("""
<div class="hero-wrapper">
    <div class="badge">⚡ NEURAL ARCHITECTURE SUITE</div>
    <h1 style="font-size: 4rem; font-weight: 800; margin-bottom: 0.5rem;">
        <span class="neon-glow">NeuralForge</span>
    </h1>
    <p style="font-size: 1.3rem; color: #cbd5e1; max-width: 700px; margin-bottom: 2rem;">
        Design, train, and experiment with 16 cutting‑edge neural modules — all in one immersive workspace.
    </p>
    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
        <div class="launch-btn" onclick="alert('Launching workspace... (sidebar ready)')">🚀 ENTER STUDIO</div>
        <div style="background: rgba(255,255,255,0.05); border-radius: 40px; padding: 0.7rem 1.5rem; color: #94a3b8;">
            🧠 v3.0 • 2302420002
        </div>
    </div>
    <div class="dev-info">
        <span>👩‍💻 <strong style="color:#00d4aa;"></strong> / Lead AI Engineer</span>
        <span>🆔 <strong style="color:#f97316;">2302420002</strong></span>
        <span>⚡ 16 modules • 10+ datasets • RL • GAN • Transformers</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 3D Neural Network Animation
st.markdown("### 🧬 Neuro‑Dynamic Core")
st.markdown("*Abstract 3D neural field – rotating, adaptive intelligence*")
nn_fig = create_rotating_neural_net()
st.plotly_chart(nn_fig, use_container_width=True)

st.markdown("---")
st.markdown("## ⚡ Full Arsenal")
st.markdown("*Click any module to explore – every block is interactive*")

# Module data
modules = [
    ("⬡", "Perceptron", "Single-neuron classifier • 2D boundary animation", "teal"),
    ("⟶", "Forward Pass", "10 activations • Derivative viewer • Taylor series", "purple"),
    ("↺", "Backpropagation", "Chain‑rule visualizer • Gradient flow", "teal"),
    ("↗", "Gradient Descent", "5 optimizers • 3D loss landscapes", "orange"),
    ("⬛", "ANN / MLP", "NumPy/PyTorch • CSV upload • Weight histograms", "teal"),
    ("◫", "CNN", "MNIST/Fashion‑MNIST • Feature maps • CAM", "purple"),
    ("⇌", "RNN / LSTM", "Sequence prediction • Attention overlay", "teal"),
    ("◎", "Autoencoder / VAE", "Latent space interpolation • Denoising", "orange"),
    ("◉", "OpenCV Vision", "15 preproc ops • Fourier • CNN pipeline", "teal"),
    ("⚡", "Transformer Attn", "Multi‑head attention • QKV visualization", "purple"),
    ("🎮", "GAN Lab", "Train DCGAN • Mode collapse detection", "orange"),
    ("🤖", "RL Agent", "DQN on CartPole • Q‑value heatmaps", "teal"),
    ("🧬", "NAS Explorer", "Architecture search • Efficiency frontier", "purple"),
    ("📊", "Model Comparison", "Radar charts • Benchmarking", "orange"),
    ("🧠", "AI Explainer", "Claude‑powered • Ask anything", "teal"),
    ("📤", "Export Hub", "ONNX • PyTorch • Python script", "purple"),
]

# Create grid with custom styling
cols = st.columns(3)
for idx, (icon, name, desc, color) in enumerate(modules):
    border_color = "#00d4aa" if color == "teal" else "#f97316" if color == "orange" else "#818cf8"
    with cols[idx % 3]:
        st.markdown(f"""
        <div class="module-card" style="border-left: 4px solid {border_color};">
            <div class="module-icon">{icon}</div>
            <div class="module-title">{name}</div>
            <div class="module-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Optional: Quick demo teaser (compact)
with st.expander("▶ Quick Neural Playground – instant 3‑layer MLP", expanded=False):
    col_a, col_b = st.columns([1, 2])
    with col_a:
        lr = st.slider("LR", 0.001, 0.3, 0.03, key="welcome_lr")
        epochs = st.slider("Epochs", 20, 200, 80, key="welcome_ep")
        ds = st.selectbox("Dataset", ["moons", "circles"], key="welcome_ds")
        if st.button("🔥 Train Demo", type="primary"):
            # Simple training snippet (same logic as before, but shortened for demo)
            # (showing only accuracy to keep it clean)
            X, y = make_moons(n_samples=200, noise=0.1, random_state=42) if ds == "moons" else make_circles(n_samples=200, noise=0.07, factor=0.5, random_state=42)
            X = StandardScaler().fit_transform(X)
            y = y.astype(int)
            h_dim = 12
            W1 = np.random.randn(2, h_dim)*0.2
            b1 = np.zeros(h_dim)
            W2 = np.random.randn(h_dim, 2)*0.2
            b2 = np.zeros(2)
            acc_hist = []
            for ep in range(epochs):
                a1 = np.maximum(0, X@W1 + b1)
                logits = a1@W2 + b2
                exp = np.exp(logits - logits.max(1, keepdims=True))
                probs = exp / exp.sum(1, keepdims=True)
                pred = np.argmax(probs, axis=1)
                acc = np.mean(pred == y)
                acc_hist.append(acc)
                # backward
                y_onehot = np.eye(2)[y]
                d_logits = (probs - y_onehot)/len(X)
                dW2 = a1.T @ d_logits
                db2 = d_logits.sum(0)
                da1 = d_logits @ W2.T
                da1[a1 <= 0] = 0
                dW1 = X.T @ da1
                db1 = da1.sum(0)
                W1 -= lr * dW1
                b1 -= lr * db1
                W2 -= lr * dW2
                b2 -= lr * db2
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=acc_hist, mode='lines', line=dict(color='#00d4aa', width=2)))
            fig.update_layout(title="Accuracy over epochs", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=250, margin=dict(l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✅ Final accuracy: {acc_hist[-1]:.2%}")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1.5rem; border-top: 1px solid rgba(0,212,170,0.2);">
    <span style="color:#6c7a8e;">⚡ NeuralForge Ultra – Welcome Edition | Crafted by </span>
    <strong style="color:#00d4aa;"> (2302420002)</strong>
    <span style="color:#6c7a8e;"> • Use sidebar to navigate full toolset</span>
    <br><small style="color:#4a5568;">Immersive neural studio – 16 modules ready to deploy</small>
</div>
""", unsafe_allow_html=True)

# Restore sidebar (collapsed by default but usable)
st.sidebar.markdown("## 🧠 NeuralForge")
st.sidebar.info("Welcome back, . Select any module from the grid above to begin your neural engineering journey.")
st.sidebar.markdown(f"**Developer:** Shabnam  \n**ID:** 2302420002  \n**Status:** Full access")
