"""Reusable chart builders for NeuralForge Ultra."""
from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st

TEAL   = "#00d4aa"
ORANGE = "#f97316"
PURPLE = "#818cf8"
PINK   = "#ff6eb4"
YELLOW = "#ffd700"
BG     = "#0d1117"
SURFACE= "#161b22"


def _dark_layout(fig, title="", height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e6edf3", size=14)),
        paper_bgcolor=SURFACE, plot_bgcolor=BG,
        font=dict(color="#8b949e", family="IBM Plex Sans"),
        margin=dict(l=10, r=10, t=40 if title else 16, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


def plot_decision_boundary(X, y, w, b, title="Decision Boundary"):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.8, edgecolors="none", s=30)
    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    if abs(w[1]) > 1e-6:
        xs = np.linspace(xmin, xmax, 200)
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, color=TEAL, linewidth=2)
    ax.set_xlabel("x₁", color="#8b949e"); ax.set_ylabel("x₂", color="#8b949e")
    ax.set_title(title, color="#e6edf3", fontsize=13)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values(): spine.set_edgecolor("rgba(255,255,255,0.08)")
    plt.tight_layout()
    return fig


def plot_loss_curve(losses, val_losses=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, mode="lines", name="Train",
                             line=dict(color=TEAL, width=2)))
    if val_losses:
        fig.add_trace(go.Scatter(y=val_losses, mode="lines", name="Val",
                                 line=dict(color=ORANGE, width=2, dash="dash")))
    return _dark_layout(fig, "Loss Curve", 300)


def plot_accuracy_curve(accs, val_accs=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=accs, mode="lines+markers", name="Train",
                             line=dict(color=PURPLE, width=2), marker=dict(size=4)))
    if val_accs:
        fig.add_trace(go.Scatter(y=val_accs, mode="lines+markers", name="Val",
                                 line=dict(color=PINK, width=2, dash="dash"), marker=dict(size=4)))
    return _dark_layout(fig, "Accuracy Curve", 260)


def plot_activation(name):
    x = np.linspace(-6, 6, 300)
    funcs = {
        "Sigmoid":   lambda z: 1 / (1 + np.exp(-z)),
        "ReLU":      lambda z: np.maximum(0, z),
        "LeakyReLU": lambda z: np.where(z >= 0, z, 0.01 * z),
        "Tanh":      np.tanh,
        "Softplus":  lambda z: np.log1p(np.exp(z)),
        "GELU":      lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3))),
        "Swish":     lambda z: z / (1 + np.exp(-z)),
        "Mish":      lambda z: z * np.tanh(np.log1p(np.exp(z))),
        "ELU":       lambda z: np.where(z >= 0, z, np.exp(z) - 1),
        "Softmax":   lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum(),
    }
    y = funcs.get(name, lambda z: z)(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color=TEAL, width=2.5), name=name))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    return _dark_layout(fig, f"{name} activation", 280)


def plot_contour_path(xs, ys, loss_fn=None):
    grid = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(grid, grid)
    Z = X**2 + Y**2 if loss_fn is None else loss_fn(X, Y)
    fig = go.Figure(data=go.Contour(z=Z, x=grid, y=grid,
                                    colorscale="Viridis", showscale=False,
                                    contours=dict(coloring="heatmap")))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                             line=dict(color=ORANGE, width=2.5),
                             marker=dict(size=5, color=ORANGE), name="path"))
    return _dark_layout(fig, "Optimization Path", 380)


def plot_3d_surface():
    g = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(g, g)
    Z = X**2 + Y**2
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Teal", showscale=False)])
    fig.update_layout(paper_bgcolor=SURFACE, font=dict(color="#8b949e"),
                      height=400, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def plot_confusion_matrix(cm, labels=None):
    labels = labels or list(range(cm.shape[0]))
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels,
                                    colorscale="Blues", text=cm,
                                    texttemplate="%{text}",
                                    textfont=dict(size=14, color="white")))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual",
                      paper_bgcolor=SURFACE, font=dict(color="#8b949e"),
                      height=320, margin=dict(l=40, r=10, t=30, b=40))
    return fig


def plot_weight_heatmap(W, title="Weight Matrix"):
    fig = px.imshow(W, color_continuous_scale="RdBu_r", aspect="auto")
    fig.update_layout(paper_bgcolor=SURFACE, font=dict(color="#8b949e"),
                      title=dict(text=title, font=dict(color="#e6edf3")),
                      height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_gradients(names, values):
    colors = [ORANGE if v < 0 else TEAL for v in values]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors))
    return _dark_layout(fig, "Gradient Magnitudes", 280)


def plot_attention_heatmap(attn, tokens_q=None, tokens_k=None, title="Attention"):
    """Plot scaled dot-product attention weights."""
    fig = go.Figure(data=go.Heatmap(
        z=attn,
        x=tokens_k or [f"K{i}" for i in range(attn.shape[1])],
        y=tokens_q or [f"Q{i}" for i in range(attn.shape[0])],
        colorscale="Viridis", showscale=True,
        text=np.round(attn, 2), texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(paper_bgcolor=SURFACE, font=dict(color="#8b949e"),
                      title=dict(text=title, font=dict(color="#e6edf3")),
                      height=380, margin=dict(l=60, r=10, t=50, b=60))
    return fig


def plot_latent_space(z, labels=None, title="Latent Space"):
    """2D scatter of latent embeddings."""
    colors = labels if labels is not None else np.zeros(len(z))
    fig = px.scatter(x=z[:, 0], y=z[:, 1], color=colors.astype(str),
                     color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    return _dark_layout(fig, title, 400)


def plot_gan_progress(real_losses, fake_losses, d_losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=d_losses, name="Discriminator", line=dict(color=TEAL, width=2)))
    fig.add_trace(go.Scatter(y=real_losses, name="G vs Real", line=dict(color=ORANGE, width=2)))
    fig.add_trace(go.Scatter(y=fake_losses, name="G vs Fake", line=dict(color=PURPLE, width=2)))
    return _dark_layout(fig, "GAN Training Dynamics", 300)


def plot_reward_curve(rewards, smoothed=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rewards, mode="lines", name="Episode Reward",
                             line=dict(color=TEAL, width=1, dash="dot"), opacity=0.5))
    if smoothed is not None:
        fig.add_trace(go.Scatter(y=smoothed, mode="lines", name="Smoothed",
                                 line=dict(color=ORANGE, width=2.5)))
    return _dark_layout(fig, "RL Reward Curve", 300)


def plot_architecture_graph(layer_sizes, layer_names=None):
    """Draw a neural network architecture diagram."""
    n_layers = len(layer_sizes)
    max_nodes = max(layer_sizes)
    
    node_x, node_y, node_text = [], [], []
    edge_x, edge_y = [], []
    
    for l_idx, n_nodes in enumerate(layer_sizes):
        x = l_idx / (n_layers - 1) if n_layers > 1 else 0.5
        for n_idx in range(n_nodes):
            y = (n_idx - (n_nodes - 1) / 2) / (max_nodes / 2) if max_nodes > 1 else 0
            node_x.append(x)
            node_y.append(y)
            label = layer_names[l_idx] if layer_names and l_idx < len(layer_names) else f"L{l_idx}"
            node_text.append(f"{label}<br>Node {n_idx+1}")
    
    # Edges between adjacent layers
    offset = 0
    for l_idx in range(n_layers - 1):
        n1, n2 = layer_sizes[l_idx], layer_sizes[l_idx + 1]
        next_off = offset + n1
        for i in range(min(n1, 5)):  # limit edges for visual clarity
            for j in range(min(n2, 5)):
                x1 = l_idx / (n_layers - 1)
                x2 = (l_idx + 1) / (n_layers - 1)
                y1 = (i - (n1 - 1) / 2) / (max_nodes / 2) if max_nodes > 1 else 0
                y2 = (j - (n2 - 1) / 2) / (max_nodes / 2) if max_nodes > 1 else 0
                edge_x += [x1, x2, None]
                edge_y += [y1, y2, None]
        offset = next_off
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="rgba(0,212,170,0.2)", width=1),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                             marker=dict(size=14, color=TEAL,
                                         line=dict(color=PURPLE, width=2)),
                             text=node_text, hoverinfo="text"))
    fig.update_layout(showlegend=False, paper_bgcolor=SURFACE, plot_bgcolor=BG,
                      height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                      yaxis=dict(showgrid=False, showticklabels=False, zeroline=False))
    return fig
