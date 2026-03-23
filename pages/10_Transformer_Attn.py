"""Transformer Attention — Multi-head attention visualizer with QKV decomposition."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row, card
from utils.viz import plot_attention_heatmap, _dark_layout

st.set_page_config(page_title="Transformer Attention", layout="wide", page_icon="⚡")
apply_theme()
render_sidebar("Transformer Attn")

hero(
    "Transformer Attention",
    "Explore Scaled Dot-Product Attention, Multi-Head Attention, positional encodings, and QKV decomposition — interactively.",
    pill="Lesson 10", pill_variant="purple",
)

TEAL = "#00d4aa"; ORANGE = "#f97316"; PURPLE = "#818cf8"; SURFACE = "#161b22"; BG = "#0d1117"

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**Attention Config**")
    n_heads   = st.slider("Number of heads", 1, 8, 4)
    seq_len   = st.slider("Sequence length", 4, 16, 8)
    d_model   = st.select_slider("d_model", [32, 64, 128, 256], 64)
    temp      = st.slider("Temperature (softmax)", 0.1, 5.0, 1.0, 0.1)
    show_mask = st.checkbox("Causal mask (decoder)", False)

d_k = d_model // n_heads

tabs = st.tabs(["🔍 Attention Map", "🧮 QKV Math", "📐 Positional Encoding", "🔀 Multi-Head"])

# ── Tab 1: Attention Map ─────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("#### Scaled Dot-Product Attention")
    col1, col2 = st.columns([1, 2])

    with col1:
        tokens = st.text_input("Enter tokens (space-separated)",
                                "the cat sat on the mat today")
        tokens = tokens.strip().split()[:seq_len]
        n = len(tokens)

        seed = st.number_input("Random seed", 0, 100, 42, key="attn_seed")
        np.random.seed(int(seed))

        Q = np.random.randn(n, d_k)
        K = np.random.randn(n, d_k)
        V = np.random.randn(n, d_k)

        scores = Q @ K.T / np.sqrt(d_k) / temp

        if show_mask:
            mask = np.triu(np.ones((n, n)), k=1) * -1e9
            scores += mask

        exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)

        card(f"""
        <b style="color:var(--accent)">Formula:</b><br>
        <code style="color:var(--text)">Attention(Q,K,V) = softmax(QKᵀ / √{d_k}) · V</code><br><br>
        <b>Sequence:</b> {" → ".join(tokens)}<br>
        <b>d_k:</b> {d_k} &nbsp; <b>Heads:</b> {n_heads}
        """)

        metric_row([("Seq Len", n), ("d_k", d_k), ("Params", f"{2*n*d_k}")])

    with col2:
        fig = plot_attention_heatmap(attn, tokens, tokens, "Attention Weights")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Raw attention matrix"):
            import pandas as pd
            df = pd.DataFrame(np.round(attn, 3), index=tokens, columns=tokens)
            st.dataframe(df, use_container_width=True)

# ── Tab 2: QKV Math ──────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Query · Key · Value Decomposition")
    col1, col2, col3 = st.columns(3)

    np.random.seed(42)
    Q2 = np.random.randn(seq_len, d_k)
    K2 = np.random.randn(seq_len, d_k)
    V2 = np.random.randn(seq_len, d_k)

    for col, mat, name, color in [(col1, Q2, "Query (Q)", TEAL),
                                   (col2, K2, "Key (K)", ORANGE),
                                   (col3, V2, "Value (V)", PURPLE)]:
        with col:
            st.markdown(f"**{name}** `shape: ({seq_len}, {d_k})`")
            fig = px.imshow(mat[:min(seq_len,8), :min(d_k,16)],
                            color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(paper_bgcolor=SURFACE, height=200,
                               margin=dict(l=0,r=0,t=10,b=0),
                               font=dict(color="#8b949e"))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Attention Output")
    scores2 = Q2 @ K2.T / np.sqrt(d_k)
    attn2 = np.exp(scores2) / np.exp(scores2).sum(axis=-1, keepdims=True)
    out = attn2 @ V2

    fig_out = px.imshow(out[:min(seq_len,8), :min(d_k,16)],
                        color_continuous_scale="Viridis", aspect="auto")
    fig_out.update_layout(paper_bgcolor=SURFACE, height=220,
                           margin=dict(l=0,r=0,t=10,b=0),
                           font=dict(color="#8b949e"),
                           title=dict(text="Output = Attn · V", font=dict(color="#e6edf3")))
    st.plotly_chart(fig_out, use_container_width=True)

# ── Tab 3: Positional Encoding ────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### Sinusoidal Positional Encoding")
    max_pos = st.slider("Max positions", 8, 64, 32)
    pe_dim  = st.slider("Embedding dim", 8, 128, d_model, 8)

    pos = np.arange(max_pos)[:, np.newaxis]
    dim = np.arange(pe_dim)[np.newaxis, :]
    pe = np.where(
        dim % 2 == 0,
        np.sin(pos / 10000 ** (dim / pe_dim)),
        np.cos(pos / 10000 ** ((dim - 1) / pe_dim))
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.imshow(pe, aspect="auto", color_continuous_scale="RdBu_r",
                        labels=dict(x="Dimension", y="Position"))
        fig.update_layout(paper_bgcolor=SURFACE, font=dict(color="#8b949e"),
                           height=380, margin=dict(l=40, r=10, t=30, b=40),
                           title=dict(text="Positional Encoding Matrix",
                                      font=dict(color="#e6edf3")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        for i in [0, 1, 4, 8]:
            if i < pe_dim:
                fig2.add_trace(go.Scatter(y=pe[:, i], mode="lines",
                                          name=f"dim {i}", line=dict(width=2)))
        fig2.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=BG,
                            font=dict(color="#8b949e"), height=380,
                            margin=dict(l=10, r=10, t=30, b=10),
                            title=dict(text="PE by Dimension",
                                       font=dict(color="#e6edf3")))
        fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig2, use_container_width=True)

    card("""
    <b style="color:var(--accent)">Formula:</b><br>
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))<br>
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))<br><br>
    Each position gets a unique encoding. Similar positions have similar encodings.
    The model can learn to use these for position-aware attention.
    """)

# ── Tab 4: Multi-Head ────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(f"#### Multi-Head Attention ({n_heads} heads)")
    st.markdown("Each head learns to attend to different aspects of the sequence.")

    np.random.seed(42)
    toks = tokens[:min(len(tokens), seq_len)]
    n = len(toks)
    heads_attn = []

    for h in range(n_heads):
        np.random.seed(h * 7 + 13)
        Qh = np.random.randn(n, d_k)
        Kh = np.random.randn(n, d_k)
        sc = Qh @ Kh.T / np.sqrt(d_k)
        if show_mask:
            sc += np.triu(np.ones((n, n)), k=1) * -1e9
        ex = np.exp(sc - sc.max(axis=-1, keepdims=True))
        heads_attn.append(ex / ex.sum(axis=-1, keepdims=True))

    n_cols = min(4, n_heads)
    rows = (n_heads + n_cols - 1) // n_cols
    for row in range(rows):
        cols = st.columns(n_cols)
        for ci in range(n_cols):
            h = row * n_cols + ci
            if h >= n_heads:
                break
            with cols[ci]:
                fig = plot_attention_heatmap(heads_attn[h], toks, toks, f"Head {h+1}")
                fig.update_layout(height=220, margin=dict(l=40, r=5, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Attention (across heads)")
    avg_attn = np.mean(heads_attn, axis=0)
    fig_avg = plot_attention_heatmap(avg_attn, toks, toks, "Averaged Multi-Head Attention")
    st.plotly_chart(fig_avg, use_container_width=True)
