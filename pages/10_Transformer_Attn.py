import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Transformer Attention",
    layout="wide",
    page_icon="⚡"
)

# ====================== SAFE CSS ======================
st.markdown("""
<style>
    section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
    h1, h2, h3, h4 { font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
    .nn-card { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 1.4rem !important; margin-bottom: 1rem !important; }
    .nn-hero { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 16px !important; padding: 2.2rem 2rem !important; margin-bottom: 1.6rem !important; }
    .nn-pill { display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px; background: rgba(129, 140, 248, 0.15) !important; color: #818cf8 !important; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("⚡ Transformer Attention")
    st.markdown("Multi-Head Attention Visualizer")
    st.markdown("---")
    st.markdown("**Attention Config**")
    n_heads = st.slider("Number of Heads", 1, 8, 4)
    seq_len = st.slider("Sequence Length", 4, 16, 8)
    d_model = st.select_slider("d_model", [32, 64, 128, 256], 64)
    temperature = st.slider("Temperature (softmax)", 0.1, 5.0, 1.0, 0.1)
    show_mask = st.checkbox("Causal Mask (Decoder-style)", False)
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 10</div>
    <h1>Transformer Attention</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Interactive exploration of Scaled Dot-Product Attention, QKV decomposition,<br>
        Multi-Head Attention, and Sinusoidal Positional Encoding.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

d_k = d_model // n_heads

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Attention Map", 
    "🧮 QKV Decomposition", 
    "📐 Positional Encoding", 
    "🔀 Multi-Head Attention"
])

# ── Tab 1: Attention Map ─────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Scaled Dot-Product Attention")
    
    tokens_input = st.text_input("Enter tokens (space-separated)", 
                                 "the cat sat on the mat today", key="tokens")
    tokens = tokens_input.strip().split()[:seq_len]
    n = len(tokens)
    
    seed = st.number_input("Random Seed", 0, 100, 42, key="attn_seed")
    np.random.seed(int(seed))
    
    Q = np.random.randn(n, d_k).astype(np.float32)
    K = np.random.randn(n, d_k).astype(np.float32)
    
    scores = Q @ K.T / np.sqrt(d_k) / temperature
    if show_mask:
        mask = np.triu(np.ones((n, n)), k=1) * -1e9
        scores += mask
    
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.markdown(f"""
        <div class="nn-card">
            <b style="color:#818cf8">Formula:</b><br>
            <code>Attention(Q, K, V) = softmax( (Q Kᵀ) / √d_k ) V</code><br><br>
            Tokens: <b>{" → ".join(tokens)}</b><br>
            Length: <b>{n}</b> | d_k: <b>{d_k}</b> | Heads: <b>{n_heads}</b>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        fig_attn = px.imshow(
            attn_weights,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            labels=dict(x="Key Position", y="Query Position"),
            title="Attention Weights"
        )
        fig_attn.update_layout(height=460, paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
        st.plotly_chart(fig_attn, use_container_width=True)
    
    # FIXED: Use unique column names to avoid duplicate error
    with st.expander("📊 Raw Attention Matrix"):
        # Create unique column names by adding index when duplicates exist
        col_names = []
        seen = {}
        for t in tokens:
            if t in seen:
                seen[t] += 1
                col_names.append(f"{t}_{seen[t]}")
            else:
                seen[t] = 0
                col_names.append(t)
        
        df = pd.DataFrame(np.round(attn_weights, 3), 
                         index=[f"{t} ({i})" for i, t in enumerate(tokens)], 
                         columns=col_names)
        
        st.dataframe(df, use_container_width=True)

# ── Tab 2: QKV Math ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Query · Key · Value Decomposition")
    np.random.seed(42)
    Q2 = np.random.randn(seq_len, d_k).astype(np.float32)
    K2 = np.random.randn(seq_len, d_k).astype(np.float32)
    V2 = np.random.randn(seq_len, d_k).astype(np.float32)
    
    c1, c2, c3 = st.columns(3)
    for col, mat, name in zip([c1, c2, c3], [Q2, K2, V2], ["Query (Q)", "Key (K)", "Value (V)"]):
        with col:
            st.markdown(f"**{name}**")
            fig = px.imshow(mat[:8, :min(d_k,12)], color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=220, paper_bgcolor="#161b22", title=name)
            st.plotly_chart(fig, use_container_width=True)
    
    scores2 = Q2 @ K2.T / np.sqrt(d_k)
    attn2 = np.exp(scores2 - scores2.max(axis=-1, keepdims=True))
    attn2 /= attn2.sum(axis=-1, keepdims=True)
    out = attn2 @ V2
    
    st.markdown("#### Attention Output = Attn × V")
    fig_out = px.imshow(out[:8, :min(d_k,12)], color_continuous_scale="Viridis", aspect="auto")
    fig_out.update_layout(height=260, paper_bgcolor="#161b22", title="Output Matrix")
    st.plotly_chart(fig_out, use_container_width=True)

# ── Tab 3 & 4 remain almost identical (only minor cleanups) ─────────────────────
with tab3:
    st.markdown("#### Sinusoidal Positional Encoding")
    max_pos = st.slider("Maximum Positions", 8, 64, 32, key="pe_max")
    pe_dim = st.slider("Embedding Dimension", 8, 256, d_model, 8, key="pe_dim")
    
    pos = np.arange(max_pos)[:, np.newaxis]
    dim = np.arange(pe_dim)[np.newaxis, :]
    pe = np.where(dim % 2 == 0,
                  np.sin(pos / 10000 ** (dim / pe_dim)),
                  np.cos(pos / 10000 ** ((dim - 1) / pe_dim)))
    
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_pe = px.imshow(pe, aspect="auto", color_continuous_scale="RdBu_r")
        fig_pe.update_layout(height=420, paper_bgcolor="#161b22", title="Positional Encoding Matrix")
        st.plotly_chart(fig_pe, use_container_width=True)
    with c2:
        fig_wave = go.Figure()
        for i in [0, 1, 4, 8, 16]:
            if i < pe_dim:
                fig_wave.add_trace(go.Scatter(y=pe[:, i], mode="lines", name=f"dim {i}"))
        fig_wave.update_layout(height=420, paper_bgcolor="#161b22", title="PE Waves")
        st.plotly_chart(fig_wave, use_container_width=True)

with tab4:
    st.markdown(f"#### Multi-Head Attention ({n_heads} heads)")
    toks = tokens[:min(len(tokens), seq_len)]
    n = len(toks)
    heads_attn = []
    for h in range(n_heads):
        np.random.seed(h * 11 + 7)
        Qh = np.random.randn(n, d_k).astype(np.float32)
        Kh = np.random.randn(n, d_k).astype(np.float32)
        sc = Qh @ Kh.T / np.sqrt(d_k)
        if show_mask:
            sc += np.triu(np.ones((n, n)), k=1) * -1e9
        ex = np.exp(sc - sc.max(axis=-1, keepdims=True))
        heads_attn.append(ex / ex.sum(axis=-1, keepdims=True))
    
    # Show heads
    for i in range(0, n_heads, 4):
        cols = st.columns(min(4, n_heads - i))
        for j, col in enumerate(cols):
            h = i + j
            if h >= n_heads: break
            with col:
                fig = px.imshow(heads_attn[h], color_continuous_scale="Viridis", title=f"Head {h+1}")
                fig.update_layout(height=220, paper_bgcolor="#161b22")
                st.plotly_chart(fig, use_container_width=True)
    
    avg_attn = np.mean(heads_attn, axis=0)
    st.markdown("#### Averaged Multi-Head Attention")
    fig_avg = px.imshow(avg_attn, color_continuous_scale="Viridis", title="Average Attention")
    fig_avg.update_layout(height=380, paper_bgcolor="#161b22")
    st.plotly_chart(fig_avg, use_container_width=True)

st.caption("Transformer Attention Visualizer — NeuralForge Ultra v3.0")
