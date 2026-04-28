import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import time

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Hopfield Network",
    layout="wide",
    page_icon="🧠"
)

# ====================== CSS ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
h1,h2,h3,h4 {
    font-family:'IBM Plex Sans',sans-serif !important;
    font-weight:700 !important;
    letter-spacing:-0.02em !important;
}
.nn-hero {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 2.2rem 2rem !important;
    margin-bottom: 1.6rem !important;
}
.nn-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    background: rgba(139,92,246,0.18) !important;
    color: #a78bfa !important;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.nn-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.4rem !important;
    margin-bottom: 1rem !important;
}
.step-badge {
    display: inline-block;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: rgba(139,92,246,0.25);
    color: #a78bfa;
    font-weight: 700;
    font-size: 0.85rem;
    text-align: center;
    line-height: 28px;
    margin-right: 8px;
}
.energy-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.pixel-grid {
    display: inline-grid;
    gap: 2px;
    padding: 8px;
    background: #0f172a;
    border-radius: 8px;
    border: 1px solid #1e293b;
}
.match-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
}
button[kind="primary"],button[kind="secondary"] {
    border-radius: 8px !important;
    font-weight: 700 !important;
}
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ====================== LETTER PATTERNS (9x7 grid = 63 neurons) ======================
# Each letter is a 9-row x 7-col binary pattern, flattened. +1 = ON, -1 = OFF

LETTERS = {}

LETTERS['A'] = np.array([
    [0,1,1,1,1,1,0],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,1,1,1,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [0,0,0,0,0,0,0],
])

LETTERS['B'] = np.array([
    [1,1,1,1,1,0,0],
    [1,1,0,0,1,1,0],
    [1,1,0,0,1,1,0],
    [1,1,1,1,1,0,0],
    [1,1,0,0,1,1,0],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0],
])

LETTERS['C'] = np.array([
    [0,1,1,1,1,1,0],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,1,1],
    [0,1,1,1,1,1,0],
    [0,0,0,0,0,0,0],
])

LETTERS['E'] = np.array([
    [1,1,1,1,1,1,1],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,1,1,1,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0],
])

LETTERS['H'] = np.array([
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,1,1,1,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [0,0,0,0,0,0,0],
])

LETTERS['I'] = np.array([
    [1,1,1,1,1,1,1],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0],
])

LETTERS['L'] = np.array([
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0],
])

LETTERS['N'] = np.array([
    [1,1,0,0,0,1,1],
    [1,1,1,0,0,1,1],
    [1,1,1,0,0,1,1],
    [1,1,0,1,0,1,1],
    [1,1,0,1,0,1,1],
    [1,1,0,0,1,1,1],
    [1,1,0,0,1,1,1],
    [1,1,0,0,0,1,1],
    [0,0,0,0,0,0,0],
])

LETTERS['O'] = np.array([
    [0,1,1,1,1,1,0],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [0,1,1,1,1,1,0],
    [0,0,0,0,0,0,0],
])

LETTERS['S'] = np.array([
    [0,1,1,1,1,1,0],
    [1,1,0,0,0,1,1],
    [1,1,0,0,0,0,0],
    [0,1,1,1,1,0,0],
    [0,0,0,1,1,1,0],
    [0,0,0,0,0,1,1],
    [1,1,0,0,0,1,1],
    [0,1,1,1,1,1,0],
    [0,0,0,0,0,0,0],
])

LETTERS['T'] = np.array([
    [1,1,1,1,1,1,1],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0],
])

LETTERS['Z'] = np.array([
    [1,1,1,1,1,1,1],
    [0,0,0,0,0,1,1],
    [0,0,0,0,1,1,0],
    [0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0],
    [0,1,1,0,0,0,0],
    [1,1,0,0,0,0,0],
    [1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0],
])

ROWS, COLS = 9, 7
N_NEURONS  = ROWS * COLS  # 63

def to_bipolar(pattern_01):
    """Convert 0/1 grid to +1/-1 vector."""
    return (pattern_01.flatten() * 2 - 1).astype(float)

def to_grid(bipolar_vec):
    """Convert +1/-1 vector back to 0/1 grid."""
    return ((bipolar_vec.reshape(ROWS, COLS) + 1) / 2).clip(0,1)

# ====================== HOPFIELD CORE ======================
class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def train(self, patterns):
        """Hebbian learning rule."""
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= self.n
        np.fill_diagonal(self.W, 0)

    def energy(self, state):
        return -0.5 * state @ self.W @ state

    def update_async(self, state, steps=20):
        """Asynchronous update — one random neuron at a time."""
        s = state.copy()
        history = [s.copy()]
        energy_hist = [self.energy(s)]
        for _ in range(steps * self.n):
            i = np.random.randint(self.n)
            s[i] = 1.0 if (self.W[i] @ s) >= 0 else -1.0
            if len(history) == 0 or not np.array_equal(s, history[-1]):
                history.append(s.copy())
                energy_hist.append(self.energy(s))
            if len(history) > steps + 1:
                break
        return history, energy_hist

    def update_sync(self, state, steps=15):
        """Synchronous update — all neurons at once."""
        s = state.copy()
        history = [s.copy()]
        energy_hist = [self.energy(s)]
        for _ in range(steps):
            s_new = np.sign(self.W @ s)
            s_new[s_new == 0] = s[s_new == 0]
            energy_hist.append(self.energy(s_new))
            history.append(s_new.copy())
            if np.array_equal(s_new, s):
                break
            s = s_new
        return history, energy_hist

def add_noise(bipolar_vec, noise_pct):
    """Flip a fraction of bits."""
    noisy = bipolar_vec.copy()
    n_flip = int(len(noisy) * noise_pct / 100)
    flip_idx = np.random.choice(len(noisy), n_flip, replace=False)
    noisy[flip_idx] *= -1
    return noisy

def pattern_similarity(a, b):
    """How similar two bipolar vectors are (0-100%)."""
    return round(float(np.mean(a == b) * 100), 1)

# ====================== PLOTTING ======================
CELL_PX = 18

def grid_to_fig(grid_01, title="", highlight_diff=None, ref_grid=None):
    """Render a 9x7 pixel grid as a Plotly heatmap."""
    z = grid_01.copy()
    fig = go.Figure(go.Heatmap(
        z=z,
        colorscale=[[0,"#0f172a"],[1,"#a78bfa"]],
        zmin=0, zmax=1,
        showscale=False,
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0", size=13), x=0.5),
        width=CELL_PX*COLS + 60,
        height=CELL_PX*ROWS + 60,
        margin=dict(l=10,r=10,t=35,b=10),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig

def energy_fig(energy_hist):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=energy_hist, mode="lines+markers",
        line=dict(color="#a78bfa", width=2),
        marker=dict(size=6, color="#a78bfa"),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.1)"
    ))
    fig.update_layout(
        title="Energy over Iterations",
        height=220,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        margin=dict(t=35,b=30,l=40,r=20),
        xaxis=dict(title="Step", gridcolor="#1e293b"),
        yaxis=dict(title="E", gridcolor="#1e293b"),
    )
    return fig

def weight_fig(W):
    fig = go.Figure(go.Heatmap(
        z=W,
        colorscale="RdBu",
        zmid=0,
        showscale=True,
        colorbar=dict(tickfont=dict(color="#94a3b8"), len=0.8)
    ))
    fig.update_layout(
        title="Weight Matrix W",
        height=320,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        margin=dict(t=35,b=10,l=10,r=60),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🧠 Hopfield Network")
    st.markdown("Associative Memory & Pattern Recall")
    st.markdown("---")

    st.markdown("**Available Letters**")
    st.markdown(" · ".join(f"`{k}`" for k in sorted(LETTERS.keys())))
    st.markdown("---")

    st.markdown("""
**How it works**
1. Select letters to **store** as memories
2. Pick a letter to **recall**
3. Add noise to corrupt it
4. Network **converges** to nearest memory

**Key equations**

`W = (1/N) Σ xᵢ xᵢᵀ`  
`s(t+1) = sign(W·s(t))`  
`E = -½ sᵀWs`
    """)
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 15</div>
    <h1>Hopfield Network — Pattern Memory</h1>
    <p style="color:var(--muted);font-size:1.1rem;">
        Store letters as memories · Corrupt with noise · Watch the network recall the original.<br>
        Classic associative memory with energy-based dynamics.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
## What is a Hopfield Network?

A **Hopfield Network** is a recurrent neural network that acts as an **associative memory** — given a partial or noisy input, it recalls the closest stored pattern.

### Architecture
- **N neurons**, each with state $s_i \in \{+1, -1\}$
- **Fully connected** (every neuron talks to every other)
- **Symmetric weights**: $W_{ij} = W_{ji}$, $W_{ii} = 0$

### Learning — Hebbian Rule
$$W = \frac{1}{N} \sum_{\mu=1}^{p} \boldsymbol{\xi}^\mu (\boldsymbol{\xi}^\mu)^T$$

Patterns that fire together, wire together.

### Update Rule
**Asynchronous** (one neuron at a time):
$$s_i(t+1) = \text{sign}\left(\sum_j W_{ij} s_j(t)\right)$$

**Synchronous** (all at once):
$$\boldsymbol{s}(t+1) = \text{sign}(W \cdot \boldsymbol{s}(t))$$

### Energy Function
$$E = -\frac{1}{2} \boldsymbol{s}^T W \boldsymbol{s}$$

The network **always decreases energy** until it reaches a stable attractor (stored memory).

### Capacity
A network of $N$ neurons can reliably store $\approx 0.138 \times N$ patterns.  
For our 63-neuron network: **~8 letters max** for reliable recall.

### Spurious States
Sometimes the network converges to a **spurious attractor** — a mixture of stored patterns. This is a known limitation of Hopfield networks.
    """)

st.markdown("---")

# ====================== MAIN TABS ======================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔤 Store & Recall",
    "🎨 Interactive Pixel Canvas",
    "⚡ Energy Landscape",
    "🔬 Weight Matrix"
])

# ════════════════════════════════════════════════════════
# TAB 1 — Store Letters & Recall
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Step 1 — Choose Letters to Store as Memories")

    all_letters = sorted(LETTERS.keys())
    default_store = ["A", "B", "C", "H", "I"]

    stored_letters = st.multiselect(
        "Select letters to store (recommended: 3–7 for reliable recall)",
        options=all_letters,
        default=default_store,
        key="stored"
    )

    max_capacity = int(0.138 * N_NEURONS)
    if len(stored_letters) > max_capacity:
        st.warning(f"⚠️ Storing more than {max_capacity} patterns may cause unreliable recall (spurious states).")
    elif len(stored_letters) == 0:
        st.info("Select at least one letter to store.")
        st.stop()

    # Preview stored letters
    st.markdown("**Stored patterns:**")
    prev_cols = st.columns(len(stored_letters))
    for ci, ltr in enumerate(stored_letters):
        with prev_cols[ci]:
            grid = LETTERS[ltr]
            st.plotly_chart(grid_to_fig(grid, title=f"'{ltr}'"), use_container_width=False,
                            config={"displayModeBar":False})

    st.markdown("---")
    st.subheader("Step 2 — Train the Network")

    if st.button("🧠 Train Hopfield Network", type="primary", use_container_width=True):
        patterns_bp = [to_bipolar(LETTERS[l]) for l in stored_letters]
        net = HopfieldNetwork(N_NEURONS)
        net.train(patterns_bp)
        st.session_state["hopfield_net"]      = net
        st.session_state["stored_letters"]    = stored_letters
        st.session_state["patterns_bp"]       = patterns_bp
        st.success(f"✅ Network trained on {len(stored_letters)} patterns ({N_NEURONS} neurons, {N_NEURONS**2} weights).")

    st.markdown("---")
    st.subheader("Step 3 — Recall a Pattern")

    if "hopfield_net" not in st.session_state:
        st.info("Train the network first (Step 2).")
    else:
        net           = st.session_state["hopfield_net"]
        stored_ltrs   = st.session_state["stored_letters"]
        patterns_bp   = st.session_state["patterns_bp"]

        rc1, rc2 = st.columns([1, 2], gap="large")

        with rc1:
            query_letter = st.selectbox("Letter to query (can be any stored letter)", stored_ltrs, key="query_ltr")
            noise_pct    = st.slider("Noise level (%)", 0, 60, 30, 5, key="noise_pct",
                                     help="Percentage of pixels to randomly flip")
            update_mode  = st.radio("Update mode", ["Synchronous", "Asynchronous"], horizontal=True, key="upd_mode")
            n_steps      = st.slider("Max iterations", 5, 50, 20, 5, key="n_steps")

            if st.button("🔄 Add Noise & Recall", type="primary", use_container_width=True, key="recall_btn"):
                original_bp = to_bipolar(LETTERS[query_letter])
                noisy_bp    = add_noise(original_bp, noise_pct)

                if update_mode == "Synchronous":
                    history, energy_hist = net.update_sync(noisy_bp, steps=n_steps)
                else:
                    history, energy_hist = net.update_async(noisy_bp, steps=n_steps)

                recalled_bp = history[-1]

                # Find best match among stored patterns
                sims = {l: pattern_similarity(recalled_bp, p) for l,p in zip(stored_ltrs, patterns_bp)}
                best_match  = max(sims, key=sims.get)
                best_sim    = sims[best_match]
                orig_sim    = pattern_similarity(recalled_bp, original_bp)

                st.session_state["recall_result"] = {
                    "original_bp":  original_bp,
                    "noisy_bp":     noisy_bp,
                    "recalled_bp":  recalled_bp,
                    "history":      history,
                    "energy_hist":  energy_hist,
                    "best_match":   best_match,
                    "best_sim":     best_sim,
                    "orig_sim":     orig_sim,
                    "sims":         sims,
                    "query_letter": query_letter,
                    "noise_pct":    noise_pct,
                }

        with rc2:
            if "recall_result" in st.session_state:
                res = st.session_state["recall_result"]

                # 3-panel display
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.plotly_chart(grid_to_fig(to_grid(res["original_bp"]),
                        title=f"Original '{res['query_letter']}'"),
                        use_container_width=False, config={"displayModeBar":False})
                    st.caption("✅ Stored memory")

                with p2:
                    st.plotly_chart(grid_to_fig(to_grid(res["noisy_bp"]),
                        title=f"Noisy ({res['noise_pct']}%)"),
                        use_container_width=False, config={"displayModeBar":False})
                    st.caption("⚡ Network input")

                with p3:
                    st.plotly_chart(grid_to_fig(to_grid(res["recalled_bp"]),
                        title=f"Recalled → '{res['best_match']}'"),
                        use_container_width=False, config={"displayModeBar":False})
                    st.caption(f"🧠 Network output")

                # Result card
                match_color = "#22c55e" if res["best_sim"]>85 else "#f59e0b" if res["best_sim"]>60 else "#ef4444"
                verdict = "✅ Perfect Recall" if res["orig_sim"]>90 else "⚠️ Partial Recall" if res["orig_sim"]>60 else "❌ Spurious State"

                st.markdown(f"""
                <div class="nn-card" style="margin-top:0.5rem">
                    <div style="display:flex;gap:2rem;flex-wrap:wrap;align-items:center">
                        <div>
                            <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase">Match to Original</div>
                            <div class="match-pct" style="font-size:2rem;color:{match_color}">{res['orig_sim']}%</div>
                        </div>
                        <div>
                            <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase">Best Pattern Match</div>
                            <div class="match-pct" style="font-size:2rem;color:#a78bfa">'{res['best_match']}' @ {res['best_sim']}%</div>
                        </div>
                        <div>
                            <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase">Verdict</div>
                            <div style="font-size:1rem;font-weight:700">{verdict}</div>
                        </div>
                        <div>
                            <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase">Iterations</div>
                            <div class="match-pct" style="font-size:2rem;color:#38bdf8">{len(res['history'])-1}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Similarity to all stored patterns
                st.markdown("**Similarity to all stored patterns:**")
                sim_cols = st.columns(len(res["sims"]))
                for ci, (ltr, sim) in enumerate(sorted(res["sims"].items(), key=lambda x:-x[1])):
                    col = "#22c55e" if sim>85 else "#f59e0b" if sim>60 else "#64748b"
                    with sim_cols[ci]:
                        st.markdown(f"""
                        <div style="text-align:center;padding:0.5rem;background:#0f172a;border-radius:8px;border:1px solid #1e293b">
                            <div style="font-size:1.2rem;font-weight:700;color:{col}">'{ltr}'</div>
                            <div style="font-family:'IBM Plex Mono';font-size:0.85rem;color:{col}">{sim}%</div>
                        </div>""", unsafe_allow_html=True)

                # Energy chart
                st.markdown("**Energy during convergence:**")
                st.plotly_chart(energy_fig(res["energy_hist"]), use_container_width=True,
                                config={"displayModeBar":False})

                # Recall animation (step through history)
                with st.expander("🎬 Step-through Recall Animation", expanded=False):
                    step = st.slider("Step", 0, len(res["history"])-1, 0, key="anim_step")
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        st.plotly_chart(grid_to_fig(to_grid(res["history"][step]),
                            title=f"State at step {step}"),
                            use_container_width=False, config={"displayModeBar":False})
                    with ac2:
                        e_val = res["energy_hist"][min(step, len(res["energy_hist"])-1)]
                        st.markdown(f"""
                        <div class="nn-card" style="margin-top:2rem">
                            <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase">Energy at step {step}</div>
                            <div class="energy-val">{e_val:.2f}</div>
                            <div style="color:#64748b;font-size:0.75rem;margin-top:0.5rem">
                                {'🟢 Converged' if step==len(res["history"])-1 else '🔄 Converging...'}
                            </div>
                        </div>""", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style='padding:3rem;text-align:center;color:#475569;border:1px dashed #334155;border-radius:12px'>
                    <div style='font-size:3rem'>🧠</div>
                    <div style='margin-top:0.5rem'>Set noise level and click <b>Add Noise & Recall</b></div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 2 — Interactive Pixel Canvas (draw your own pattern)
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("🎨 Draw Your Own Pattern & Recall")
    st.caption("Click pixels to toggle them ON/OFF, then recall the closest stored memory.")

    if "hopfield_net" not in st.session_state:
        st.info("Train the network in the **Store & Recall** tab first.")
    else:
        net_c       = st.session_state["hopfield_net"]
        stored_ltrs_c = st.session_state["stored_letters"]
        patterns_c  = st.session_state["patterns_bp"]

        st.markdown("**Pixel canvas — click to toggle cells:**")

        # Init canvas in session state
        if "canvas" not in st.session_state:
            st.session_state["canvas"] = np.zeros((ROWS, COLS), dtype=int)

        canvas = st.session_state["canvas"]

        # Quick fill buttons
        qb1, qb2, qb3 = st.columns(3)
        with qb1:
            if st.button("⬜ Clear Canvas"):
                st.session_state["canvas"] = np.zeros((ROWS, COLS), dtype=int)
                st.rerun()
        with qb2:
            if st.button("⬛ Fill All"):
                st.session_state["canvas"] = np.ones((ROWS, COLS), dtype=int)
                st.rerun()
        with qb3:
            prefill = st.selectbox("Load letter", ["—"] + stored_ltrs_c, key="prefill_sel")
            if prefill != "—":
                if st.button(f"Load '{prefill}'"):
                    st.session_state["canvas"] = LETTERS[prefill].copy()
                    st.rerun()

        # Render 9x7 toggle grid
        for r in range(ROWS):
            row_cols = st.columns(COLS)
            for c in range(COLS):
                with row_cols[c]:
                    cell_val = st.session_state["canvas"][r, c]
                    btn_label = "🟪" if cell_val else "⬛"
                    if st.button(btn_label, key=f"cell_{r}_{c}"):
                        st.session_state["canvas"][r, c] = 1 - cell_val
                        st.rerun()

        st.markdown("---")
        if st.button("🧠 Recall from Canvas", type="primary", use_container_width=True):
            canvas_bp = to_bipolar(st.session_state["canvas"])
            _, energy_c = net_c.update_sync(canvas_bp, steps=20)
            hist_c, energy_c = net_c.update_sync(canvas_bp, steps=20)
            recalled_c = hist_c[-1]

            sims_c = {l: pattern_similarity(recalled_c, p) for l,p in zip(stored_ltrs_c, patterns_c)}
            best_c = max(sims_c, key=sims_c.get)

            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(grid_to_fig(st.session_state["canvas"].astype(float), title="Your Drawing"),
                                use_container_width=False, config={"displayModeBar":False})
            with cb:
                st.plotly_chart(grid_to_fig(to_grid(recalled_c), title=f"Recalled → '{best_c}'"),
                                use_container_width=False, config={"displayModeBar":False})

            best_color = "#22c55e" if sims_c[best_c]>80 else "#f59e0b"
            st.markdown(f"""
            <div class="nn-card">
                <b>Best match:</b> <span style="color:{best_color};font-family:'IBM Plex Mono';font-size:1.2rem">
                '{best_c}' @ {sims_c[best_c]}%</span>
                &nbsp;&nbsp; | &nbsp;&nbsp;
                Converged in <b>{len(hist_c)-1}</b> steps
                &nbsp;&nbsp; | &nbsp;&nbsp;
                Final energy: <b>{energy_c[-1]:.2f}</b>
            </div>""", unsafe_allow_html=True)

            st.markdown("**All pattern similarities:**")
            sim_cs = st.columns(len(sims_c))
            for ci,(ltr,sim) in enumerate(sorted(sims_c.items(),key=lambda x:-x[1])):
                col = "#22c55e" if sim>80 else "#f59e0b" if sim>55 else "#475569"
                with sim_cs[ci]:
                    st.markdown(f"""
                    <div style="text-align:center;padding:0.4rem;background:#0f172a;border-radius:6px;border:1px solid #1e293b">
                        <div style="font-size:1.1rem;font-weight:700;color:{col}">'{ltr}'</div>
                        <div style="font-family:'IBM Plex Mono';font-size:0.8rem;color:{col}">{sim}%</div>
                    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 3 — Energy Landscape
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚡ Energy Landscape Explorer")

    if "hopfield_net" not in st.session_state:
        st.info("Train the network first.")
    else:
        net_e        = st.session_state["hopfield_net"]
        stored_ltrs_e = st.session_state["stored_letters"]
        patterns_e   = st.session_state["patterns_bp"]

        st.markdown("""
        The Hopfield network's **energy function** $E = -\\frac{1}{2} s^T W s$ is a Lyapunov function —
        it always **decreases or stays flat** during updates. Stored memories are **local minima** (attractors).
        """)

        # Energy of each stored pattern
        energies_stored = {l: net_e.energy(p) for l,p in zip(stored_ltrs_e, patterns_e)}

        # Energy at various noise levels for a selected letter
        el_col1, el_col2 = st.columns([1,2], gap="large")
        with el_col1:
            base_letter = st.selectbox("Base letter for noise sweep", stored_ltrs_e, key="energy_base")
            n_trials    = st.slider("Trials per noise level", 3, 20, 8, key="n_trials")
            noise_range = st.slider("Noise range (%)", 0, 70, (0,60), key="noise_range")
            run_energy  = st.button("📊 Run Energy Sweep", use_container_width=True)

        with el_col2:
            # Bar chart — energy of each stored memory
            fig_bar = go.Figure(go.Bar(
                x=list(energies_stored.keys()),
                y=list(energies_stored.values()),
                marker_color="#a78bfa",
                text=[f"{v:.1f}" for v in energies_stored.values()],
                textposition="outside"
            ))
            fig_bar.update_layout(
                title="Energy of Each Stored Memory",
                height=260,
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                margin=dict(t=35,b=30,l=50,r=20),
                yaxis=dict(gridcolor="#1e293b"),
                xaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})

        if run_energy:
            base_bp = to_bipolar(LETTERS[base_letter])
            noise_levels = list(range(noise_range[0], noise_range[1]+1, 5))
            recall_rates = []
            avg_energies = []

            progress = st.progress(0)
            for ni, nl in enumerate(noise_levels):
                successes, energies_at_nl = [], []
                for _ in range(n_trials):
                    noisy = add_noise(base_bp, nl)
                    hist, ehist = net_e.update_sync(noisy, steps=25)
                    recalled = hist[-1]
                    sims = {l: pattern_similarity(recalled, p) for l,p in zip(stored_ltrs_e, patterns_e)}
                    successes.append(1 if max(sims, key=sims.get)==base_letter else 0)
                    energies_at_nl.append(ehist[-1])
                recall_rates.append(np.mean(successes)*100)
                avg_energies.append(np.mean(energies_at_nl))
                progress.progress((ni+1)/len(noise_levels))

            # Recall rate vs noise
            fig_recall = go.Figure()
            fig_recall.add_trace(go.Scatter(
                x=noise_levels, y=recall_rates, mode="lines+markers",
                line=dict(color="#22c55e", width=2), marker=dict(size=7),
                name="Recall Rate %"
            ))
            fig_recall.add_hline(y=50, line_dash="dash", line_color="#f59e0b",
                                  annotation_text="50% threshold")
            fig_recall.update_layout(
                title=f"Recall Rate vs Noise Level — '{base_letter}'",
                height=260,
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                margin=dict(t=35,b=30,l=50,r=20),
                xaxis=dict(title="Noise %", gridcolor="#1e293b"),
                yaxis=dict(title="Recall Rate %", range=[0,105], gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_recall, use_container_width=True, config={"displayModeBar":False})

            fig_eng = go.Figure()
            fig_eng.add_trace(go.Scatter(
                x=noise_levels, y=avg_energies, mode="lines+markers",
                line=dict(color="#a78bfa", width=2), marker=dict(size=7),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.1)"
            ))
            fig_eng.update_layout(
                title=f"Final Energy vs Noise Level — '{base_letter}'",
                height=220,
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                margin=dict(t=35,b=30,l=50,r=20),
                xaxis=dict(title="Noise %", gridcolor="#1e293b"),
                yaxis=dict(title="Energy", gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_eng, use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════
# TAB 4 — Weight Matrix
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔬 Weight Matrix Analysis")

    if "hopfield_net" not in st.session_state:
        st.info("Train the network first.")
    else:
        net_w = st.session_state["hopfield_net"]
        W = net_w.W

        wc1, wc2 = st.columns([2,1], gap="large")
        with wc1:
            st.plotly_chart(weight_fig(W), use_container_width=True, config={"displayModeBar":False})

        with wc2:
            eigenvalues = np.linalg.eigvalsh(W)
            top_eigs = eigenvalues[::-1][:10]

            st.markdown(f"""
            <div class="nn-card">
                <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;margin-bottom:0.5rem">Matrix Stats</div>
                <table style="width:100%;font-size:0.85rem;color:#94a3b8">
                    <tr><td>Neurons (N)</td><td style="color:#e2e8f0;font-weight:700">{N_NEURONS}</td></tr>
                    <tr><td>Weights</td><td style="color:#e2e8f0;font-weight:700">{N_NEURONS**2:,}</td></tr>
                    <tr><td>Patterns stored</td><td style="color:#e2e8f0;font-weight:700">{len(st.session_state['stored_letters'])}</td></tr>
                    <tr><td>W min</td><td style="color:#e2e8f0;font-weight:700">{W.min():.4f}</td></tr>
                    <tr><td>W max</td><td style="color:#e2e8f0;font-weight:700">{W.max():.4f}</td></tr>
                    <tr><td>W mean</td><td style="color:#e2e8f0;font-weight:700">{W.mean():.4f}</td></tr>
                    <tr><td>Max |eigenvalue|</td><td style="color:#a78bfa;font-weight:700">{abs(eigenvalues).max():.3f}</td></tr>
                    <tr><td>Diagonal</td><td style="color:#22c55e;font-weight:700">All zero ✅</td></tr>
                    <tr><td>Symmetric</td><td style="color:#22c55e;font-weight:700">Yes ✅</td></tr>
                </table>
            </div>""", unsafe_allow_html=True)

        # Eigenvalue spectrum
        fig_eig = go.Figure()
        fig_eig.add_trace(go.Bar(
            x=list(range(1, len(top_eigs)+1)),
            y=top_eigs,
            marker_color=["#a78bfa" if v>0 else "#ef4444" for v in top_eigs],
            text=[f"{v:.2f}" for v in top_eigs],
            textposition="outside"
        ))
        fig_eig.update_layout(
            title="Top 10 Eigenvalues of W",
            height=260,
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            margin=dict(t=35,b=30,l=50,r=20),
            xaxis=dict(title="Rank", gridcolor="#1e293b"),
            yaxis=dict(title="λ", gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_eig, use_container_width=True, config={"displayModeBar":False})

        st.markdown("""
        <div class="nn-card">
        <b>Reading the weight matrix:</b><br>
        🟥 <b>Red</b> = strong negative weight (neurons inhibit each other) &nbsp;|&nbsp;
        🟦 <b>Blue</b> = strong positive weight (neurons excite each other)<br>
        The pattern of large eigenvalues corresponds to stored memories — each memory adds one "dimension" to the weight matrix.
        </div>""", unsafe_allow_html=True)
