"""RL Agent — DQN on GridWorld with Q-value heatmap and policy visualization."""
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row, card
from utils.viz import plot_reward_curve, _dark_layout

st.set_page_config(page_title="RL Agent", layout="wide", page_icon="🤖")
apply_theme()
render_sidebar("RL Agent")

hero(
    "Reinforcement Learning Agent",
    "Train a tabular Q-Learning or DQN agent on GridWorld. Watch it learn to navigate, visualize Q-values and the policy in real time.",
    pill="Lesson 12", pill_variant="",
)

TEAL = "#00d4aa"; ORANGE = "#f97316"; PURPLE = "#818cf8"; SURFACE = "#161b22"; BG = "#0d1117"

# ── GridWorld ─────────────────────────────────────────────────────────────────
class GridWorld:
    ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]  # R, L, D, U
    ACTION_NAMES = ["→", "←", "↓", "↑"]

    def __init__(self, size=6, n_obstacles=5, seed=42):
        self.size = size
        np.random.seed(seed)
        self.start = (0, 0)
        self.goal  = (size-1, size-1)
        self.obstacles = set()
        while len(self.obstacles) < n_obstacles:
            r, c = np.random.randint(0, size, 2)
            if (r, c) not in [self.start, self.goal]:
                self.obstacles.add((r, c))
        self.reset()

    def reset(self):
        self.pos = self.start
        return self._state()

    def _state(self):
        return self.pos[0] * self.size + self.pos[1]

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
            self.pos = (nr, nc)
        if self.pos == self.goal:
            return self._state(), 10.0, True
        elif self.pos in self.obstacles:
            return self._state(), -5.0, False
        return self._state(), -0.1, False

    def n_states(self):
        return self.size * self.size

    def n_actions(self):
        return 4


# ── Config ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Config")
    grid_size  = st.slider("Grid size", 4, 10, 6)
    n_obs      = st.slider("Obstacles", 0, 10, 4)
    episodes   = st.slider("Episodes", 100, 5000, 1000, 100)
    alpha      = st.slider("Learning rate (α)", 0.01, 1.0, 0.3, 0.01)
    gamma      = st.slider("Discount (γ)", 0.5, 0.999, 0.95, 0.001)
    epsilon_i  = st.slider("ε-initial", 0.1, 1.0, 1.0, 0.05)
    epsilon_f  = st.slider("ε-final", 0.0, 0.3, 0.05, 0.01)
    env_seed   = st.number_input("World seed", 0, 100, 42)
    train_btn  = st.button("🤖 Train Agent", type="primary")

with col2:
    st.markdown("### 🗺️ GridWorld Preview")
    env_prev = GridWorld(grid_size, n_obs, int(env_seed))

    grid_html = '<div style="display:inline-block;border:2px solid var(--border);border-radius:8px;padding:4px">'
    for r in range(grid_size):
        grid_html += '<div style="display:flex">'
        for c in range(grid_size):
            if (r,c) == env_prev.start:
                bg, sym = "#1a472a", "🟢"
            elif (r,c) == env_prev.goal:
                bg, sym = "#4a1942", "⭐"
            elif (r,c) in env_prev.obstacles:
                bg, sym = "#3a1010", "🔴"
            else:
                bg, sym = "#161b22", "⬜"
            grid_html += f'<div style="width:36px;height:36px;background:{bg};display:flex;align-items:center;justify-content:center;font-size:14px;border:1px solid #30363d;border-radius:4px;margin:1px">{sym}</div>'
        grid_html += '</div>'
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    card("""
    🟢 <b>Start</b> · ⭐ <b>Goal</b> (+10 reward) · 🔴 <b>Obstacle</b> (-5 reward) · ⬜ <b>Free</b> (-0.1 per step)<br><br>
    <b style="color:var(--accent)">Q-Learning update:</b><br>
    <code>Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]</code>
    """)

st.markdown("---")

if train_btn:
    env = GridWorld(grid_size, n_obs, int(env_seed))
    Q = np.zeros((env.n_states(), env.n_actions()))
    
    rewards_per_ep = []
    success_per_ep = []
    epsilons = np.linspace(epsilon_i, epsilon_f, episodes)

    bar = st.progress(0)
    status = st.empty()
    chart_ph = st.empty()

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        epsilon = epsilons[ep]
        steps = 0

        while not done and steps < grid_size * grid_size * 3:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions())
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            ep_reward += reward
            steps += 1

        rewards_per_ep.append(ep_reward)
        success_per_ep.append(1.0 if ep_reward > 5 else 0.0)
        bar.progress(int((ep+1)/episodes * 100))

        if ep % 100 == 0:
            smooth = np.convolve(rewards_per_ep, np.ones(min(50, len(rewards_per_ep)))/min(50, len(rewards_per_ep)), mode='valid')
            sr = np.mean(success_per_ep[-100:]) if len(success_per_ep) >= 100 else np.mean(success_per_ep)
            status.markdown(f'<span class="status-badge status-running">Ep {ep} | ε={epsilon:.3f} | Success rate: {sr:.1%}</span>', unsafe_allow_html=True)

    status.markdown('<span class="status-badge status-done">✓ Training complete!</span>', unsafe_allow_html=True)

    # ── Results ──────────────────────────────────────────────────────────────
    smooth_r = np.convolve(rewards_per_ep, np.ones(50)/50, mode='valid')
    final_sr = np.mean(success_per_ep[-100:])

    metric_row([
        ("Episodes", episodes),
        ("Success Rate", f"{final_sr:.1%}"),
        ("Best Reward", f"{max(rewards_per_ep):.1f}"),
        ("Final ε", f"{epsilon_f:.3f}"),
    ])

    col_a, col_b = st.columns(2)

    with col_a:
        fig = plot_reward_curve(rewards_per_ep, list(smooth_r) + [None]*(episodes - len(smooth_r)))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Q-value heatmap (max Q per state)
        max_q = np.max(Q, axis=1).reshape(grid_size, grid_size)
        # Mask obstacles
        for r, c in env.obstacles:
            max_q[r, c] = np.nan

        fig_q = px.imshow(max_q, color_continuous_scale="RdYlGn", aspect="equal",
                          text_auto=".1f")
        fig_q.update_layout(paper_bgcolor=SURFACE, font=dict(color="#8b949e"),
                             height=300, margin=dict(l=10,r=10,t=40,b=10),
                             title=dict(text="Q-Value Heatmap (max over actions)",
                                        font=dict(color="#e6edf3")))
        st.plotly_chart(fig_q, use_container_width=True)

    # ── Policy grid visualization ─────────────────────────────────────────────
    st.markdown("### 🗺️ Learned Policy")
    policy = np.argmax(Q, axis=1).reshape(grid_size, grid_size)
    policy_html = '<div style="display:inline-block;border:2px solid var(--border);border-radius:8px;padding:4px">'

    for r in range(grid_size):
        policy_html += '<div style="display:flex">'
        for c in range(grid_size):
            if (r,c) == env.goal:
                bg, sym = "#2d1a4f", "⭐"
            elif (r,c) in env.obstacles:
                bg, sym = "#3a1010", "🔴"
            elif (r,c) == env.start:
                bg, sym = "#1a3a2a", GridWorld.ACTION_NAMES[policy[r,c]]
            else:
                q_val = max_q[r,c]
                if not np.isnan(q_val):
                    intensity = min(255, max(0, int((q_val + 5) / 15 * 255)))
                    bg = f"rgba(0, {intensity}, {intensity//2}, 0.3)"
                else:
                    bg = "#161b22"
                sym = GridWorld.ACTION_NAMES[policy[r,c]]
            policy_html += f'<div style="width:44px;height:44px;background:{bg};display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;border:1px solid #30363d;border-radius:4px;margin:1px;color:#00d4aa">{sym}</div>'
        policy_html += '</div>'
    policy_html += '</div>'

    col_p, col_info = st.columns([1, 1])
    with col_p:
        st.markdown(policy_html, unsafe_allow_html=True)
    with col_info:
        card(f"""
        <b style="color:var(--accent)">Policy arrows show the best action per state</b><br><br>
        Darker teal = higher Q-value<br>
        The agent learned to navigate around obstacles to reach ⭐<br><br>
        <b>Final stats:</b><br>
        • Success rate (last 100 eps): <b>{final_sr:.1%}</b><br>
        • Max Q-value: <b>{np.nanmax(max_q):.2f}</b><br>
        • States with positive Q: <b>{np.sum(max_q > 0 if not np.isnan(max_q).all() else 0)}</b>
        """)

    # ── Convergence analysis ──────────────────────────────────────────────────
    with st.expander("📈 Convergence Analysis"):
        window = 100
        rolling_success = [np.mean(success_per_ep[max(0,i-window):i+1])
                           for i in range(len(success_per_ep))]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=rolling_success, mode="lines",
                                  name="Success Rate (rolling 100)",
                                  line=dict(color=TEAL, width=2)))
        fig3.add_hline(y=0.9, line_dash="dot", line_color=ORANGE,
                       annotation_text="90% target")
        fig3.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=BG,
                            font=dict(color="#8b949e"), height=280,
                            margin=dict(l=10,r=10,t=30,b=10),
                            title=dict(text="Agent Success Rate Over Training",
                                       font=dict(color="#e6edf3")))
        fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)", tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("👈 Configure your agent and click **🤖 Train Agent**")

    with st.expander("📚 Q-Learning Theory"):
        card("""
        <b style="color:var(--accent)">The Bellman Equation:</b><br>
        <code>Q*(s,a) = 𝔼[r + γ · max_a' Q*(s',a')]</code><br><br>
        <b style="color:var(--accent2)">Key concepts:</b><br>
        • <b>ε-greedy exploration</b>: explore randomly with probability ε, exploit best known action otherwise<br>
        • <b>Discount factor γ</b>: how much future rewards matter (0=myopic, 1=infinite horizon)<br>
        • <b>Learning rate α</b>: how fast to update Q estimates<br><br>
        <b style="color:var(--accent3)">From Q-Learning to DQN:</b><br>
        Deep Q-Networks replace the Q-table with a neural network, enabling generalization to continuous/high-dimensional state spaces.
        Experience replay and target networks stabilize training.
        """)
