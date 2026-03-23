import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="RL Agent — GridWorld",
    layout="wide",
    page_icon="🤖"
)

# ====================== SAFE CSS ======================
st.markdown("""
<style>
    section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
    h1, h2, h3, h4 { font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
    .nn-card { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 1.4rem !important; margin-bottom: 1rem !important; }
    .nn-hero { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 16px !important; padding: 2.2rem 2rem !important; margin-bottom: 1.6rem !important; }
    .nn-pill { display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px; background: rgba(129, 140, 248, 0.15) !important; color: #818cf8 !important; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .status-running { background: rgba(249,115,22,0.2); color: #f97316; padding: 0.4rem 0.8rem; border-radius: 999px; font-weight: 600; }
    .status-done   { background: rgba(34,197,151,0.2); color: #22c597; padding: 0.4rem 0.8rem; border-radius: 999px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🤖 RL Agent")
    st.markdown("Q-Learning on GridWorld")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 12</div>
    <h1>RL Agent — GridWorld</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Train a Q-Learning agent to navigate a dangerous grid.<br>
        Watch Q-values and policy evolve in real time.
    </p>
</div>
""", unsafe_allow_html=True)

# ====================== GRIDWORLD CLASS ======================
class GridWorld:
    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]   # Right, Left, Down, Up
    ACTION_NAMES = ["→", "←", "↓", "↑"]

    def __init__(self, size=6, n_obstacles=4, seed=42):
        self.size = size
        np.random.seed(seed)
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacles = set()
        while len(self.obstacles) < n_obstacles:
            r, c = np.random.randint(0, size, 2)
            if (r, c) not in (self.start, self.goal):
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
        reward = 10.0 if self.pos == self.goal else (-5.0 if self.pos in self.obstacles else -0.1)
        done = self.pos == self.goal
        return self._state(), reward, done

    def n_states(self):
        return self.size * self.size

    def n_actions(self):
        return 4

# ====================== CONFIG ======================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Configuration")
    grid_size = st.slider("Grid Size", 4, 10, 6)
    n_obstacles = st.slider("Number of Obstacles", 0, 12, 4)
    episodes = st.slider("Training Episodes", 100, 5000, 800, 100)
    alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.3, 0.01)
    gamma = st.slider("Discount Factor (γ)", 0.5, 0.999, 0.95, 0.001)
    epsilon_start = st.slider("ε Start", 0.1, 1.0, 1.0, 0.05)
    epsilon_end = st.slider("ε End", 0.0, 0.3, 0.05, 0.01)
    seed = st.number_input("Environment Seed", 0, 100, 42)
    train_btn = st.button("🤖 Train Agent", type="primary", use_container_width=True)

with col2:
    st.markdown("### 🗺️ GridWorld Preview")
    preview_env = GridWorld(grid_size, n_obstacles, int(seed))
    html = '<div style="display:inline-block; border:2px solid var(--border); border-radius:8px; padding:6px; background:#0d1117;">'
    for r in range(grid_size):
        html += '<div style="display:flex;">'
        for c in range(grid_size):
            if (r, c) == preview_env.start:
                color, symbol = "#1a472a", "🟢"
            elif (r, c) == preview_env.goal:
                color, symbol = "#4a1942", "⭐"
            elif (r, c) in preview_env.obstacles:
                color, symbol = "#3a1010", "🔴"
            else:
                color, symbol = "#161b22", "⬜"
            html += f'<div style="width:38px;height:38px;background:{color};margin:1px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:18px;border:1px solid #30363d;">{symbol}</div>'
        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("""
    <div class="nn-card">
        🟢 Start &nbsp; ⭐ Goal (+10) &nbsp; 🔴 Obstacle (-5) &nbsp; ⬜ Free (-0.1/step)<br><br>
        <b>Q-Learning Update:</b><br>
        <code>Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') − Q(s,a) ]</code>
    </div>
    """, unsafe_allow_html=True)

# ====================== TRAINING ======================
if train_btn:
    env = GridWorld(grid_size, n_obstacles, int(seed))
    Q = np.zeros((env.n_states(), env.n_actions()))

    rewards_history = []
    success_history = []
    epsilons = np.linspace(epsilon_start, epsilon_end, episodes)

    bar = st.progress(0)
    status = st.empty()

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        epsilon = epsilons[ep]
        steps = 0

        while not done and steps < grid_size * grid_size * 4:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions())
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)

            # Q-Learning update
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            ep_reward += reward
            steps += 1

        rewards_history.append(ep_reward)
        success_history.append(1.0 if ep_reward > 5 else 0.0)

        bar.progress((ep + 1) / episodes)

        if ep % 50 == 0 or ep == episodes - 1:
            recent_success = np.mean(success_history[-100:]) if len(success_history) >= 100 else np.mean(success_history)
            status.markdown(
                f'<span class="status-running">Episode {ep} | ε={epsilon:.3f} | Success: {recent_success:.1%}</span>',
                unsafe_allow_html=True
            )

    status.markdown('<span class="status-done">✓ Training Complete!</span>', unsafe_allow_html=True)

    # ====================== RESULTS ======================
    final_success = np.mean(success_history[-100:])

    st.markdown("---")
    st.markdown("### 📊 Training Results")
    metric_row([
        ("Episodes", episodes),
        ("Success Rate (last 100)", f"{final_success:.1%}"),
        ("Best Episode Reward", f"{max(rewards_history):.1f}"),
        ("Final ε", f"{epsilon_end:.3f}")
    ])

    col1, col2 = st.columns(2)

    with col1:
        # Reward Curve
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(y=rewards_history, name="Episode Reward", line=dict(color="#00d4aa")))
        # Smoothed
        window = 50
        if len(rewards_history) > window:
            smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
            fig_reward.add_trace(go.Scatter(y=smoothed, name="Smoothed", line=dict(color="#f97316", width=3)))
        fig_reward.update_layout(title="Reward per Episode", height=340, paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
        st.plotly_chart(fig_reward, use_container_width=True)

    with col2:
        # Q-Value Heatmap
        max_q = np.max(Q, axis=1).reshape(grid_size, grid_size)
        for r, c in env.obstacles:
            max_q[r, c] = np.nan

        fig_q = px.imshow(max_q, color_continuous_scale="RdYlGn", text_auto=".1f", aspect="equal")
        fig_q.update_layout(title="Q-Value Heatmap (max Q per state)", height=340, paper_bgcolor="#161b22")
        st.plotly_chart(fig_q, use_container_width=True)

    # Policy Visualization
    st.markdown("### 🗺️ Learned Policy")
    policy = np.argmax(Q, axis=1).reshape(grid_size, grid_size)

    policy_html = '<div style="display:inline-block; border:2px solid var(--border); border-radius:8px; padding:8px; background:#0d1117;">'
    for r in range(grid_size):
        policy_html += '<div style="display:flex;">'
        for c in range(grid_size):
            if (r, c) == env.goal:
                bg, sym = "#4a1942", "⭐"
            elif (r, c) in env.obstacles:
                bg, sym = "#3a1010", "🔴"
            else:
                qval = max_q[r, c]
                intensity = int(min(255, max(0, (qval + 5) / 15 * 255))) if not np.isnan(qval) else 0
                bg = f"rgba(0, {intensity}, {intensity//2}, 0.35)"
                sym = GridWorld.ACTION_NAMES[policy[r, c]]
            policy_html += f'<div style="width:46px;height:46px;background:{bg};margin:2px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;border:1px solid #30363d;color:#00d4aa">{sym}</div>'
        policy_html += '</div>'
    policy_html += '</div>'

    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.markdown(policy_html, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class="nn-card">
            <b>Policy Map Legend:</b><br>
            → ← ↓ ↑ = Best action in that state<br><br>
            <b>Final Success Rate:</b> {final_success:.1%}<br>
            <b>Max Q-value:</b> {np.nanmax(max_q):.2f}<br>
            The agent has learned to avoid obstacles and reach the goal ⭐ efficiently.
        </div>
        """, unsafe_allow_html=True)

    st.balloons()

else:
    st.info("👈 Configure the environment and click **🤖 Train Agent** to start training.")
    
    with st.expander("📚 Q-Learning Theory"):
        st.markdown("""
        **Q-Learning** is an off-policy temporal difference algorithm.

        The agent learns the optimal action-value function Q*(s,a) using the Bellman equation:

        **Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]**

        - **ε-greedy**: Balances exploration vs exploitation
        - **γ (discount)**: Values future rewards
        - **α (learning rate)**: Controls update speed
        """)
