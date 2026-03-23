import streamlit as st
import requests
import time

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="AI Explainer",
    layout="wide",
    page_icon="🧠"
)

# ====================== SAFE CSS ======================
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    h1, h2, h3, h4 {
        font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    .nn-card {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1.4rem !important;
        margin-bottom: 1rem !important;
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
        background: rgba(129, 140, 248, 0.15) !important;
        color: #818cf8 !important;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .chat-user {
        background: rgba(0, 212, 170, 0.12) !important;
        border: 1px solid rgba(0, 212, 170, 0.3) !important;
        border-radius: 12px;
        padding: 1rem;
        margin: 8px 0;
        margin-left: 15%;
    }
    .chat-claude {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px;
        padding: 1rem;
        margin: 8px 0;
        margin-right: 15%;
    }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🧠 AI Explainer")
    st.markdown("Neural Network Tutor")
    st.markdown("---")
    
    st.markdown("### 🔑 Anthropic API Key")
    api_key = st.text_input(
        "Enter your Anthropic API key",
        type="password",
        placeholder="sk-ant-...",
        help="Optional. Without a key you get smart demo answers."
    )
    st.caption("Get one at [console.anthropic.com](https://console.anthropic.com)")
    
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 15</div>
    <h1>AI-Powered Neural Network Tutor</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Ask anything about neural networks, deep learning, math, or code.<br>
        Get clear, expert explanations instantly.
    </p>
</div>
""", unsafe_allow_html=True)

# ====================== SUGGESTED QUESTIONS ======================
SUGGESTIONS = [
    "Explain backpropagation with a concrete numerical example",
    "Why do we need activation functions? What happens without them?",
    "What's the difference between LSTM and GRU?",
    "Explain the vanishing gradient problem and how to fix it",
    "How does batch normalization work?",
    "What is the attention mechanism?",
    "Explain the GAN training instability problem",
    "Why does deep learning need so much data?",
    "How does dropout prevent overfitting?",
    "Explain transformer architecture from scratch"
]

st.markdown("### ⚡ Quick Questions")
cols = st.columns(3)
for i, q in enumerate(SUGGESTIONS):
    with cols[i % 3]:
        if st.button(q, key=f"sugg_{i}", use_container_width=True):
            st.session_state.pending_question = q

# ====================== CHAT STATE ======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====================== CHAT DISPLAY ======================
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
                <span style="color:#00d4aa;font-size:0.75rem;font-weight:700">YOU</span><br>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-claude">
                <span style="color:#818cf8;font-size:0.75rem;font-weight:700">🧠 CLAUDE</span><br>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)

# ====================== INPUT ======================
user_input = st.text_area(
    "Ask me anything about neural networks...",
    height=120,
    placeholder="e.g. Explain backpropagation step by step with numbers...",
    key="user_input"
)

col_send, col_clear = st.columns([4, 1])
with col_send:
    send_btn = st.button("🧠 Send to Claude", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ====================== PROCESS QUESTION ======================
pending = st.session_state.pop("pending_question", None)
if pending:
    user_input = pending

if send_btn and user_input and user_input.strip():
    question = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("🧠 Thinking..."):
        if api_key and api_key.startswith("sk-ant-"):
            # Real Claude API call
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "claude-3-5-sonnet-20240620",
                        "max_tokens": 1500,
                        "system": "You are an expert neural network tutor. Explain concepts clearly with intuition, math (use LaTeX when helpful), and practical examples. Be thorough but accessible.",
                        "messages": [{"role": "user", "content": question}]
                    },
                    timeout=45
                )
                if resp.status_code == 200:
                    answer = resp.json()["content"][0]["text"]
                else:
                    answer = f"API Error {resp.status_code}. Please check your API key."
            except Exception as e:
                answer = f"Connection error: {str(e)}"
        else:
            # Smart local fallback (works without API key)
            time.sleep(0.8)  # simulate thinking
            fallback_answers = {
                "backpropagation": "Backpropagation is the chain rule applied backwards. For a simple network: Loss = (ŷ - y)² → ∂Loss/∂w = (ŷ - y)·x. We multiply gradients layer by layer from output to input.",
                "activation": "Without activation functions the entire network collapses to a single linear transformation. Activations (ReLU, Tanh, etc.) introduce non-linearity, allowing the network to learn complex patterns.",
                "lstm vs gru": "LSTM has three gates (forget, input, output) + cell state. GRU has only two gates and merges cell/hidden state. GRU is faster and uses fewer parameters; LSTM is slightly more powerful on very long sequences.",
                "vanishing gradient": "Gradients become extremely small as they backpropagate through many layers (especially with sigmoid/tanh). Solutions: ReLU, residual connections, batch norm, better initialization.",
            }
            q_lower = question.lower()
            answer = fallback_answers.get("backpropagation" if "backprop" in q_lower else
                                         "activation" if "activat" in q_lower else
                                         "lstm" if "lstm" in q_lower or "gru" in q_lower else
                                         "vanishing" if "vanish" in q_lower else None,
                                         "Here's a detailed explanation of your question...\n\n" +
                                         "Neural networks are universal function approximators. The key ideas are:\n" +
                                         "• Non-linearity from activation functions\n" +
                                         "• Backpropagation for learning\n" +
                                         "• Architectural choices (CNNs, RNNs, Transformers)\n\n" +
                                         "Would you like me to go deeper into any specific part?")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

# ====================== WELCOME MESSAGE ======================
if not st.session_state.chat_history:
    st.markdown("""
    <div class="nn-card" style="text-align:center;padding:2rem">
        <span style="font-size:2.5rem">🧠</span><br><br>
        <b>Ask me anything about neural networks!</b><br>
        <span style="color:var(--muted)">Examples above or type your question below</span>
    </div>
    """, unsafe_allow_html=True)
