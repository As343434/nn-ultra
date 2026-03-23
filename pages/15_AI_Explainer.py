"""AI Explainer — Claude-powered neural network tutor."""
import streamlit as st
import requests
import json

from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, card

st.set_page_config(page_title="AI Explainer", layout="wide", page_icon="🧠")
apply_theme()
render_sidebar("AI Explainer")

hero(
    "AI-Powered NN Explainer",
    "Powered by Claude — ask anything about neural networks, deep learning, math, or code. Get clear, expert explanations instantly.",
    pill="Lesson 15", pill_variant="",
)

TEAL = "#00d4aa"; SURFACE = "#161b22"

# ── Suggested questions ────────────────────────────────────────────────────────
SUGGESTIONS = [
    "Explain backpropagation with a concrete numerical example",
    "Why do we need activation functions? What happens without them?",
    "What's the difference between LSTM and GRU? When to use each?",
    "Explain the vanishing gradient problem and how to fix it",
    "How does batch normalization work and why does it help?",
    "What is attention mechanism and why did it revolutionize NLP?",
    "Explain the GAN training instability problem and solutions",
    "What's the difference between L1 and L2 regularization?",
    "How does dropout prevent overfitting?",
    "Explain transformer architecture from scratch",
    "What is the curse of dimensionality?",
    "Why does deep learning need so much data?",
]

# ── Chat state ─────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### ⚡ Quick Questions")
    for q in SUGGESTIONS:
        if st.button(q, key=f"sugg_{q[:20]}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    card("""
    <b style="color:var(--accent)">What I can explain:</b><br>
    • Math & theory behind any NN concept<br>
    • Code walkthroughs & examples<br>
    • Comparisons between architectures<br>
    • Debugging tips & best practices<br>
    • Paper summaries & intuitions<br><br>
    <span style="color:var(--muted);font-size:0.8rem">Powered by Claude claude-sonnet-4-20250514</span>
    """)

with col1:
    # Chat display
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.2);
                            border-radius:12px;padding:1rem;margin:0.5rem 0;
                            margin-left:10%">
                    <span style="color:var(--accent);font-size:0.75rem;font-weight:700">YOU</span><br>
                    <span style="color:var(--text)">{msg["content"]}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);
                            border-radius:12px;padding:1rem;margin:0.5rem 0;
                            margin-right:10%">
                    <span style="color:var(--accent2);font-size:0.75rem;font-weight:700">🧠 CLAUDE</span><br>
                    <span style="color:var(--text)">{msg["content"]}</span>
                </div>
                """, unsafe_allow_html=True)

    # Input
    st.markdown("---")
    pending = st.session_state.pop("pending_question", "")
    user_input = st.text_area(
        "Ask anything about neural networks...",
        value=pending,
        height=100,
        placeholder="e.g. Explain backpropagation step by step with math...",
        key="chat_input"
    )

    col_send, col_clear = st.columns([3, 1])
    with col_send:
        send_btn = st.button("🧠 Ask Claude", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

        # Build messages for API
        system_prompt = """You are an expert neural network tutor and deep learning researcher. 
You explain concepts clearly with intuition, math, and practical examples.
Use markdown formatting: headers, bold, code blocks, bullet points.
When explaining math, write it clearly with proper notation.
Be thorough but accessible. Use analogies when helpful.
Focus on building deep understanding, not just surface-level answers."""

        messages = []
        for msg in st.session_state.chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        with st.spinner("🧠 Claude is thinking..."):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 2000,
                        "system": system_prompt,
                        "messages": messages,
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["content"][0]["text"]
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
                else:
                    st.error(f"API Error {response.status_code}: {response.text[:200]}")
                    st.info("💡 **Tip:** The Claude API requires authentication. In a deployed app, add your Anthropic API key as an environment variable `ANTHROPIC_API_KEY`.")

            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the app is running in an environment with Anthropic API access.")

    # ── Fallback: show sample explanation if no API ────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);
                    border-radius:12px;padding:1.5rem;margin:1rem 0;opacity:0.7">
            <span style="color:var(--accent2);font-size:0.75rem;font-weight:700">🧠 CLAUDE (example)</span><br>
            <span style="color:var(--muted);font-size:0.9rem">
            Ask me anything! For example:<br><br>
            <b style="color:var(--text)">"Explain backpropagation"</b> → I'll walk through the chain rule with numbers<br>
            <b style="color:var(--text)">"Why use ReLU over sigmoid?"</b> → I'll explain vanishing gradients<br>
            <b style="color:var(--text)">"Write me a transformer in PyTorch"</b> → Full annotated code<br>
            <b style="color:var(--text)">"Why did my model overfit?"</b> → Diagnosis + fixes<br>
            </span>
        </div>
        """, unsafe_allow_html=True)
