"""Sidebar navigation for NeuralForge Ultra."""
import streamlit as st


PAGES = [
    ("🏠", "Home",               "app"),
    ("⬡", "Perceptron",         "01"),
    ("⟶", "Forward Pass",       "02"),
    ("↺", "Backpropagation",    "03"),
    ("↗", "Gradient Descent",   "04"),
    ("⬛", "ANN / MLP",          "05"),
    ("◫", "CNN",                "06"),
    ("⇌", "RNN / LSTM",         "07"),
    ("◎", "Autoencoder / VAE",  "08"),
    ("◉", "OpenCV Vision",      "09"),
    ("⚡", "Transformer Attn",  "10"),
    ("🎮", "GAN Lab",            "11"),
    ("🤖", "RL Agent",           "12"),
    ("🧬", "NAS Explorer",       "13"),
    ("📊", "Model Comparison",   "14"),
    ("🧠", "AI Explainer",       "15"),
]

THEMES = ["dark", "cyberpunk", "light"]
THEME_ICONS = {"dark": "🌑", "cyberpunk": "⚡", "light": "☀️"}


def render_sidebar(current: str) -> None:
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:1rem 0 0.5rem;text-align:center">
          <span style="font-size:2rem">🧠</span>
          <div style="font-family:'IBM Plex Sans',sans-serif;font-weight:800;
                      font-size:1.1rem;color:var(--accent);margin-top:0.2rem">
            NeuralForge Ultra
          </div>
          <div style="font-size:0.7rem;color:var(--muted);letter-spacing:0.1em">v3.0 · 16 MODULES</div>
        </div>
        """, unsafe_allow_html=True)

        # Theme switcher
        st.markdown("---")
        t = st.session_state["theme"]
        cols = st.columns(3)
        for i, th in enumerate(THEMES):
            with cols[i]:
                label = f"{THEME_ICONS[th]} {th.capitalize()}"
                if st.button(label, key=f"theme_{th}",
                             type="primary" if t == th else "secondary",
                             use_container_width=True):
                    st.session_state["theme"] = th
                    st.rerun()

        st.markdown("---")
        st.markdown('<div style="font-size:0.7rem;color:var(--muted);'
                    'letter-spacing:0.1em;margin-bottom:0.5rem">MODULES</div>',
                    unsafe_allow_html=True)

        for icon, name, _ in PAGES:
            is_active = (name == current)
            style = ("background:color-mix(in srgb, var(--accent) 15%, transparent);"
                     "border-left:3px solid var(--accent);") if is_active else ""
            st.markdown(
                f'<div style="padding:0.45rem 0.75rem;border-radius:8px;'
                f'font-size:0.88rem;cursor:pointer;{style}">'
                f'{icon} {name}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown('<div style="font-size:0.68rem;color:var(--muted);text-align:center">'
                    'Built with ❤️ · NeuralForge Ultra</div>', unsafe_allow_html=True)
