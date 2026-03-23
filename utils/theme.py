"""
NeuralForge Ultra — Multi-theme system with cyberpunk, dark, and light modes.
"""
import streamlit as st

THEMES = {
    "cyberpunk": {
        "--bg":       "#0a0a0f",
        "--surface":  "#111118",
        "--surface2": "#18181f",
        "--border":   "rgba(255,0,255,0.15)",
        "--text":     "#f0f0ff",
        "--muted":    "#8888aa",
        "--accent":   "#ff00ff",
        "--accent2":  "#00ffff",
        "--accent3":  "#ffff00",
        "--danger":   "#ff3366",
        "--success":  "#00ff99",
        "--glow":     "0 0 20px rgba(255,0,255,0.4), 0 0 40px rgba(0,255,255,0.2)",
    },
    "dark": {
        "--bg":       "#0d1117",
        "--surface":  "#161b22",
        "--surface2": "#1c2333",
        "--border":   "rgba(255,255,255,0.08)",
        "--text":     "#e6edf3",
        "--muted":    "#8b949e",
        "--accent":   "#00d4aa",
        "--accent2":  "#f97316",
        "--accent3":  "#818cf8",
        "--danger":   "#f85149",
        "--success":  "#3fb950",
        "--glow":     "0 0 20px rgba(0,212,170,0.3)",
    },
    "light": {
        "--bg":       "#f6f8fa",
        "--surface":  "#ffffff",
        "--surface2": "#f0f2f5",
        "--border":   "rgba(0,0,0,0.1)",
        "--text":     "#1a1a2e",
        "--muted":    "#6e7681",
        "--accent":   "#0066cc",
        "--accent2":  "#e84000",
        "--accent3":  "#6f42c1",
        "--danger":   "#cf222e",
        "--success":  "#1a7f37",
        "--glow":     "0 0 20px rgba(0,102,204,0.2)",
    },
}

BASE_CSS = """
:root {{
  {vars}
  --card-shadow: 0 4px 24px rgba(0,0,0,0.4);
  --radius: 12px;
  --radius-lg: 20px;
}}

html, body, [data-testid="stApp"] {{
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}}

section[data-testid="stSidebar"] {{
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}}

h1, h2, h3, h4 {{
  font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
}}

.nn-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.4rem;
  margin-bottom: 1rem;
  box-shadow: var(--card-shadow);
  transition: border-color 0.2s, box-shadow 0.2s;
}}
.nn-card:hover {{
  border-color: var(--accent);
  box-shadow: var(--glow);
}}

.nn-card-accent {{
  background: linear-gradient(135deg,
    color-mix(in srgb, var(--accent) 8%, transparent) 0%,
    color-mix(in srgb, var(--accent3) 6%, transparent) 50%,
    color-mix(in srgb, var(--accent2) 5%, transparent) 100%);
  border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
  border-radius: var(--radius-lg);
  padding: 1.4rem;
  margin-bottom: 1rem;
}}

.nn-hero {{
  background: radial-gradient(ellipse 80% 60% at 50% 0%,
    color-mix(in srgb, var(--accent) 15%, transparent) 0%,
    color-mix(in srgb, var(--accent3) 8%, transparent) 40%,
    transparent 70%),
    var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2.2rem 2rem;
  margin-bottom: 1.6rem;
  position: relative;
  overflow: hidden;
}}

.nn-pill {{
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 15%, transparent);
  color: var(--accent);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}}
.nn-pill-orange {{ background: color-mix(in srgb, var(--accent2) 15%, transparent); color: var(--accent2); }}
.nn-pill-purple {{ background: color-mix(in srgb, var(--accent3) 15%, transparent); color: var(--accent3); }}

.nn-metric {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  text-align: center;
  transition: box-shadow 0.2s;
}}
.nn-metric:hover {{ box-shadow: var(--glow); }}
.nn-metric-value {{
  font-size: 1.8rem;
  font-weight: 800;
  color: var(--accent);
  font-family: 'IBM Plex Sans', sans-serif;
}}
.nn-metric-label {{
  font-size: 0.75rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.2rem;
}}

.status-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
}}
.status-running {{ background: color-mix(in srgb, var(--accent2) 20%, transparent); color: var(--accent2); }}
.status-done    {{ background: color-mix(in srgb, var(--success) 20%, transparent); color: var(--success); }}
.status-idle    {{ background: color-mix(in srgb, var(--muted) 20%, transparent); color: var(--muted); }}

/* Cyberpunk scanline effect */
.cyber-scanlines::after {{
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,0.05) 2px, rgba(0,0,0,0.05) 4px
  );
  pointer-events: none;
  z-index: 9999;
}}

[data-testid="stMetric"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.8rem 1rem;
}}
[data-testid="stMetricValue"] {{ color: var(--accent) !important; }}

div[data-testid="stExpander"] > details {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}}

button[kind="primary"] {{
  background: var(--accent) !important;
  color: #000 !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
  box-shadow: 0 0 12px color-mix(in srgb, var(--accent) 40%, transparent) !important;
}}
button[kind="secondary"] {{
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  border-radius: 8px !important;
}}

[data-testid="stProgress"] > div > div {{
  background: var(--accent) !important;
}}

hr {{
  border-color: var(--border) !important;
  margin: 1.5rem 0 !important;
}}
"""

FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
"""

def apply_theme() -> None:
    t = st.session_state.get("theme", "dark")
    vars_str = "\n  ".join(f"{k}: {v};" for k, v in THEMES[t].items())
    css = BASE_CSS.format(vars=vars_str)
    st.markdown(FONTS + f"<style>{css}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, pill: str = "", pill_variant: str = "") -> None:
    pill_cls = f"nn-pill-{pill_variant}" if pill_variant else "nn-pill"
    pill_html = f'<span class="{pill_cls}">{pill}</span><br>' if pill else ""
    st.markdown(f"""
    <div class="nn-hero">
      {pill_html}
      <h1 style="margin:0.3rem 0 0.5rem;font-size:1.8rem">{title}</h1>
      <p style="color:var(--muted);margin:0;font-size:0.97rem;font-family:'IBM Plex Sans',sans-serif">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_row(metrics: list) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="nn-metric">
              <div class="nn-metric-value">{value}</div>
              <div class="nn-metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)


def card(content: str, variant: str = "default") -> None:
    cls = "nn-card-accent" if variant == "accent" else "nn-card"
    st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)
