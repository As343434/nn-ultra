# 🧠 NeuralForge Ultra — Neural Network Toolbox v3.0

> The most advanced interactive Neural Network Toolbox — 16 modules, 3 themes, AI-powered explanations, GAN Lab, RL Agent, Transformer Attention, Neural Architecture Search, and more.

---

## 🆚 NeuralForge Ultra vs Shabnam's NeuralViz Pro

| Feature | NeuralViz Pro (Shabnam) | NeuralForge Ultra (Yours) |
|---|---|---|
| Modules | 9 | **16** |
| Themes | 1 (dark) | **3** (dark, cyberpunk, light) |
| Activations | 8 | **10** (+ Mish, ELU) |
| Transformer Attention | ❌ | ✅ Multi-head, QKV, Positional Encoding |
| GAN Lab | ❌ | ✅ Train GAN in-browser, mode collapse detector |
| RL Agent | ❌ | ✅ Q-Learning on GridWorld, policy viz |
| Neural Arch Search | ❌ | ✅ Pareto frontier, efficiency explorer |
| Model Comparison | ❌ | ✅ Radar chart, loss curves, benchmark |
| AI Explainer | ❌ | ✅ Claude-powered NN tutor |
| Architecture viz | ❌ | ✅ Interactive graph |
| Theme switcher | ❌ | ✅ Live in sidebar |
| Attention maps | ❌ | ✅ Per-head + averaged |
| Loss surface types | 1 | **5** (saddle, narrow valley, noisy...) |
| GAN distributions | ❌ | ✅ 5 (ring, grid, mixture, banana, spiral) |
| RL environments | ❌ | ✅ GridWorld with obstacles |
| Mode collapse detector | ❌ | ✅ |

---

## 🗂️ All 16 Modules

| # | Module | What you'll learn |
|---|---|---|
| 1 | ⬡ Perceptron | Single-neuron classifier, decision boundary, weight updates |
| 2 | ⟶ Forward Pass | 10 activation functions, layer math, derivatives |
| 3 | ↺ Backpropagation | Chain rule, gradient flow, MSE & BCE loss |
| 4 | ↗ Gradient Descent | GD/SGD/Momentum/Adam/RMSProp on 3D surface |
| 5 | ⬛ ANN / MLP | Configurable MLP, NumPy + PyTorch, custom CSV |
| 6 | ◫ CNN | Conv-net on MNIST/Fashion-MNIST, filter viewer |
| 7 | ⇌ RNN / LSTM / GRU | Sequence modeling, hidden-state heatmap |
| 8 | ◎ Autoencoder / VAE | Latent space, denoising, reconstruction |
| 9 | ◉ OpenCV Vision | 15 preprocessing ops, Fourier transform |
| 10 | ⚡ Transformer Attn | Multi-head attention, QKV, positional encoding |
| 11 | 🎮 GAN Lab | Train DCGAN, mode collapse detection |
| 12 | 🤖 RL Agent | Q-Learning on GridWorld, policy visualization |
| 13 | 🧬 NAS Explorer | Neural Architecture Search, Pareto frontier |
| 14 | 📊 Model Comparison | Radar chart benchmark, loss curves |
| 15 | 🧠 AI Explainer | Claude-powered deep learning tutor |
| 16 | 📤 Export Hub | PyTorch, ONNX, pickle, Python export |

---

## 🚀 Quick Start

```bash
# 1. Unzip / clone
cd nn_ultra

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run!
streamlit run app.py
```

---

## 🎨 Themes

Switch themes live in the sidebar:
- 🌑 **Dark** — professional dark with teal/orange accents
- ⚡ **Cyberpunk** — magenta/cyan neon with glow effects
- ☀️ **Light** — clean light mode for presentations

---

## 🧠 AI Explainer Setup

The AI Explainer (Module 15) uses the Claude API. To enable it in your deployed app:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Or add it to `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-key-here"
```

---

## 📦 Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

---

## 🏗️ Project Structure

```
nn_ultra/
├── app.py                    # Home dashboard
├── requirements.txt
├── README.md
├── utils/
│   ├── theme.py              # 3-theme system with CSS variables
│   ├── nav.py                # Sidebar navigation + theme switcher
│   ├── viz.py                # 15+ chart builders (Plotly + Matplotlib)
│   ├── data.py               # Dataset loaders
│   └── export.py             # pickle / torch / csv export
└── pages/
    ├── 01_Perceptron.py
    ├── 02_Forward_Pass.py
    ├── 03_Backpropagation.py
    ├── 04_Gradient_Descent.py
    ├── 05_ANN_MLP.py
    ├── 06_CNN.py
    ├── 07_RNN_LSTM.py
    ├── 08_Autoencoder.py
    ├── 09_OpenCV_Vision.py
    ├── 10_Transformer_Attn.py  ← NEW
    ├── 11_GAN_Lab.py           ← NEW
    ├── 12_RL_Agent.py          ← NEW
    ├── 13_NAS_Explorer.py      ← NEW
    ├── 14_Model_Comparison.py  ← NEW
    └── 15_AI_Explainer.py      ← NEW (Claude-powered)
```

---

Built with ❤️ using Streamlit · PyTorch · NumPy · Plotly · Scikit-learn · Claude API
