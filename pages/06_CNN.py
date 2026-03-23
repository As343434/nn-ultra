"""CNN — Convolutional Network on MNIST/Fashion-MNIST with filter viewer."""
import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st

from utils.export import download_code, download_torch
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero, metric_row
from utils.viz import plot_loss_curve, plot_confusion_matrix

st.set_page_config(page_title="CNN", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("CNN")

hero(
    "CNN",
    "Train a convolutional network on MNIST or Fashion-MNIST. Visualize feature maps and learned filters.",
    pill="Lesson 6", pill_variant="purple",
)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
A **convolution** slides a small filter $f$ over the input $x$:

$$(f * x)(i,j) = \sum_{m,n} f_{m,n}\,x_{i+m,\,j+n}$$

**Key operations:**
- `Conv2d` — learns spatial filters (edges, textures, shapes)
- `ReLU`   — introduces nonlinearity
- `MaxPool2d` — down-samples, adds translation invariance
- `Flatten + Linear` — outputs class logits

**Architecture used:** Conv(1→C) → ReLU → MaxPool → Conv(C→2C) → ReLU → MaxPool → FC → 10
""")

try:
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from sklearn.metrics import confusion_matrix as sk_cm
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

if not TORCH_OK:
    st.error("PyTorch / torchvision not installed.")
    st.stop()

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    dname   = st.selectbox("Dataset", ["MNIST", "Fashion-MNIST"])
    epochs  = st.slider("Epochs", 1, 10, 3, 1)
    lr      = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
    filters = st.slider("Filters (C)", 8, 64, 16, 8)
    bs      = st.slider("Batch size", 32, 256, 128, 32)

with col2:
    st.markdown("""
    <div class="nn-card">
      <div style="font-size:0.85rem;color:#8b949e;font-family:'IBM Plex Sans',sans-serif">
        Architecture:<br>
        <code>Conv2d(1,C,3) → ReLU → MaxPool(2)<br>
        Conv2d(C,2C,3) → ReLU → MaxPool(2)<br>
        Flatten → Linear(2C·7·7, 128) → ReLU → Linear(128,10)</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Click **▶ Train CNN** to load data, train, and view feature maps.")

train_btn = st.button("▶ Train CNN", type="primary")

if train_btn:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    Cls = datasets.MNIST if dname == "MNIST" else datasets.FashionMNIST
    train_ds = Cls("./data", train=True,  download=True, transform=transform)
    test_ds  = Cls("./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(1, filters, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(filters, filters*2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(filters*2*7*7, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses = []
    bar = st.progress(0)
    total_batches = epochs * len(train_loader)
    batch_count = 0

    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()
            ep_loss += loss.item()
            batch_count += 1
            bar.progress(int(batch_count / total_batches * 100))
        sched.step()
        losses.append(ep_loss / len(train_loader))

    # Evaluation
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.append(model(xb).argmax(1).numpy())
            all_labels.append(yb.numpy())
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = np.mean(all_preds == all_labels)

    st.success(f"✓ Done — Test accuracy: **{acc:.2%}**")
    metric_row([("Accuracy", f"{acc:.2%}"), ("Final Loss", f"{losses[-1]:.4f}"),
                ("Filters C", filters), ("Epochs", epochs)])

    st.plotly_chart(plot_loss_curve(losses), use_container_width=True)

    cm = sk_cm(all_labels, all_preds)
    labels = list(range(10))
    if dname == "Fashion-MNIST":
        labels = ["T-shirt","Trouser","Pullover","Dress","Coat",
                  "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    st.plotly_chart(plot_confusion_matrix(cm, labels), use_container_width=True)

    # Feature maps from first conv layer on a sample
    st.markdown("#### Learned filters — first conv layer")
    sample, _ = train_ds[0]
    with torch.no_grad():
        fmaps = model[0](sample.unsqueeze(0))[0].numpy()  # (C, 28, 28)

    n_show = min(8, filters)
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 1.8, 2.2),
                             facecolor="#0d1117")
    for i, ax in enumerate(axes):
        ax.imshow(fmaps[i], cmap="inferno")
        ax.set_title(f"f{i+1}", color="#8b949e", fontsize=8)
        ax.axis("off")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig)

    # First-layer filter weights
    st.markdown("#### First-layer kernel weights")
    wts = model[0].weight.detach().numpy()  # (C, 1, 3, 3)
    fig2, axes2 = plt.subplots(1, n_show, figsize=(n_show * 1.8, 2.2),
                                facecolor="#0d1117")
    for i, ax in enumerate(axes2):
        ax.imshow(wts[i, 0], cmap="RdBu_r", vmin=-wts.max(), vmax=wts.max())
        ax.set_title(f"k{i+1}", color="#8b949e", fontsize=8)
        ax.axis("off")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig2)

    download_torch("⬇ Download model", model.state_dict(), "cnn_model.pt")

    code = f"""\
import torch, torch.nn as nn
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_ds = datasets.{dname.replace('-','')}('./data', train=True, download=True, transform=transform)

model = nn.Sequential(
    nn.Conv2d(1, {filters}, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d({filters}, {filters*2}, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear({filters*2}*7*7, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
opt = torch.optim.Adam(model.parameters(), lr={lr})
"""
    download_code("⬇ Export Python", code, "cnn.py")
