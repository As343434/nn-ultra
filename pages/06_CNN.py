import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="CNN — Convolutional Network",
    layout="wide",
    page_icon="◫"
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
    button[kind="primary"] {
        background: var(--accent) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("◫ CNN")
    st.markdown("Convolutional Neural Network")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 6</div>
    <h1>CNN — Convolutional Network</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Train a convolutional neural network on MNIST or Fashion-MNIST.<br>
        Visualize learned filters and feature maps in real time.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
A **Convolutional Neural Network (CNN)** uses filters that slide over the image to detect patterns:

$$(f * x)(i,j) = \sum_{m,n} f_{m,n}\, x_{i+m,j+n}$$

**Key Components:**
- `Conv2d` → Learns spatial features (edges, textures, shapes)
- `ReLU` → Non-linearity
- `MaxPool2d` → Downsampling + translation invariance
- Fully Connected layers at the end for classification

**Architecture:**
Conv(1→C) → ReLU → MaxPool(2) → Conv(C→2C) → ReLU → MaxPool(2) → Flatten → Linear → 10 classes
""")

st.markdown("---")

# ====================== CONTROLS ======================
col1, col2 = st.columns([1, 2])

with col1:
    dataset_name = st.selectbox("Dataset", ["MNIST", "Fashion-MNIST"])
    epochs = st.slider("Epochs", 1, 10, 3, 1)
    lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    num_filters = st.slider("Number of Filters (C)", 8, 64, 16, 8)
    batch_size = st.slider("Batch Size", 32, 256, 128, 32)

with col2:
    st.markdown("""
    <div class="nn-card">
        <strong>Model Architecture:</strong><br>
        <code style="font-size:0.85rem;">
        Conv2d(1, C, 3, padding=1) → ReLU → MaxPool2d(2)<br>
        Conv2d(C, 2C, 3, padding=1) → ReLU → MaxPool2d(2)<br>
        Flatten → Linear(2C×7×7, 128) → ReLU → Linear(128, 10)
        </code>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 Adjust parameters and click **Train CNN** below")

# ====================== TRAINING BUTTON ======================
train_btn = st.button("▶ Train CNN", type="primary", use_container_width=True)

if train_btn:
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    DatasetClass = datasets.MNIST if dataset_name == "MNIST" else datasets.FashionMNIST
    train_dataset = DatasetClass(root="./data", train=True, download=True, transform=transform)
    test_dataset = DatasetClass(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Model definition
    model = nn.Sequential(
        nn.Conv2d(1, num_filters, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(num_filters * 2 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_batches = epochs * len(train_loader)
    batch_count = 0

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            progress_bar.progress(min(int(batch_count / total_batches * 100), 100))

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        status_text.text(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    st.success(f"✅ Training Complete! Final Loss: {losses[-1]:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{accuracy:.2%}")
    c2.metric("Final Loss", f"{losses[-1]:.4f}")
    c3.metric("Filters", num_filters)

    # Loss Curve
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=losses, mode='lines+markers', 
                                  line=dict(color='#00d4aa', width=3),
                                  name='Training Loss'))
    fig_loss.update_layout(title="Training Loss Curve", height=320,
                           paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                           font=dict(color="#c9d1d9"))
    st.plotly_chart(fig_loss, use_container_width=True)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = list(range(10))
    if dataset_name == "Fashion-MNIST":
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       labels=dict(x="Predicted", y="True Label"),
                       x=class_names, y=class_names)
    fig_cm.update_layout(title="Confusion Matrix", height=500)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ====================== VISUALIZATIONS ======================
    st.markdown("### 🔍 Feature Maps & Learned Filters")

    # Get one sample
    sample_img, _ = train_dataset[0]
    sample_img = sample_img.unsqueeze(0)  # (1, 1, 28, 28)

    # First Conv Layer Feature Maps
    with torch.no_grad():
        first_conv = model[0](sample_img)  # (1, C, 28, 28)
        feature_maps = first_conv[0].numpy()  # (C, 28, 28)

    st.markdown("#### Feature Maps from First Convolution Layer")
    cols = st.columns(8)
    for i in range(min(8, num_filters)):
        with cols[i % 8]:
            fig = px.imshow(feature_maps[i], color_continuous_scale="inferno", 
                            title=f"Filter {i+1}")
            fig.update_layout(height=180, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    # First Layer Kernel Weights
    st.markdown("#### Learned Kernel Weights (First Conv Layer)")
    kernels = model[0].weight.detach().numpy()  # (C, 1, 3, 3)

    cols2 = st.columns(8)
    for i in range(min(8, num_filters)):
        with cols2[i % 8]:
            fig = px.imshow(kernels[i, 0], color_continuous_scale="RdBu_r",
                            title=f"Kernel {i+1}")
            fig.update_layout(height=140, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Download Options
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        torch.save(model.state_dict(), "cnn_model.pt")
        with open("cnn_model.pt", "rb") as f:
            st.download_button(
                label="⬇ Download Trained Model (.pt)",
                data=f,
                file_name="cnn_model.pt",
                mime="application/octet-stream"
            )
    with col_d2:
        code_str = f"""import torch
import torch.nn as nn
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_ds = datasets.{dataset_name.replace('-','')}('./data', train=True, download=True, transform=transform)

model = nn.Sequential(
    nn.Conv2d(1, {num_filters}, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d({num_filters}, {num_filters*2}, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear({num_filters*2}*7*7, 128), nn.ReLU(),
    nn.Linear(128, 10)
)
"""
        st.download_button(
            label="⬇ Export Model Code",
            data=code_str,
            file_name="cnn_model.py",
            mime="text/plain"
        )

    st.balloons()
