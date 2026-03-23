import io
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="OpenCV Vision",
    layout="wide",
    page_icon="◉"
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
        background: rgba(249, 115, 22, 0.15) !important;
        color: #f97316 !important;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    button[kind="primary"], button[kind="secondary"] {
        border-radius: 8px !important;
        font-weight: 700 !important;
    }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("◉ OpenCV Vision")
    st.markdown("Classical Image Preprocessing")
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 9</div>
    <h1>OpenCV + Vision Pipeline</h1>
    <p style="color: var(--muted); font-size: 1.1rem;">
        Apply classical computer vision operations to prepare images for CNNs.<br>
        Visualize every step and export the pipeline.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
**Why preprocess images?**  
Raw images often contain noise, poor contrast, and unnecessary color channels.  
Proper preprocessing dramatically improves CNN training speed and accuracy.

| Operation              | Purpose |
|------------------------|--------|
| Grayscale              | Reduce to 1 channel |
| Gaussian Blur          | Remove noise |
| Canny Edge             | Detect strong edges |
| Threshold / Adaptive   | Binarization |
| Contours               | Find object boundaries |
| Histogram Equalization | Improve contrast |
| Laplacian              | Highlight sharp changes |
| Morphology (Erode/Dilate) | Clean noise or fill gaps |
""")

st.markdown("---")

# ====================== CONTROLS ======================
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload Image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
    
    operation = st.selectbox("Select Operation", [
        "Grayscale", "Gaussian Blur", "Canny Edge",
        "Threshold", "Adaptive Threshold", "Contours",
        "Histogram Equalisation", "Laplacian",
        "Morphology — Dilate", "Morphology — Erode"
    ])
    
    blur_kernel = st.slider("Gaussian Blur Kernel (odd)", 3, 21, 7, 2)
    threshold_val = st.slider("Threshold Value", 0, 255, 127, 1)
    canny_low = st.slider("Canny Low Threshold", 10, 200, 50, 5)
    canny_high = st.slider("Canny High Threshold", 50, 400, 150, 5)
    morph_kernel = st.slider("Morphology Kernel Size", 2, 15, 5, 1)
    
    resize_to_224 = st.checkbox("Resize to 224×224 (for CNN)", value=False)

with col2:
    if not uploaded_file:
        st.info("⬅ Please upload an image to start processing.")
        st.stop()

    # Load and preprocess image
    img_pil = Image.open(uploaded_file).convert("RGB")
    if resize_to_224:
        img_pil = img_pil.resize((224, 224), Image.LANCZOS)
    
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Define operations
    def draw_contours(img_color, gray_img, low, high):
        edges = cv2.Canny(gray_img, low, high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img_color.copy()
        cv2.drawContours(output, contours, -1, (0, 212, 170), 3)
        return output

    operations = {
        "Grayscale": lambda: gray,
        "Gaussian Blur": lambda: cv2.GaussianBlur(img, (blur_kernel | 1, blur_kernel | 1), 0),
        "Canny Edge": lambda: cv2.Canny(gray, canny_low, canny_high),
        "Threshold": lambda: cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)[1],
        "Adaptive Threshold": lambda: cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        "Contours": lambda: draw_contours(img, gray, canny_low, canny_high),
        "Histogram Equalisation": lambda: cv2.equalizeHist(gray),
        "Laplacian": lambda: cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F)),
        "Morphology — Dilate": lambda: cv2.dilate(
            gray, cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))),
        "Morphology — Erode": lambda: cv2.erode(
            gray, cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))),
    }

    processed = operations[operation]()

    # ====================== VISUALIZATION ======================
    st.markdown(f"### Original vs {operation}")
    
    col_orig, col_proc = st.columns(2)
    
    with col_orig:
        st.image(img, caption="Original Image", use_column_width=True)
    
    with col_proc:
        if processed.ndim == 2:
            st.image(processed, caption=operation, use_column_width=True, clamp=True)
        else:
            st.image(processed, caption=operation, use_column_width=True)

    # Pixel Intensity Histogram
    st.markdown("### Pixel Intensity Distribution")
    flat_pixels = processed.flatten() if processed.ndim == 2 else processed.mean(axis=2).flatten()
    
    hist_fig = px.histogram(
        flat_pixels, nbins=100,
        title="Pixel Intensity Histogram",
        color_discrete_sequence=["#00d4aa"]
    )
    hist_fig.update_layout(
        height=280,
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9")
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Image Statistics
    stats = {
        "Shape": processed.shape,
        "Min": int(flat_pixels.min()),
        "Max": int(flat_pixels.max()),
        "Mean": round(float(flat_pixels.mean()), 1),
        "Std": round(float(flat_pixels.std()), 1)
    }
    
    st.markdown(f"""
    <div class="nn-card">
        <strong>Image Statistics</strong><br>
        Shape: <code>{stats['Shape']}</code> &nbsp;|&nbsp;
        Min: <code>{stats['Min']}</code> &nbsp;|&nbsp;
        Max: <code>{stats['Max']}</code> &nbsp;|&nbsp;
        Mean: <code>{stats['Mean']}</code> &nbsp;|&nbsp;
        Std: <code>{stats['Std']}</code>
    </div>
    """, unsafe_allow_html=True)

    # Queue for CNN & Downloads
    st.markdown("---")
    if st.button("📤 Queue Processed Image for CNN Module", type="secondary", use_container_width=True):
        buf = io.BytesIO()
        if processed.ndim == 2:
            processed_rgb = np.stack([processed] * 3, axis=-1)
        else:
            processed_rgb = processed
        Image.fromarray(processed_rgb).save(buf, format="PNG")
        st.session_state["processed_image_for_cnn"] = buf.getvalue()
        st.success("✅ Image queued successfully! Go to the CNN module to use it.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="⬇ Download Processed Image",
            data=Image.fromarray(processed if processed.ndim == 3 else np.stack([processed]*3, -1)).tobytes("png"),
            file_name=f"processed_{operation.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
    
    with col_d2:
        code_str = f"""import cv2
import numpy as np
from PIL import Image

img = np.array(Image.open("your_image.jpg").convert("RGB"))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# {operation}
"""
        op_code_map = {
            "Grayscale": "result = gray",
            "Gaussian Blur": f"result = cv2.GaussianBlur(img, ({blur_kernel|1}, {blur_kernel|1}), 0)",
            "Canny Edge": f"result = cv2.Canny(gray, {canny_low}, {canny_high})",
            "Threshold": f"_, result = cv2.threshold(gray, {threshold_val}, 255, cv2.THRESH_BINARY)",
            "Adaptive Threshold": "result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)",
            "Contours": f"""edges = cv2.Canny(gray, {canny_low}, {canny_high})
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 212, 170), 3)""",
            "Histogram Equalisation": "result = cv2.equalizeHist(gray)",
            "Laplacian": "result = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))",
            "Morphology — Dilate": f"k = cv2.getStructuringElement(cv2.MORPH_RECT, ({morph_kernel}, {morph_kernel}))\nresult = cv2.dilate(gray, k)",
            "Morphology — Erode": f"k = cv2.getStructuringElement(cv2.MORPH_RECT, ({morph_kernel}, {morph_kernel}))\nresult = cv2.erode(gray, k)",
        }
        code_str += op_code_map.get(operation, "result = gray") + "\n\ncv2.imwrite('output.png', result)"
        
        st.download_button(
            label="⬇ Export Python Code",
            data=code_str,
            file_name="opencv_preprocessing.py",
            mime="text/plain"
        )
