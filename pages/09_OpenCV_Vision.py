"""OpenCV Vision — image preprocessing pipeline for CNN."""
import io

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

from utils.export import download_code, download_pickle
from utils.nav import render_sidebar
from utils.theme import apply_theme, hero

st.set_page_config(page_title="OpenCV Vision", layout="wide", page_icon="⬡")
apply_theme()
render_sidebar("OpenCV Vision")

hero(
    "OpenCV + Vision",
    "Preprocess images with classical CV operations before feeding them to your CNN.",
    pill="Lesson 9", pill_variant="orange",
)

with st.expander("📖 Theory", expanded=False):
    st.markdown(r"""
**Why preprocess?**  Raw images contain noise, lighting variation, and irrelevant colour channels.
Preprocessing improves CNN convergence and accuracy.

| Operation | What it does |
|---|---|
| Grayscale | $I = 0.299R + 0.587G + 0.114B$ — reduces input channels to 1 |
| Gaussian Blur | Smooths noise; kernel size controls strength |
| Canny Edge | Finds edges via gradient magnitude + hysteresis thresholding |
| Threshold | Binarises pixel intensities above a value |
| Contours | Outlines connected edge regions |
| Histogram EQ | Redistributes intensity for better contrast |
| Laplacian | Second-order derivative — highlights sharp transitions |
| Morphology | Erode / Dilate to remove noise or fill gaps |
""")

try:
    import cv2
    CV2_OK = True
except Exception as e:
    CV2_OK = False
    cv2_err = str(e)

if not CV2_OK:
    st.error(f"OpenCV import failed: {cv2_err}")
    st.stop()

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    upload = st.file_uploader("Upload image (PNG / JPG)", type=["png", "jpg", "jpeg"])
    tool   = st.selectbox("Operation", [
        "Grayscale", "Gaussian Blur", "Canny Edge",
        "Threshold", "Adaptive Threshold", "Contours",
        "Histogram Equalisation", "Laplacian", "Morphology — Dilate", "Morphology — Erode",
    ])
    blur_k  = st.slider("Blur kernel (odd)", 3, 21, 7, 2)
    thresh  = st.slider("Threshold value", 0, 255, 127, 1)
    canny_l = st.slider("Canny low",  10, 200, 50, 5)
    canny_h = st.slider("Canny high", 50, 400, 150, 5)
    morph_k = st.slider("Morphology kernel", 2, 15, 5, 1)

    resize  = st.checkbox("Resize to 224×224", value=False)

with col2:
    if not upload:
        st.info("⬅ Upload an image to begin.")
        st.stop()

    img_pil  = Image.open(upload).convert("RGB")
    if resize:
        img_pil = img_pil.resize((224, 224), Image.LANCZOS)
    img      = np.array(img_pil)
    gray     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ops = {
        "Grayscale":               lambda: gray,
        "Gaussian Blur":           lambda: cv2.GaussianBlur(img, (blur_k|1, blur_k|1), 0),
        "Canny Edge":              lambda: cv2.Canny(gray, canny_l, canny_h),
        "Threshold":               lambda: cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1],
        "Adaptive Threshold":      lambda: cv2.adaptiveThreshold(
                                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2),
        "Contours":                lambda: _draw_contours(img, gray, canny_l, canny_h),
        "Histogram Equalisation":  lambda: cv2.equalizeHist(gray),
        "Laplacian":               lambda: cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F)),
        "Morphology — Dilate":     lambda: cv2.dilate(gray,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))),
        "Morphology — Erode":      lambda: cv2.erode(gray,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))),
    }

    def _draw_contours(img, gray, lo, hi):
        edges     = cv2.Canny(gray, lo, hi)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = img.copy()
        cv2.drawContours(out, contours, -1, (0, 212, 170), 2)
        return out

    display = ops[tool]()

    # ── Side-by-side original vs processed ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    ax1.imshow(img); ax1.set_title("Original", color="#e6edf3"); ax1.axis("off")
    if display.ndim == 2:
        ax2.imshow(display, cmap="gray")
    else:
        ax2.imshow(display)
    ax2.set_title(tool, color="#00d4aa"); ax2.axis("off")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig)

    # ── Pixel histogram ───────────────────────────────────────────────
    flat = display.reshape(-1) if display.ndim == 2 else display.mean(-1).reshape(-1)
    hist_fig = px.histogram(flat, nbins=60, title="Pixel intensity distribution",
                            template="plotly_dark",
                            color_discrete_sequence=["#00d4aa"])
    hist_fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                           font=dict(color="#8b949e"), height=240,
                           margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(hist_fig, use_container_width=True)

    # ── Stats ─────────────────────────────────────────────────────────
    ch = display.reshape(-1) if display.ndim == 2 else display.mean(-1).reshape(-1)
    st.markdown(f"""
    <div class="nn-card" style="font-size:0.85rem;font-family:'IBM Plex Sans',sans-serif;color:#8b949e">
      Shape: <code>{display.shape}</code> &nbsp;|&nbsp;
      Min: <code>{ch.min():.0f}</code> &nbsp;|&nbsp;
      Max: <code>{ch.max():.0f}</code> &nbsp;|&nbsp;
      Mean: <code>{ch.mean():.1f}</code> &nbsp;|&nbsp;
      Std: <code>{ch.std():.1f}</code>
    </div>
    """, unsafe_allow_html=True)

    # ── Feed to CNN session state ────────────────────────────────────
    if st.button("📤 Queue for CNN tab", type="secondary"):
        buf = io.BytesIO()
        Image.fromarray(display if display.ndim == 3 else
                        np.stack([display]*3, -1)).save(buf, format="PNG")
        st.session_state["vision_image"] = buf.getvalue()
        st.success("Image queued — switch to the CNN tab.")

    c1, c2 = st.columns(2)
    with c1:
        download_pickle("⬇ Save config",
                        {"tool": tool, "blur_k": blur_k, "thresh": thresh,
                         "canny_l": canny_l, "canny_h": canny_h},
                        "cv_config.pkl")
    with c2:
        code = f"""\
import cv2
import numpy as np
from PIL import Image

img  = np.array(Image.open("image.jpg").convert("RGB"))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# {tool}
"""
        op_code = {
            "Grayscale":               "result = gray",
            "Gaussian Blur":           f"result = cv2.GaussianBlur(img, ({blur_k|1},{blur_k|1}), 0)",
            "Canny Edge":              f"result = cv2.Canny(gray, {canny_l}, {canny_h})",
            "Threshold":               f"_, result = cv2.threshold(gray, {thresh}, 255, cv2.THRESH_BINARY)",
            "Adaptive Threshold":      "result = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)",
            "Contours":                f"edges=cv2.Canny(gray,{canny_l},{canny_h}); cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); result=img.copy(); cv2.drawContours(result,cnts,-1,(0,212,170),2)",
            "Histogram Equalisation":  "result = cv2.equalizeHist(gray)",
            "Laplacian":               "result = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))",
            "Morphology — Dilate":     f"k=cv2.getStructuringElement(cv2.MORPH_RECT,({morph_k},{morph_k})); result=cv2.dilate(gray,k)",
            "Morphology — Erode":      f"k=cv2.getStructuringElement(cv2.MORPH_RECT,({morph_k},{morph_k})); result=cv2.erode(gray,k)",
        }
        code += op_code.get(tool, "result = gray") + "\n"
        download_code("⬇ Export Python", code, "opencv_pipeline.py")
