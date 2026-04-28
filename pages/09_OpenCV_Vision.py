import io
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import time

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="OpenCV Vision", layout="wide", page_icon="◉")

# ====================== CSS ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap');
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
h1,h2,h3,h4 { font-family:'IBM Plex Sans',sans-serif !important; font-weight:700 !important; letter-spacing:-0.02em !important; }
.nn-card { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:12px !important; padding:1.4rem !important; margin-bottom:1rem !important; }
.nn-hero { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:16px !important; padding:2.2rem 2rem !important; margin-bottom:1.6rem !important; }
.nn-pill { display:inline-block; padding:0.25rem 0.8rem; border-radius:999px; background:rgba(249,115,22,0.15) !important; color:#f97316 !important; font-size:0.75rem; font-weight:700; text-transform:uppercase; }
.live-badge { display:inline-block; padding:0.2rem 0.7rem; border-radius:999px; background:rgba(34,197,94,0.2); color:#22c55e; font-size:0.72rem; font-weight:700; text-transform:uppercase; animation:pulse-badge 1.5s infinite; }
@keyframes pulse-badge { 0%,100%{opacity:1} 50%{opacity:0.4} }
.trait-bar-wrap { margin-bottom:7px; }
.trait-label { display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-bottom:3px; }
.trait-bar-bg { background:#1e293b; border-radius:6px; height:9px; overflow:hidden; }
.trait-bar-fill { height:100%; border-radius:6px; transition:width 0.4s ease; }
.nature-tag { font-size:1.05rem; font-weight:700; padding:0.35rem 1rem; border-radius:999px; background:rgba(249,115,22,0.15); color:#f97316; display:inline-block; margin:0.5rem 0; }
button[kind="primary"],button[kind="secondary"] { border-radius:8px !important; font-weight:700 !important; }
hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ====================== CASCADE LOADERS ======================
@st.cache_resource
def load_cascades():
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    ec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    sc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    return fc, ec, sc

face_cas, eye_cas, smile_cas = load_cascades()

# ====================== HELPERS ======================
def detect_faces_full(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
    return faces, gray

def analyze_nature(roi_gray, roi_bgr):
    h, w = roi_gray.shape
    scores = {}
    smiles = smile_cas.detectMultiScale(roi_gray, 1.7, 22)
    scores["😊 Smiling"]             = int(min(len(smiles)*35 + np.random.randint(5,25), 95))
    eyes = eye_cas.detectMultiScale(roi_gray, 1.1, 10)
    scores["👁️ Alert / Eyes Open"]   = 88 if len(eyes)>=2 else (50 if len(eyes)==1 else 18)
    bright = roi_gray.mean()
    scores["☀️ Bright / Expressive"] = int(np.clip((bright/255)*100 + np.random.randint(-8,8), 5, 95))
    lap = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    scores["🎯 Focused / Intense"]   = int(np.clip(lap/3 + np.random.randint(5,18), 5, 95))
    b = roi_bgr[:,:,0].mean(); r = roi_bgr[:,:,2].mean()
    scores["🌡️ Warm Skin Tone"]      = int(np.clip(((r-b)/255)*100+40+np.random.randint(-4,4), 10, 90))
    left = roi_gray[:, :w//2]; right = cv2.flip(roi_gray[:, w//2:], 1)
    mw = min(left.shape[1], right.shape[1])
    diff = np.abs(left[:,:mw].astype(int)-right[:,:mw].astype(int)).mean()
    scores["🔄 Facial Symmetry"]     = int(np.clip(100-diff+np.random.randint(-4,4), 20, 95))
    return scores

def nature_label(scores):
    top = sorted(scores.items(), key=lambda x:-x[1])
    d = top[0][0]
    if "Smiling" in d and top[0][1]>60:  return "😄 Cheerful & Warm"
    if "Focused" in d and top[0][1]>65:  return "🧠 Analytical & Intense"
    if "Alert" in d and top[0][1]>70:    return "👀 Attentive & Aware"
    if "Bright" in d:                     return "✨ Expressive & Open"
    return "🤔 Neutral / Composed"

def face_match_score(g1, g2):
    h1 = cv2.calcHist([g1],[0],None,[64],[0,256]); cv2.normalize(h1,h1)
    h2 = cv2.calcHist([g2],[0],None,[64],[0,256]); cv2.normalize(h2,h2)
    corr = max(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL), 0)
    orb = cv2.ORB_create(200)
    kp1,d1 = orb.detectAndCompute(cv2.resize(g1,(128,128)), None)
    kp2,d2 = orb.detectAndCompute(cv2.resize(g2,(128,128)), None)
    orb_s = 0.0
    if d1 is not None and d2 is not None and len(d1)>5 and len(d2)>5:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good = [m for m in bf.match(d1,d2) if m.distance<60]
        orb_s = min(len(good)/max(len(kp1),len(kp2),1), 1.0)
    return round(float(np.clip((corr*0.6+orb_s*0.4)*100, 0, 100)), 1)

def draw_face_boxes(img_bgr, faces, gray):
    out = img_bgr.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,100),2)
        roi_g = gray[y:y+h, x:x+w]
        roi_b = out[y:y+h, x:x+w]
        for (ex,ey,ew,eh) in eye_cas.detectMultiScale(roi_g,1.1,10):
            cv2.circle(roi_b,(ex+ew//2,ey+eh//2),ew//2,(255,200,0),2)
        cv2.putText(out,f"Face {w}x{h}",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,100),1)
    return out

def pil2bgr(p): return cv2.cvtColor(np.array(p.convert("RGB")), cv2.COLOR_RGB2BGR)
def bgr2pil(b): return Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))

def render_trait_bars(scores):
    html = ""
    for trait, pct in sorted(scores.items(), key=lambda x:-x[1]):
        color = "#22c55e" if pct>65 else "#f59e0b" if pct>40 else "#ef4444"
        html += f"""<div class='trait-bar-wrap'>
          <div class='trait-label'><span>{trait}</span><span><b>{pct}%</b></span></div>
          <div class='trait-bar-bg'><div class='trait-bar-fill' style='width:{pct}%;background:{color}'></div></div>
        </div>"""
    return html

def render_match_card(score):
    color = "#22c55e" if score>65 else "#f59e0b" if score>40 else "#ef4444"
    verdict = "✅ Likely Same Person" if score>65 else "⚠️ Possibly Same" if score>40 else "❌ Different People"
    return f"""<div style='text-align:center;padding:1rem;background:#0f172a;border-radius:12px;border:1px solid #1e293b'>
      <div style='font-size:3rem;font-weight:900;color:{color}'>{score}%</div>
      <div style='color:{color};font-size:0.9rem;margin-top:0.3rem'>{verdict}</div>
      <div style='color:#475569;font-size:0.72rem;margin-top:0.5rem'>Histogram + ORB Keypoints</div>
    </div>"""

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("◉ OpenCV Vision")
    st.markdown("Classical Image Preprocessing + Face AI")
    st.markdown("---")
    st.markdown("""
**Tabs**  
🖼️ Preprocessing Pipeline  
📸 Face Detect & Analyse  
🔴 Live Nature Feed  
🔍 Face Match (Images)  
🔴 Live Face Match
    """)
    st.markdown("---")
    st.caption("NeuralForge Ultra v3.0")

# ====================== HERO ======================
st.markdown("""
<div class="nn-hero">
    <div class="nn-pill">Lesson 9</div>
    <h1>OpenCV + Vision Pipeline</h1>
    <p style="color:var(--muted);font-size:1.1rem;">
        Classical preprocessing · Face detection · Nature analysis · Live webcam · Face matching
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
| Morphology             | Clean noise / fill gaps |
| Face Detection         | Haar Cascade + heuristic analysis |
| Face Matching          | Histogram correlation + ORB |
""")

st.markdown("---")

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖼️ Preprocessing",
    "📸 Face Detect & Analyse",
    "🔴 Live Nature Feed",
    "🔍 Face Match (Images)",
    "🔴 Live Face Match"
])

# ════════════════════════════════════════════════
# TAB 1 — Original Preprocessing Pipeline
# ════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("Upload Image (PNG / JPG / JPEG)", type=["png","jpg","jpeg"], key="prep_upload")
        operation = st.selectbox("Select Operation", [
            "Grayscale","Gaussian Blur","Canny Edge",
            "Threshold","Adaptive Threshold","Contours",
            "Histogram Equalisation","Laplacian",
            "Morphology — Dilate","Morphology — Erode"
        ])
        blur_kernel   = st.slider("Gaussian Blur Kernel (odd)", 3, 21, 7, 2)
        threshold_val = st.slider("Threshold Value", 0, 255, 127, 1)
        canny_low     = st.slider("Canny Low Threshold", 10, 200, 50, 5)
        canny_high    = st.slider("Canny High Threshold", 50, 400, 150, 5)
        morph_kernel  = st.slider("Morphology Kernel Size", 2, 15, 5, 1)
        resize_to_224 = st.checkbox("Resize to 224×224 (for CNN)", value=False)

    with col2:
        if not uploaded_file:
            st.info("⬅ Please upload an image to start processing.")
        else:
            img_pil = Image.open(uploaded_file).convert("RGB")
            if resize_to_224:
                img_pil = img_pil.resize((224,224), Image.LANCZOS)
            img  = np.array(img_pil)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            def draw_contours_op(ic, ig, lo, hi):
                edges = cv2.Canny(ig, lo, hi)
                contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                out = ic.copy(); cv2.drawContours(out, contours,-1,(0,212,170),3)
                return out

            ops = {
                "Grayscale":             lambda: gray,
                "Gaussian Blur":         lambda: cv2.GaussianBlur(img,(blur_kernel|1,blur_kernel|1),0),
                "Canny Edge":            lambda: cv2.Canny(gray, canny_low, canny_high),
                "Threshold":             lambda: cv2.threshold(gray, threshold_val,255,cv2.THRESH_BINARY)[1],
                "Adaptive Threshold":    lambda: cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2),
                "Contours":              lambda: draw_contours_op(img, gray, canny_low, canny_high),
                "Histogram Equalisation":lambda: cv2.equalizeHist(gray),
                "Laplacian":             lambda: cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F)),
                "Morphology — Dilate":   lambda: cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT,(morph_kernel,morph_kernel))),
                "Morphology — Erode":    lambda: cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_RECT,(morph_kernel,morph_kernel))),
            }
            processed = ops[operation]()

            st.markdown(f"### Original vs {operation}")
            co, cp = st.columns(2)
            with co: st.image(img, caption="Original Image", use_column_width=True)
            with cp:
                if processed.ndim==2:
                    st.image(processed, caption=operation, use_column_width=True, clamp=True)
                else:
                    st.image(processed, caption=operation, use_column_width=True)

            st.markdown("### Pixel Intensity Distribution")
            flat = processed.flatten() if processed.ndim==2 else processed.mean(axis=2).flatten()
            hist_fig = px.histogram(flat, nbins=100, title="Pixel Intensity Histogram",
                                    color_discrete_sequence=["#00d4aa"])
            hist_fig.update_layout(height=280, paper_bgcolor="#161b22",
                                   plot_bgcolor="#0d1117", font=dict(color="#c9d1d9"))
            st.plotly_chart(hist_fig, use_container_width=True)

            stats = {"Shape":processed.shape,"Min":int(flat.min()),"Max":int(flat.max()),
                     "Mean":round(float(flat.mean()),1),"Std":round(float(flat.std()),1)}
            st.markdown(f"""
            <div class="nn-card">
                <strong>Image Statistics</strong><br>
                Shape: <code>{stats['Shape']}</code> &nbsp;|&nbsp;
                Min: <code>{stats['Min']}</code> &nbsp;|&nbsp;
                Max: <code>{stats['Max']}</code> &nbsp;|&nbsp;
                Mean: <code>{stats['Mean']}</code> &nbsp;|&nbsp;
                Std: <code>{stats['Std']}</code>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            if st.button("📤 Queue Processed Image for CNN Module", type="secondary", use_container_width=True):
                buf = io.BytesIO()
                prgb = np.stack([processed]*3,-1) if processed.ndim==2 else processed
                Image.fromarray(prgb).save(buf, format="PNG")
                st.session_state["processed_image_for_cnn"] = buf.getvalue()
                st.success("✅ Image queued successfully! Go to the CNN module to use it.")

            cd1,cd2 = st.columns(2)
            with cd1:
                pil_out = Image.fromarray(processed if processed.ndim==3 else np.stack([processed]*3,-1))
                buf2 = io.BytesIO(); pil_out.save(buf2, format="PNG")
                st.download_button("⬇ Download Processed Image", data=buf2.getvalue(),
                    file_name=f"processed_{operation.lower().replace(' ','_')}.png", mime="image/png")
            with cd2:
                op_code = {
                    "Grayscale":"result = gray",
                    "Gaussian Blur":f"result = cv2.GaussianBlur(img,({blur_kernel|1},{blur_kernel|1}),0)",
                    "Canny Edge":f"result = cv2.Canny(gray,{canny_low},{canny_high})",
                    "Threshold":f"_,result = cv2.threshold(gray,{threshold_val},255,cv2.THRESH_BINARY)",
                    "Adaptive Threshold":"result = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)",
                    "Contours":f"edges=cv2.Canny(gray,{canny_low},{canny_high})\ncontours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)\nresult=img.copy()\ncv2.drawContours(result,contours,-1,(0,212,170),3)",
                    "Histogram Equalisation":"result = cv2.equalizeHist(gray)",
                    "Laplacian":"result = cv2.convertScaleAbs(cv2.Laplacian(gray,cv2.CV_64F))",
                    "Morphology — Dilate":f"k=cv2.getStructuringElement(cv2.MORPH_RECT,({morph_kernel},{morph_kernel}))\nresult=cv2.dilate(gray,k)",
                    "Morphology — Erode":f"k=cv2.getStructuringElement(cv2.MORPH_RECT,({morph_kernel},{morph_kernel}))\nresult=cv2.erode(gray,k)",
                }.get(operation,"result=gray")
                code_str = f"import cv2\nimport numpy as np\nfrom PIL import Image\n\nimg=np.array(Image.open('your_image.jpg').convert('RGB'))\ngray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)\n\n# {operation}\n{op_code}\n\ncv2.imwrite('output.png',result)"
                st.download_button("⬇ Export Python Code", data=code_str,
                    file_name="opencv_preprocessing.py", mime="text/plain")


# ════════════════════════════════════════════════
# TAB 2 — Face Detect & Analyse (Static Image)
# ════════════════════════════════════════════════
with tab2:
    st.subheader("📸 Face Detection + Nature Analysis")
    t2c1, t2c2 = st.columns([1,2], gap="large")
    with t2c1:
        src2 = st.radio("Source", ["📷 Webcam Snapshot","🖼️ Upload Image"], key="src2", horizontal=True)
        img2_pil = None
        if src2 == "📷 Webcam Snapshot":
            cam = st.camera_input("Take a photo", key="cam2")
            if cam: img2_pil = Image.open(cam)
        else:
            up2 = st.file_uploader("Upload", type=["jpg","jpeg","png","bmp","webp"], key="up2")
            if up2: img2_pil = Image.open(up2)
        run2 = st.button("🔍 Detect & Analyse Faces", use_container_width=True, key="btn2")
    with t2c2:
        if img2_pil and run2:
            bgr2 = pil2bgr(img2_pil)
            faces2, gray2 = detect_faces_full(bgr2)
            st.image(bgr2pil(draw_face_boxes(bgr2,faces2,gray2)),
                     use_column_width=True, caption=f"Detected {len(faces2)} face(s)")
            if len(faces2)==0:
                st.warning("No faces detected. Try a well-lit, front-facing image.")
            else:
                for i,(x,y,w,h) in enumerate(faces2):
                    roi_g = gray2[y:y+h, x:x+w]
                    roi_b = bgr2[y:y+h, x:x+w]
                    with st.expander(f"🧑 Face #{i+1}  ({w}×{h}px)", expanded=(i==0)):
                        fa1,fa2 = st.columns([1,2])
                        with fa1: st.image(bgr2pil(roi_b), width=120, caption="ROI")
                        with fa2:
                            scores = analyze_nature(roi_g, roi_b)
                            st.markdown(f"<div class='nature-tag'>{nature_label(scores)}</div>", unsafe_allow_html=True)
                            st.markdown(render_trait_bars(scores), unsafe_allow_html=True)
        elif not img2_pil:
            st.info("Provide an image then click Detect & Analyse.")


# ════════════════════════════════════════════════
# TAB 3 — Live Webcam Nature Feed
# ════════════════════════════════════════════════
with tab3:
    st.subheader("🔴 Live Face Nature Detection")
    st.markdown("<span class='live-badge'>● LIVE</span> &nbsp; Webcam frame analysed on every capture", unsafe_allow_html=True)
    st.markdown("")

    lc1, lc2 = st.columns([1,2], gap="large")
    with lc1:
        auto_refresh = st.checkbox("🔄 Auto-Refresh (loops every 2s)", value=False, key="live_auto")
        live_frame = st.camera_input("📷 Capture frame", key="live_cam")
        if auto_refresh:
            time.sleep(2)
            st.rerun()
        st.markdown("""
        <div style='padding:0.8rem;background:#0f172a;border-radius:8px;border:1px solid #1e293b;font-size:0.78rem;color:#64748b;margin-top:0.5rem'>
        💡 <b>Tip:</b> Enable Auto-Refresh to keep re-capturing your webcam automatically. Each snapshot is processed instantly.
        </div>""", unsafe_allow_html=True)

    with lc2:
        if live_frame:
            live_bgr = pil2bgr(Image.open(live_frame))
            faces_l, gray_l = detect_faces_full(live_bgr)
            st.image(bgr2pil(draw_face_boxes(live_bgr,faces_l,gray_l)),
                     use_column_width=True, caption=f"Live — {len(faces_l)} face(s)")
            if len(faces_l)>0:
                for i,(x,y,w,h) in enumerate(faces_l):
                    roi_g = gray_l[y:y+h, x:x+w]
                    roi_b = live_bgr[y:y+h, x:x+w]
                    scores = analyze_nature(roi_g, roi_b)
                    lbl = nature_label(scores)
                    st.markdown(f"**Face #{i+1}:** <span class='nature-tag'>{lbl}</span>", unsafe_allow_html=True)
                    st.markdown(render_trait_bars(scores), unsafe_allow_html=True)
                    if i < len(faces_l)-1: st.markdown("---")
            else:
                st.warning("No faces in frame. Look at the camera and ensure good lighting.")
        else:
            st.markdown("""
            <div style='padding:3rem;text-align:center;color:#475569;border:1px dashed #334155;border-radius:12px'>
                <div style='font-size:3rem'>🎥</div>
                <div style='margin-top:0.5rem'>Capture a webcam frame to start live nature analysis</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 4 — Static Face Match
# ════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Face Match — Upload Two Images")
    st.caption("Compare two photos and get a face similarity % score.")
    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.markdown("**Image A**")
        fa = st.file_uploader("Upload Image A", type=["jpg","jpeg","png","bmp","webp"], key="mfa")
    with m2:
        st.markdown("**Image B**")
        fb = st.file_uploader("Upload Image B", type=["jpg","jpeg","png","bmp","webp"], key="mfb")

    if st.button("⚡ Compare Faces", use_container_width=True, key="cmp_btn"):
        if fa and fb:
            bgrA = pil2bgr(Image.open(fa)); bgrB = pil2bgr(Image.open(fb))
            fA,gA = detect_faces_full(bgrA); fB,gB = detect_faces_full(bgrB)
            r1,r2,r3 = st.columns([2,1,2], gap="medium")
            with r1: st.image(bgr2pil(draw_face_boxes(bgrA,fA,gA)), use_column_width=True, caption=f"Image A — {len(fA)} face(s)")
            with r3: st.image(bgr2pil(draw_face_boxes(bgrB,fB,gB)), use_column_width=True, caption=f"Image B — {len(fB)} face(s)")
            with r2:
                if len(fA)==0 or len(fB)==0:
                    st.error("Face not detected in one/both images.")
                else:
                    xA,yA,wA,hA=fA[0]; xB,yB,wB,hB=fB[0]
                    sc = face_match_score(gA[yA:yA+hA,xA:xA+wA], gB[yB:yB+hB,xB:xB+wB])
                    st.markdown(render_match_card(sc), unsafe_allow_html=True)
        else:
            st.warning("Please upload both images first.")


# ════════════════════════════════════════════════
# TAB 5 — Live Face Match (Webcam vs Reference)
# ════════════════════════════════════════════════
with tab5:
    st.subheader("🔴 Live Face Match")
    st.markdown("<span class='live-badge'>● LIVE</span> &nbsp; Webcam matched against a reference face in real-time", unsafe_allow_html=True)
    st.markdown("")

    ref_col, live_col = st.columns([1,2], gap="large")

    with ref_col:
        st.markdown("#### 📁 Reference Image")
        ref_file = st.file_uploader("Upload reference face", type=["jpg","jpeg","png","bmp","webp"], key="ref_img")
        if ref_file:
            ref_bgr = pil2bgr(Image.open(ref_file))
            ref_faces, ref_gray = detect_faces_full(ref_bgr)
            if len(ref_faces)>0:
                rx,ry,rw,rh = ref_faces[0]
                st.session_state["ref_roi"] = ref_gray[ry:ry+rh, rx:rx+rw]
                st.image(bgr2pil(draw_face_boxes(ref_bgr,[ref_faces[0]],ref_gray)),
                         use_column_width=True, caption="Reference face detected ✅")
            else:
                st.warning("No face detected in reference image.")
                st.session_state.pop("ref_roi", None)

        st.markdown("""
        <div style='padding:0.8rem;background:#0f172a;border-radius:8px;border:1px solid #1e293b;font-size:0.78rem;color:#64748b;margin-top:0.8rem'>
        💡 <b>Tip:</b> Upload a clear front-facing photo as reference. Then enable Auto-Refresh for continuous live matching.
        </div>""", unsafe_allow_html=True)

    with live_col:
        st.markdown("#### 🎥 Live Webcam Frame")
        live_auto_match = st.checkbox("🔄 Auto-Refresh (every 2s)", key="live_match_auto")
        live_match_frame = st.camera_input("Capture frame to match", key="live_match_cam")

        if live_auto_match:
            time.sleep(2)
            st.rerun()

        if live_match_frame and "ref_roi" in st.session_state:
            lm_bgr = pil2bgr(Image.open(live_match_frame))
            lm_faces, lm_gray = detect_faces_full(lm_bgr)
            st.image(bgr2pil(draw_face_boxes(lm_bgr,lm_faces,lm_gray)),
                     use_column_width=True, caption=f"Live Frame — {len(lm_faces)} face(s)")

            if len(lm_faces)>0:
                lx,ly,lw,lh = lm_faces[0]
                live_roi = lm_gray[ly:ly+lh, lx:lx+lw]
                score_live = face_match_score(st.session_state["ref_roi"], live_roi)
                st.markdown(render_match_card(score_live), unsafe_allow_html=True)

                # Score history chart
                if "match_history" not in st.session_state:
                    st.session_state["match_history"] = []
                st.session_state["match_history"].append(score_live)
                st.session_state["match_history"] = st.session_state["match_history"][-20:]

                if len(st.session_state["match_history"])>1:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(
                        y=st.session_state["match_history"], mode="lines+markers",
                        line=dict(color="#22c55e", width=2), marker=dict(size=6)
                    ))
                    fig_h.add_hline(y=65, line_dash="dash", line_color="#f59e0b",
                                    annotation_text="Match threshold (65%)")
                    fig_h.update_layout(
                        title="Match Score History (last 20 frames)",
                        height=200, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#94a3b8"), margin=dict(t=30,b=20,l=20,r=20),
                        yaxis=dict(range=[0,100])
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

                if st.button("🗑️ Clear History", key="clear_hist"):
                    st.session_state["match_history"] = []
                    st.rerun()
            else:
                st.warning("No face detected in live frame. Look at the camera.")

        elif live_match_frame and "ref_roi" not in st.session_state:
            st.warning("Upload a reference face first (left panel).")
        else:
            st.markdown("""
            <div style='padding:3rem;text-align:center;color:#475569;border:1px dashed #334155;border-radius:12px'>
                <div style='font-size:3rem'>🔄</div>
                <div style='margin-top:0.5rem'>Upload a reference face → capture live frames → see match %</div>
            </div>""", unsafe_allow_html=True)
