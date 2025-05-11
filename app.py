import streamlit as st
from PIL import Image
import io
import os
import fitz  # PyMuPDF
import numpy as np
import cv2
import torch

from helper import analyze_ela

# ─── Streamlit Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="VeriCheck AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar Navigation & Sliders ─────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("Document & Image Authenticity Checker")
    st.markdown("---")

    # Navigation
    menu = st.radio("Menu", ["Upload & Analyze", "Log"])

    # ELA & Std-Dev tuning sliders
    threshold = st.slider(
        "ELA mask threshold", min_value=0, max_value=255, value=50,
        help="Lower → more sensitive to small differences"
    )
    std_low = st.slider(
        "Low-risk Std Dev cutoff", min_value=0.0, max_value=100.0, value=10.0,
        help="Std Dev below this is ✓ authentic"
    )
    std_high = st.slider(
        "High-risk Std Dev cutoff", min_value=0.0, max_value=200.0, value=35.0,
        help="Std Dev above this flags ⚠️ tampered"
    )

# ─── Load YOLOv5 Model ─────────────────────────────────────────────────────
@st.cache_resource
def load_yolo_model(weights_path="best.pt"):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
        return model
    except:
        return None

yolo = load_yolo_model()
yolo_available = yolo is not None

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#003366; margin-bottom: 0;'>VeriCheck AI Platform</h2><hr>",
    unsafe_allow_html=True
)

# ─── Upload & Analyze ─────────────────────────────────────────────────────
if menu == "Upload & Analyze":
    st.markdown("## Upload an Invoice (Photo or PDF)")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","pdf"])

    if uploaded:
        name = uploaded.name.lower()

        # ─ Photo Path ────────────────────────────────────────────────────────
        if name.endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Original Image", use_container_width=True)

            # Run ELA with tuned threshold
            ela_img, hl_img, std, regions = analyze_ela(img, threshold=threshold)

            # Display metrics
            st.markdown("### 🔍 ELA Analysis Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("ELA Std Dev", f"{std:.2f}")
            c2.metric("Suspicious Pixels", f"{regions}")
            # Use your slider cutoffs
            if std > std_high:
                score = "⚠️ High Risk"
            elif std < std_low:
                score = "✅ Low Risk"
            else:
                score = "🔍 Uncertain"
            c3.metric("Tamper Score", score)

            # Show ELA images
            tabs = st.tabs(["ELA Image", "Highlights"])
            with tabs[0]:
                st.image(ela_img, use_container_width=True)
            with tabs[1]:
                st.image(hl_img, use_container_width=True)

            # ML-based detection
            if yolo_available:
                st.markdown("### 🤖 ML-Based Detection")
                results = yolo(np.array(img))
                det = results.xyxy[0].cpu().numpy()
                box_img = np.array(img).copy()
                for *box, conf, cls in det:
                    if conf > 0.3:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(box_img, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(box_img, f"{conf:.2f}", (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                st.image(box_img, caption="Detected Tampered Regions", use_container_width=True)
            else:
                st.info("ML model not loaded; only ELA analysis shown.")

        # ─ PDF Path ──────────────────────────────────────────────────────────
        elif name.endswith(".pdf"):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            st.success(f"PDF loaded: {len(doc)} pages")
            for i, page in enumerate(doc):
                st.markdown(f"### 📄 Page {i+1}")
                text = page.get_text()
                st.code(text or "No extractable text.")

                imgs = page.get_images(full=True)
                for idx, img_meta in enumerate(imgs):
                    xref = img_meta[0]
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]
                    page_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    st.image(page_img, caption=f"Page Image {idx+1}", width=300)

                    ela_img, hl_img, std, regions = analyze_ela(page_img, threshold=threshold)
                    st.image(ela_img, caption="ELA Analysis", use_container_width=True)
                    st.image(hl_img, caption="Highlights", use_container_width=True)
                    st.write(f"**Std Dev:** `{std:.2f}` | **Pixels:** `{regions}`")
                    if std > std_high:
                        st.error("⚠️ Likely manipulated.")
                    elif std < std_low:
                        st.success("✅ Appears authentic.")
                    else:
                        st.warning("🔍 Uncertain.")

# ─── Feedback Log
