import streamlit as st
from PIL import Image
import io
import os
import fitz  # PyMuPDF
import numpy as np
import cv2
import torch
from datetime import datetime

from helper import analyze_ela

# ─── Streamlit Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="VeriCheck AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar & Sliders ────────────────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("Document & Image Authenticity Checker")
    st.markdown("---")
    menu = st.radio("Menu", ["Upload & Analyze", "Log"])
    threshold = st.slider(
        "ELA mask threshold", 0, 255, 50,
        help="Lower → more sensitive"
    )
    std_low = st.slider(
        "Low-risk Std Dev cutoff", 0.0, 100.0, 10.0,
        help="Below = ✓ authentic"
    )
    std_high = st.slider(
        "High-risk Std Dev cutoff", 0.0, 200.0, 35.0,
        help="Above = ⚠️ tampered"
    )

# ─── Load YOLOv5 (if you have best.pt) ────────────────────────────────────
@st.cache_resource
def load_yolo(weights="best.pt"):
    try:
        return torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    except:
        return None

yolo = load_yolo()
yolo_available = yolo is not None

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#003366; margin-bottom: 0;'>VeriCheck AI Platform</h2><hr>",
    unsafe_allow_html=True
)

# ─── UPLOAD & ANALYZE ─────────────────────────────────────────────────────
if menu == "Upload & Analyze":
    st.markdown("## Upload Invoice (Photo or PDF)")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","pdf"])
    if uploaded:
        name = uploaded.name
        st.markdown(f"**File:** {name}")
        
        # IMAGE path
        if name.lower().endswith((".jpg",".jpeg",".png")):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Original", use_container_width=True)

            # ELA
            ela_img, hl_img, std, regions = analyze_ela(img, threshold=threshold)

            st.markdown("### 🔍 ELA Results")
            c1,c2,c3 = st.columns(3)
            c1.metric("Std Dev", f"{std:.2f}")
            c2.metric("Pixels", f"{regions}")
            if std > std_high:
                score = "⚠️ High Risk"
            elif std < std_low:
                score = "✅ Low Risk"
            else:
                score = "🔍 Uncertain"
            c3.metric("Tamper Score", score)

            tabs = st.tabs(["ELA Image","Highlights"])
            with tabs[0]:
                st.image(ela_img, use_container_width=True)
            with tabs[1]:
                st.image(hl_img, use_container_width=True)

            # ML Detector
            if yolo_available:
                st.markdown("### 🤖 ML Detector")
                results = yolo(np.array(img))
                det = results.xyxy[0].cpu().numpy()
                box_img = np.array(img).copy()
                for *b,conf,cls in det:
                    if conf>0.3:
                        x1,y1,x2,y2 = map(int,b)
                        cv2.rectangle(box_img,(x1,y1),(x2,y2),(0,255,0),2)
                st.image(box_img, caption="YOLO Boxes", use_container_width=True)
            else:
                st.info("No ML model loaded.")

            # ── Feedback ────────────────────────────────────────────
            st.markdown("#### Was this result correct?")
            feedback = st.radio("", ["Yes","No"], key=name)
            if st.button("Submit Feedback", key=name+"_btn"):
                os.makedirs("logs", exist_ok=True)
                log_path = "logs/feedback_log.csv"
                header = not os.path.exists(log_path)
                with open(log_path, "a") as f:
                    if header:
                        f.write("timestamp,filename,std,regions,score,feedback\n")
                    f.write(f"{datetime.utcnow().isoformat()},{name},{std:.2f},{regions},{score},{feedback}\n")
                st.success("Thanks—feedback recorded.")

        # PDF path
        elif name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            st.success(f"PDF with {len(doc)} pages")
            for i,page in enumerate(doc):
                st.markdown(f"### Page {i+1}")
                # extract text
                txt = page.get_text().strip() or "No text"
                st.code(txt)
                # extract images
                for img_meta in page.get_images(full=True):
                    xref = img_meta[0]
                    base = doc.extract_image(xref)
                    img = Image.open(io.BytesIO(base["image"])).convert("RGB")
                    st.image(img, width=300)
                    ela_img, hl_img, std, regions = analyze_ela(img, threshold=threshold)
                    st.image(ela_img, use_container_width=True)
                    st.image(hl_img, use_container_width=True)
                    st.write(f"**Std Dev:** {std:.2f} | **Pixels:** {regions}")
                    if std>std_high:
                        st.error("⚠️ Likely manipulated.")
                    elif std<std_low:
                        st.success("✅ Authentic")
                    else:
                        st.warning("🔍 Uncertain")

# ─── LOG VIEWER ────────────────────────────────────────────────────────────
elif menu == "Log":
    st.markdown("## Feedback Log")
    if os.path.exists("logs/feedback_log.csv"):
        st.dataframe(
            open("logs/feedback_log.csv").read().splitlines()[1:],
            columns=open("logs/feedback_log.csv").read().splitlines()[0].split(",")
        )
    else:
        st.info("No feedback yet.")
