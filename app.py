import streamlit as st
from PIL import Image
import io
import os
import fitz  # PyMuPDF
import numpy as np
import cv2
import torch
import base64
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

from helper import analyze_ela

# ─── Page Config ───────────────────────────────────────────────────────────
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
    threshold = st.slider("ELA mask threshold", 0, 255, 50,
                          help="Lower → more sensitive")
    std_low   = st.slider("Low-risk Std Dev cutoff", 0.0, 100.0, 10.0,
                          help="Below = ✓ authentic")
    std_high  = st.slider("High-risk Std Dev cutoff", 0.0, 200.0, 35.0,
                          help="Above = ⚠️ tampered")

# ─── Load YOLOv5 Model ─────────────────────────────────────────────────────
@st.cache_resource
def load_yolo(weights="best.pt"):
    try:
        return torch.hub.load(
            'ultralytics/yolov5', 'custom',
            path=weights, force_reload=True
        )
    except:
        return None

yolo = load_yolo()
yolo_available = yolo is not None

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#003366;'>VeriCheck AI Platform</h2><hr>",
    unsafe_allow_html=True
)

# ─── UPLOAD & ANALYZE ─────────────────────────────────────────────────────
if menu == "Upload & Analyze":
    st.markdown("## Upload Invoice (Photo or PDF)")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","pdf"])
    if uploaded:
        name = uploaded.name
        st.markdown(f"**File:** {name}")

        # ── PHOTO branch ─────────────────────────────────────────────
        if name.lower().endswith((".jpg",".jpeg",".png")):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Original", use_container_width=True)

            # Run ELA
            ela_img, hl_img, std, regions = analyze_ela(
                img, threshold=threshold, quality=60
            )

            # Show metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Std Dev", f"{std:.2f}")
            c2.metric("Pixels", f"{regions}")
            if std > std_high:   score = "⚠️ High Risk"
            elif std < std_low:  score = "✅ Low Risk"
            else:                score = "🔍 Uncertain"
            c3.metric("Tamper Score", score)

            # Show ELA & highlights
            tabs = st.tabs(["ELA Image","Highlights"])
            with tabs[0]:
                st.image(ela_img, use_container_width=True)
            with tabs[1]:
                st.image(hl_img, use_container_width=True)

            # ML-based detection
            if yolo_available:
                st.markdown("### 🤖 ML Detector")
                res = yolo(np.array(img))
                det = res.xyxy[0].cpu().numpy()
                box_img = np.array(img).copy()
                for *b, conf, cls in det:
                    if conf > 0.3:
                        x1,y1,x2,y2 = map(int, b)
                        cv2.rectangle(box_img,(x1,y1),(x2,y2),(0,255,0),2)
                st.image(box_img, caption="YOLO Boxes", use_container_width=True)
            else:
                st.info("No ML model loaded.")

            # Feedback & annotation canvas
            st.markdown("#### Was this result correct?")
            feedback = st.radio("", ["Yes","No"], key=name)
            if feedback == "No":
                st.markdown("### ✏️ Mark the tampered regions")

                # Encode PIL image as base64 data-URI
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#ff0000",
                    background_image_url=uri,
                    update_streamlit=True,
                    height=img.height,
                    width=img.width,
                    drawing_mode="rect",
                    key=name+"_canvas",
                )
                if st.button("Save Annotations", key=name+"_annotate"):
                    shapes = canvas_result.json_data["objects"]
                    os.makedirs("new_annotations/images", exist_ok=True)
                    os.makedirs("new_annotations/labels", exist_ok=True)
                    img.save(f"new_annotations/images/{name}")
                    lbl = os.path.splitext(name)[0] + ".txt"
                    with open(f"new_annotations/labels/{lbl}", "w") as f:
                        for obj in shapes:
                            l, t = obj["left"], obj["top"]
                            w, h = obj["width"], obj["height"]
                            x_c = (l + w/2) / img.width
                            y_c = (t + h/2) / img.height
                            w_n = w / img.width
                            h_n = h / img.height
                            f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                    st.success("Annotations saved!")

            else:
                if st.button("Submit Feedback", key=name+"_btn"):
                    os.makedirs("logs", exist_ok=True)
                    path = "logs/feedback_log.csv"
                    header = not os.path.exists(path)
                    with open(path,"a") as f:
                        if header:
                            f.write("timestamp,filename,std,regions,score,feedback\n")
                        f.write(f"{datetime.utcnow().isoformat()},"
                                f"{name},{std:.2f},{regions},{score},{feedback}\n")
                    st.success("Thank you—feedback recorded!")

        # ── PDF branch ───────────────────────────────────────────────
        elif name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            st.success(f"PDF with {len(doc)} pages")
            for i, page in enumerate(doc):
                st.markdown(f"### Page {i+1}")
                txt = page.get_text().strip() or "No text"
                st.code(txt)
                for img_meta in page.get_images(full=True):
                    xref = img_meta[0]
                    base = doc.extract_image(xref)
                    page_img = Image.open(io.BytesIO(base["image"])).convert("RGB")
                    st.image(page_img, width=300)
                    ela_i, hl_i, s, r = analyze_ela(page_img, threshold=threshold, quality=60)
                    st.image(ela_i, use_container_width=True)
                    st.image(hl_i, use_container_width=True)
                    st.write(f"Std Dev: {s:.2f} | Pixels: {r}")
                    if s>std_high:   st.error("⚠️ Tampered")
                    elif s<std_low:  st.success("✅ Authentic")
                    else:            st.warning("🔍 Uncertain")

elif menu == "Log":
    st.markdown("## Feedback Log")
    path = "logs/feedback_log.csv"
    if os.path.exists(path):
        lines = open(path).read().splitlines()
        hdr = lines[0].split(",")
        rows = [r.split(",") for r in lines[1:]]
        st.dataframe(rows, columns=hdr)
    else:
        st.info("No feedback logged yet.")
