import streamlit as st
from PIL import Image
import io
import os
import fitz  # PyMuPDF
import numpy as np
import cv2
import torch
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

# ─── Load YOLOv5 Model ─────────────────────────────────────────────────────
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
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded:
        name = uploaded.name
        st.markdown(f"**File:** {name}")

        # ── PHOTO branch ─────────────────────────────────────────────
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Original", use_container_width=True)

            # ELA with stronger JPEG compression
            ela_img, hl_img, std, regions = analyze_ela(
                img,
                threshold=threshold,
                quality=60
            )

            st.markdown("### 🔍 ELA Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Std Dev", f"{std:.2f}")
            c2.metric("Pixels", f"{regions}")
            if std > std_high:
                score = "⚠️ High Risk"
            elif std < std_low:
                score = "✅ Low Risk"
            else:
                score = "🔍 Uncertain"
            c3.metric("Tamper Score", score)

            tabs = st.tabs(["ELA Image", "Highlights"])
            with tabs[0]:
                st.image(ela_img, use_container_width=True)
            with tabs[1]:
                st.image(hl_img, use_container_width=True)

            # ML-Based Detection
            if yolo_available:
                st.markdown("### 🤖 ML Detector")
                results = yolo(np.array(img))
                det = results.xyxy[0].cpu().numpy()
                box_img = np.array(img).copy()
                for *b, conf, cls in det:
                    if conf > 0.3:
                        x1, y1, x2, y2 = map(int, b)
                        cv2.rectangle(box_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                st.image(box_img, caption="YOLO Boxes", use_container_width=True)
            else:
                st.info("No ML model loaded.")

            # ── Feedback & Annotation Canvas ────────────────────────────
            st.markdown("#### Was this result correct?")
            feedback = st.radio("", ["Yes", "No"], key=name)
            if feedback == "No":
                st.markdown("### ✏️ Mark the tampered regions")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#ff0000",
                    background_image=img,
                    update_streamlit=True,
                    height=img.height,
                    width=img.width,
                    drawing_mode="rect",
                    key="canvas",
                )
                if st.button("Save Annotations", key=name + "_annotate"):
                    shapes = canvas_result.json_data["objects"]
                    os.makedirs("new_annotations/images", exist_ok=True)
                    os.makedirs("new_annotations/labels", exist_ok=True)
                    # Save image
                    img.save(f"new_annotations/images/{name}")
                    # Write YOLO .txt
                    label_path = f"new_annotations/labels/{os.path.splitext(name)[0]}.txt"
                    with open(label_path, "w") as ftxt:
                        for obj in shapes:
                            left, top = obj["left"], obj["top"]
                            w, h = obj["width"], obj["height"]
                            x_c = (left + w/2) / img.width
                            y_c = (top + h/2) / img.height
                            w_n = w / img.width
                            h_n = h / img.height
                            ftxt.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                    st.success("Annotations saved for retraining!")
            else:
                if st.button("Submit Feedback", key=name + "_btn"):
                    os.makedirs("logs", exist_ok=True)
                    log_path = "logs/feedback_log.csv"
                    header = not os.path.exists(log_path)
                    with open(log_path, "a") as f:
                        if header:
                            f.write("timestamp,filename,std,regions,score,feedback\n")
                        f.write(f"{datetime.utcnow().isoformat()},{name},"
                                f"{std:.2f},{regions},{score},{feedback}\n")
                    st.success("Thanks—feedback recorded.")

        # ── PDF branch ───────────────────────────────────────────────
        elif name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            st.success(f"PDF with {len(doc)} pages")
            for i, page in enumerate(doc):
                st.markdown(f"### Page {i+1}")
                txt = page.get_text().strip() or "No text"
                st.code(txt)

                imgs = page.get_images(full=True)
                for idx, img_meta in enumerate(imgs):
                    xref = img_meta[0]
                    base = doc.extract_image(xref)
                    page_img = Image.open(io.BytesIO(base["image"])).convert("RGB")
                    st.image(page_img, caption=f"Image {idx+1}", width=300)

                    ela_img, hl_img, std, regions = analyze_ela(
                        page_img,
                        threshold=threshold,
                        quality=60
                    )
                    st.image(ela_img, caption="ELA Analysis", use_container_width=True)
                    st.image(hl_img, caption="Highlights", use_container_width=True)
                    st.write(f"**Std Dev:** {std:.2f} | **Pixels:** {regions}")
                    if std > std_high:
                        st.error("⚠️ Likely manipulated.")
                    elif std < std_low:
                        st.success("✅ Authentic")
                    else:
                        st.warning("🔍 Uncertain")

# ─── FEEDBACK LOG VIEWER ──────────────────────────────────────────────────
elif menu == "Log":
    st.markdown("## Feedback Log")
    log_path = "logs/feedback_log.csv"
    if os.path.exists(log_path):
        lines = open(log_path).read().splitlines()
        header = lines[0].split(",")
        data = [row.split(",") for row in lines[1:]]
        st.dataframe(data, columns=header)
    else:
        st.info("No feedback logged yet.")
