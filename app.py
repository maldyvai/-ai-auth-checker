import streamlit as st
from PIL import Image
import io, os
import numpy as np
import torch
from streamlit_drawable_canvas import st_canvas
from helper import analyze_ela

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VeriCheck AI",
    page_icon="📄",
    layout="wide"
)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("**Photo Tamper Detection & Annotation**")
    threshold = st.slider("ELA sensitivity", 0, 255, 50)
    std_low   = st.slider("Low-risk cutoff", 0.0, 100.0, 10.0)
    std_high  = st.slider("High-risk cutoff", 0.0, 200.0, 35.0)

# ─── Load YOLOv5 via torch.hub ──────────────────────────────────────────────
@st.cache_resource
def load_yolo(weights="best.pt"):
    # this pulls the YOLOv5 repo on first run
    return torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)

yolo = load_yolo()

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#003366;'>VeriCheck AI Platform</h1><hr>", unsafe_allow_html=True)

# ─── Main ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a JPG/PNG invoice or photo", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload a photo to begin.")
else:
    # Load image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    # 1) ELA analysis
    ela_img, hl_img, std, regs = analyze_ela(img, threshold=threshold, quality=60)
    st.markdown(f"**ELA Std Dev:** {std:.2f} **Regions:** {regs}")
    if std > std_high:
        st.error("⚠️ High Risk")
    elif std < std_low:
        st.success("✅ Low Risk")
    else:
        st.warning("🔍 Uncertain")

    st.subheader("ELA Image")
    st.image(ela_img, use_container_width=True)
    st.subheader("Highlight Map")
    st.image(hl_img, use_container_width=True)

    # 2) YOLOv5 detection
    st.subheader("AI-Detected Tampered Regions")
    results = yolo(np.array(img))
    box_img = img.copy()
    draw = Image.Draw.Draw(box_img)
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.3:
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
    st.image(box_img, use_container_width=True)

    # 3) Feedback & annotation
    st.subheader("Was the AI’s detection correct?")
    feedback = st.radio("", ["Yes", "No"], key="fb")
    if feedback == "No":
        st.markdown("#### Draw the *correct* tampered regions:")
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
        if st.button("Save Corrections"):
            shapes = canvas_result.json_data["objects"]
            os.makedirs("new_annotations/images", exist_ok=True)
            os.makedirs("new_annotations/labels", exist_ok=True)
            fn = uploaded.name
            # Save the original image for training
            img.save(f"new_annotations/images/{fn}")
            # Save YOLO-format label
            lbl = os.path.splitext(fn)[0] + ".txt"
            with open(f"new_annotations/labels/{lbl}", "w") as f:
                for o in shapes:
                    l, t = o["left"], o["top"]
                    w, h = o["width"], o["height"]
                    x_c = (l + w/2) / img.width
                    y_c = (t + h/2) / img.height
                    w_n = w / img.width
                    h_n = h / img.height
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
            st.success("Corrections saved! Use `new_annotations/` for your next training run.")

    else:
        st.success("Great! No corrections needed.")

