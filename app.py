import streamlit as st
from PIL import Image, ImageDraw
import os
import numpy as np
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from helper import analyze_ela  # your existing ELA routine

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VeriCheck AI",
    page_icon="📄",
    layout="wide"
)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("**Invoice/Image Tamper Detection**")
    threshold = st.slider("ELA sensitivity", 0, 255, 50)
    std_low   = st.slider("Low-risk cutoff", 0.0, 100.0, 10.0)
    std_high  = st.slider("High-risk cutoff", 0.0, 200.0, 35.0)

# ─── Load your custom YOLO model ────────────────────────────────────────────
@st.cache_resource
def load_model(path="best.pt"):
    return YOLO(path)

model = load_model()

# ─── App Header ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#003366;'>VeriCheck AI Platform</h1><hr>",
            unsafe_allow_html=True)

# ─── Upload & Analyze ───────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a JPG/PNG invoice or photo", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    # 1) ELA analysis
    ela_img, hl_img, std, regions = analyze_ela(img, threshold=threshold, quality=60)
    st.markdown(f"**ELA Std Dev:** {std:.2f} **Regions:** {regions}")
    if std > std_high:
        st.error("⚠️ High Risk (likely tampered)")
    elif std < std_low:
        st.success("✅ Low Risk (likely authentic)")
    else:
        st.warning("🔍 Uncertain")

    st.subheader("ELA Image")
    st.image(ela_img, use_container_width=True)
    st.subheader("Highlight Map")
    st.image(hl_img, use_container_width=True)

    # 2) YOLO detection
    st.subheader("AI-Detected Tampered Regions")
    results = model(np.array(img))
    box_img = img.copy()
    draw = ImageDraw.Draw(box_img)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if box.conf[0] > 0.3:
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
    st.image(box_img, use_container_width=True)

    # 3) User feedback & correction
    st.markdown("### Was the AI’s result correct?")
    feedback = st.radio("", ["Yes", "No"], key="fb")
    if feedback == "No":
        st.markdown("#### Please draw the *correct* tampered regions below:")
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
            # Save image
            filename = uploaded.name
            img.save(f"new_annotations/images/{filename}")
            # Save YOLO txt
            base = os.path.splitext(filename)[0]
            with open(f"new_annotations/labels/{base}.txt", "w") as f:
                for obj in shapes:
                    l, t = obj["left"], obj["top"]
                    w, h = obj["width"], obj["height"]
                    x_c = (l + w/2) / img.width
                    y_c = (t + h/2) / img.height
                    w_n = w / img.width
                    h_n = h / img.height
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
            st.success("Your corrections have been saved for retraining!")
else:
    st.info("Upload an image to get started.")
