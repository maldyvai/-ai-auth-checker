import streamlit as st
from PIL import Image
import io, os
import numpy as np
from datetime import datetime
from helper import analyze_ela
from streamlit_drawable_canvas import st_canvas

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="VeriCheck AI", page_icon="📄", layout="wide")

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("**Image Authenticity & Annotation**")
    threshold = st.slider("ELA sensitivity", 0, 255, 50)
    std_low   = st.slider("Low-risk cutoff", 0.0, 100.0, 10.0)
    std_high  = st.slider("High-risk cutoff", 0.0, 200.0, 35.0)

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("<h2 style='color:#003366;'>VeriCheck AI Platform</h2><hr>", unsafe_allow_html=True)

# ─── Main ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a JPG/PNG image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload a photo to analyze.")
else:
    name = uploaded.name
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    # 1) ELA analysis
    ela_img, hl_img, std, regions = analyze_ela(img, threshold=threshold, quality=60)
    st.markdown(f"**ELA Std Dev:** {std:.2f} **Regions:** {regions}")
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

    # 2) Feedback & correction
    st.markdown("### Was this result correct?")
    fb = st.radio("", ["Yes","No"], key="fb")
    if fb == "No":
        st.markdown("#### Draw the *correct* tampered regions:")
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=img,
            update_streamlit=True,
            drawing_mode="rect",
            key="canvas",
        )
        if st.button("Save Corrections"):
            shapes = canvas.json_data.get("objects", [])
            # ensure folders
            os.makedirs("new_annotations/images", exist_ok=True)
            os.makedirs("new_annotations/labels", exist_ok=True)
            # save image
            img.save(f"new_annotations/images/{name}")
            # save YOLO-format labels
            base = os.path.splitext(name)[0]
            with open(f"new_annotations/labels/{base}.txt","w") as f:
                for o in shapes:
                    l,t = o["left"], o["top"]
                    w,h = o["width"], o["height"]
                    x_c = (l + w/2) / img.width
                    y_c = (t + h/2) / img.height
                    w_n = w / img.width
                    h_n = h / img.height
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
            st.success("Corrections saved! Check `new_annotations/` for training data.")
    else:
        st.success("Great—no corrections needed!")

    # 3) View feedback log
    st.markdown("### Feedback Log")
    logp = "new_annotations/labels"
    count = len(os.listdir("new_annotations/images")) if os.path.exists("new_annotations/images") else 0
    st.write(f"Collected {count} corrected examples.")

