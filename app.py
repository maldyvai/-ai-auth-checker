import streamlit as st
from PIL import Image
import io, os, base64
import numpy as np
from datetime import datetime
from helper import analyze_ela
from streamlit_drawable_canvas import st_canvas

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="VeriCheck AI", page_icon="📄", layout="wide")

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("**Image Authenticity Checker**")
    threshold = st.slider("ELA sensitivity", 0, 255, 50)
    std_low   = st.slider("Low‐risk cutoff", 0.0, 100.0, 10.0)
    std_high  = st.slider("High‐risk cutoff", 0.0, 200.0, 35.0)

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#003366;'>VeriCheck AI Platform</h2><hr>",
    unsafe_allow_html=True
)

# ─── MAIN ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a JPG/PNG image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload a photo to analyze.")
else:
    name = uploaded.name
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    # 1) ELA analysis
    ela_img, hl_img, std, regions = analyze_ela(
        img, threshold=threshold, quality=60
    )
    st.markdown(f"**Std Dev:** {std:.2f} **Pixels:** {regions}")
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

    # 2) Feedback & Correction
    st.markdown("### Was this result correct?")
    fb = st.radio("", ["Yes", "No"], key="fb")
    if fb == "No":
        st.markdown("#### Draw the *correct* tampered regions:")

        # Convert PIL image to data-URI
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data_uri = (
            "data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode()
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            update_streamlit=True,
            height=img.height,
            width=img.width,
            drawing_mode="rect",
            key="canvas",
        )

        if st.button("Save Corrections"):
            shapes = canvas_result.json_data.get("objects", [])
            os.makedirs("new_annotations/images", exist_ok=True)
            os.makedirs("new_annotations/labels", exist_ok=True)

            # Save the original image
            img.save(f"new_annotations/images/{name}")

            # Save YOLO-format label file
            base = os.path.splitext(name)[0]
            with open(f"new_annotations/labels/{base}.txt", "w") as f:
                for o in shapes:
                    l, t = o["left"], o["top"]
                    w, h = o["width"], o["height"]
                    x_c = (l + w/2) / img.width
                    y_c = (t + h/2) / img.height
                    w_n = w / img.width
                    h_n = h / img.height
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

            st.success("Corrections saved! Check `new_annotations/` for your training data.")

    else:
        st.success("Great—no corrections needed!")

    # 3) Show count of collected corrections
    count = 0
    if os.path.isdir("new_annotations/images"):
        count = len(os.listdir("new_annotations/images"))
    st.markdown(f"**Collected corrected examples:** {count}")

