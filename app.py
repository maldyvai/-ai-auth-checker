import sys
import streamlit.web.elements.image as _web_image
sys.modules['streamlit.elements.image'] = _web_image

import streamlit as st
from PIL import Image, ImageDraw
import io
import os
import fitz  # PyMuPDF
import numpy as np
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
    std_low = st.slider("Low-risk Std Dev cutoff", 0.0, 100.0, 10.0,
                        help="Below = ✓ authentic")
    std_high = st.slider("High-risk Std Dev cutoff", 0.0, 200.0, 35.0,
                         help="Above = ⚠️ tampered")

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#003366;'>VeriCheck AI Platform</h2><hr>",
    unsafe_allow_html=True
)

# ─── UPLOAD & ANALYZE ─────────────────────────────────────────────────────
if menu == "Upload & Analyze":
    st.markdown("## Upload Invoice (Photo or PDF)")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])
    if not uploaded:
        st.info("Please upload an image or PDF to begin.")
    else:
        name = uploaded.name
        st.markdown(f"**File:** {name}")

        # ── IMAGE branch ────────────────────────────────────────────
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Original", use_container_width=True)

            # Perform ELA
            ela_img, hl_img, std, regions = analyze_ela(
                img, threshold=threshold, quality=60
            )

            # Show metrics
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

            # Display ELA results
            tabs = st.tabs(["ELA Image", "Highlights"])
            with tabs[0]:
                st.image(ela_img, use_container_width=True)
            with tabs[1]:
                st.image(hl_img, use_container_width=True)

            # Annotation feedback
            st.markdown("#### Was this result correct?")
            feedback = st.radio("", ["Yes", "No"], key=name)
            if feedback == "No":
                st.markdown("### ✏️ Mark the tampered regions")
                canvas = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#ff0000",
                    background_image=img,
                    update_streamlit=True,
                    height=img.height,
                    width=img.width,
                    drawing_mode="rect",
                    key=name + "_canvas",
                )
                if st.button("Save Annotations", key=name + "_annotate"):
                    shapes = canvas.json_data["objects"]
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
                    st.success("Thank you—feedback recorded!")

        # ── PDF branch ────────────────────────────────────────────────
        elif name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            st.success(f"PDF with {len(doc)} pages")
            for i, page in enumerate(doc, start=1):
                st.markdown(f"### Page {i}")
                text = page.get_text().strip() or "No text found"
                st.code(text)
                for img_meta in page.get_images(full=True):
                    xref = img_meta[0]
                    raw = doc.extract_image(xref)["image"]
                    page_img = Image.open(io.BytesIO(raw)).convert("RGB")
                    st.image(page_img, caption=f"Page {i} Image", width=300)

                    ela2, hl2, std2, reg2 = analyze_ela(
                        page_img, threshold=threshold, quality=60
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Std Dev", f"{std2:.2f}")
                    c2.metric("Pixels", f"{reg2}")
                    if std2 > std_high:
                        score2 = "⚠️ High Risk"
                    elif std2 < std_low:
                        score2 = "✅ Low Risk"
                    else:
                        score2 = "🔍 Uncertain"
                    c3.metric("Score", score2)

                    tabs2 = st.tabs(["ELA", "Highlights"])
                    with tabs2[0]:
                        st.image(ela2, use_container_width=True)
                    with tabs2[1]:
                        st.image(hl2, use_container_width=True)

# ─── FEEDBACK LOG VIEWER ────────────────────────────────────────────────────
elif menu == "Log":
    st.markdown("## Feedback Log")
    lp = "logs/feedback_log.csv"
    if os.path.exists(lp):
        lines = open(lp).read().splitlines()
        hdr = lines[0].split(",")
        data = [r.split(",") for r in lines[1:]]
        st.dataframe(data, columns=hdr)
    else:
        st.info("No feedback logged yet.")
