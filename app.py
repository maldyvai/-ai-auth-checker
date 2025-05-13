import streamlit as st
from PIL import Image
import io
import os
import fitz    # PyMuPDF
import numpy as np
from datetime import datetime

from helper import analyze_ela

st.set_page_config(
    page_title="VeriCheck AI",
    page_icon="📄",
    layout="wide"
)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("Authenticity Checker")
    menu = st.radio("Menu", ["Analyze", "Feedback Log"])
    threshold = st.slider("ELA threshold", 0, 255, 50)
    std_low = st.slider("Low‐risk cutoff", 0.0, 100.0, 10.0)
    std_high = st.slider("High‐risk cutoff", 0.0, 200.0, 35.0)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("<h2 style='color:#003366;'>VeriCheck AI</h2><hr>", unsafe_allow_html=True)

if menu == "Analyze":
    uploaded = st.file_uploader("Upload JPG/PNG or PDF", type=["jpg","png","jpeg","pdf"])
    if not uploaded:
        st.info("Please upload a file.")
    else:
        name = uploaded.name
        st.markdown(f"**File:** {name}")

        # ── Image branch ────────────────────────────────────────────────
        if name.lower().endswith(("jpg","jpeg","png")):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

            ela_img, hl_img, std, regions = analyze_ela(img, threshold=threshold, quality=60)

            st.write(f"**Std Dev:** {std:.2f} | **Pixels:** {regions}")
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

            # Feedback
            st.markdown("Was this correct?")
            choice = st.radio("", ["Yes","No"], key=name)
            if st.button("Submit Feedback", key=name+"_fb"):
                os.makedirs("logs", exist_ok=True)
                path = "logs/feedback.csv"
                header = not os.path.exists(path)
                with open(path,"a") as f:
                    if header:
                        f.write("timestamp,filename,std,regions,choice\n")
                    f.write(f"{datetime.utcnow().isoformat()},{name},{std:.2f},{regions},{choice}\n")
                st.success("Feedback recorded.")

        # ── PDF branch ────────────────────────────────────────────────────
        else:
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            st.success(f"PDF loaded: {len(doc)} pages")
            for i, page in enumerate(doc, start=1):
                st.subheader(f"Page {i}")
                txt = page.get_text().strip() or "No text"
                st.code(txt)
                for img_meta in page.get_images(full=True):
                    xref = img_meta[0]
                    base = doc.extract_image(xref)
                    page_img = Image.open(io.BytesIO(base["image"])).convert("RGB")
                    st.image(page_img, width=300, caption="Embedded image")
                    ela_i, hl_i, std_i, reg_i = analyze_ela(page_img, threshold=threshold, quality=60)
                    st.write(f"Std Dev: {std_i:.2f} | Pixels: {reg_i}")
                    if std_i > std_high:
                        st.error("⚠️ High Risk")
                    elif std_i < std_low:
                        st.success("✅ Low Risk")
                    else:
                        st.warning("🔍 Uncertain")

elif menu == "Feedback Log":
    st.subheader("Feedback Log")
    path = "logs/feedback.csv"
    if not os.path.exists(path):
        st.info("No feedback yet.")
    else:
        lines = open(path).read().splitlines()
        header = lines[0].split(",")
        data = [row.split(",") for row in lines[1:]]
        st.dataframe(data, columns=header)
