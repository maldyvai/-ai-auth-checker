import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import fitz  # PyMuPDF
import io
import os
from helper import analyze_ela

# Streamlit page config
st.set_page_config(
    page_title="VeriCheck AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("Document & Image Authenticity")
    st.markdown("---")
    menu = st.radio("Menu", ["Upload & Analyze", "Log"])

# Header
st.markdown("<h2 style='color:#003366;'>VeriCheck AI Platform</h2><hr>", unsafe_allow_html=True)

if menu == "Upload & Analyze":
    st.markdown("## Upload File")
    uploaded_file = st.file_uploader("Choose image or PDF", type=["jpg","jpeg","png","pdf"])
    if uploaded_file:
        filename = uploaded_file.name.lower()
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original", use_container_width=True)
            ela_img, highlight_img, std, regions = analyze_ela(image)
            st.markdown("### Results")
            col1, col2 = st.columns(2)
            col1.metric("ELA Std Dev", f"{std:.2f}")
            col2.metric("Pixels > threshold", f"{regions}")
            tabs = st.tabs(["ELA", "Highlights"])
            with tabs[0]:
                st.image(ela_img, use_container_width=True)
            with tabs[1]:
                st.image(highlight_img, use_container_width=True)
        elif filename.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            st.success(f"PDF pages: {len(doc)}")
            for i in range(len(doc)):
                page = doc[i]
                st.markdown(f"### Page {i+1}")
                text = page.get_text()
                st.code(text if text.strip() else "No text")
                img_list = page.get_images(full=True)
                for idx, img in enumerate(img_list):
                    xref = img[0]
                    b = doc.extract_image(xref)
                    img_data = b["image"]
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    st.image(image, caption=f"Image {idx+1}", width=300)
                    ela_img, highlight_img, std, regions = analyze_ela(image)
                    st.image(ela_img, caption="ELA", use_container_width=True)
                    st.image(highlight_img, caption="Highlights", use_container_width=True)
elif menu == "Log":
    st.markdown("## Feedback Log")
    log_path = "logs/feedback_log.csv"
    if os.path.exists(log_path):
        data = open(log_path).read()
        st.code(data, language="csv")
    else:
        st.info("No feedback yet.")
