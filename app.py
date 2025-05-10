import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import io
import fitz  # PyMuPDF
import os

# Set Streamlit page config
st.set_page_config(
    page_title="VeriCheck AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for light modern theme
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #005bb5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("Your Document & Image Authenticity Assistant")
    st.markdown("---")
    st.markdown("**Menu**")
    menu = st.radio("Go to", ["📤 Upload & Analyze", "🧾 Log", "⚙️ Settings"])

# Header
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h2 style="color:#003366; margin-bottom: 0;">📄 VeriCheck AI Platform</h2>
        <span style="color: #888; font-size: 0.9em;">Trusted by professionals</span>
    </div>
    <hr>
""", unsafe_allow_html=True)

def analyze_ela(image):
    image_np = np.array(image)
    image_pil = image.copy()
    enhancer = ImageEnhance.Contrast(image_pil)
    enhanced_image = enhancer.enhance(2.0)
    ela_image = ImageChops.difference(image_pil, enhanced_image)
    std = np.std(np.array(ela_image))
    regions = np.count_nonzero(np.array(ela_image) > 50)
    highlight_image = np.array(image_pil)
    highlight_image[np.array(ela_image) > 50] = [255, 0, 0]
    highlight_image_pil = Image.fromarray(highlight_image)
    return ela_image, highlight_image_pil, std, regions

if menu == "📤 Upload & Analyze":
    st.markdown("## Upload File for Authenticity Check")
    uploaded_file = st.file_uploader("Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file:
        filename = uploaded_file.name.lower()

        if filename.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🖼 Original Image", use_column_width=True)

            ela_img, highlight_img, std, regions = analyze_ela(image)

            st.markdown("### 🔍 Analysis Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("ELA Std Dev", f"{std:.2f}")
            col2.metric("Suspicious Pixels", f"{regions}")
            risk = "⚠️ Moderate Risk" if std > 35 else ("✅ Low Risk" if std < 10 else "🔍 Uncertain")
            col3.metric("Tamper Score", risk)

            tab1, tab2 = st.tabs(["ELA Image", "Highlights"])
            with tab1:
                st.image(ela_img, caption="ELA Image", use_column_width=True)
            with tab2:
                st.image(highlight_img, caption="Highlighted Regions", use_column_width=True)

        elif filename.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            st.success(f"📄 PDF loaded with {len(doc)} pages")

            for page_num in range(len(doc)):
                page = doc[page_num]
                st.markdown(f"### 📄 Page {page_num + 1}")
                text = page.get_text()
                if text.strip():
                    st.code(text)
                else:
                    st.info("No extractable text on this page.")

                image_list = page.get_images(full=True)
                if image_list:
                    for i, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                        st.image(image_pil, caption=f"🖼 Extracted Image {i+1}", width=350)
                        ela_img, highlight_img, std, regions = analyze_ela(image_pil)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(ela_img, caption="ELA", use_column_width=True)
                        with col2:
                            st.image(highlight_img, caption="Highlights", use_column_width=True)

                        st.write(f"**ELA Std Dev:** `{std:.2f}` | **Suspicious Areas:** `{regions}`")

                        if regions > 3 or std > 35:
                            st.error("⚠️ Likely manipulated.")
                        elif regions > 0:
                            st.warning("⚠️ Some signs of editing.")
                        else:
                            st.success("✅ Appears authentic.")

elif menu == "🧾 Log":
    st.subheader("📊 Feedback & Activity Log")
    if os.path.exists("logs/feedback_log.csv"):
        with open("logs/feedback_log.csv", "r") as f:
            content = f.read()
            st.code(content, language="csv")
    else:
        st.info("No feedback logged yet.")

elif menu == "⚙️ Settings":
    st.subheader("⚙️ Application Settings")
    st.write("(More customization options coming soon.)")
