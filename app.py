import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import io
import fitz  # PyMuPDF
import os

# Set Streamlit page config
st.set_page_config(
    page_title="AI Authenticity Checker",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom sidebar
with st.sidebar:
    st.title("🛡️ AI Authenticity Platform")
    st.markdown("Upload documents and images to verify their integrity.")
    st.markdown("---")
    st.markdown("**Built for:**")
    st.markdown("- Insurance providers\n- Financial institutions\n- Legal verification teams")

# Main Title
st.markdown("<h1 style='color:#003366;'>📄 Document & Image Authenticity Checker</h1>", unsafe_allow_html=True)
st.caption("Securely analyze documents for tampering or forgery.")

# File uploader
st.markdown("## 📥 Upload File")
uploaded_file = st.file_uploader("Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"])

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

if uploaded_file:
    filename = uploaded_file.name.lower()

    if filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🗈 Original Image", use_column_width=True)

        ela_img, highlight_img, std, regions = analyze_ela(image)

        st.markdown("### 📊 ELA Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(ela_img, caption="🔍 ELA Analysis", use_column_width=True)
        with col2:
            st.image(highlight_img, caption="📍 Suspicious Regions", use_column_width=True)

        st.write(f"**Standard Deviation:** `{std:.2f}` | **Suspicious Pixels:** `{regions}`")

        if regions > 3 or std > 35:
            st.error("⚠️ Image is likely manipulated.")
        elif regions > 0:
            st.warning("⚠️ Minor signs of editing detected.")
        else:
            st.success("✅ Image appears authentic.")

    elif filename.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        st.success(f"📄 PDF loaded with {len(doc)} pages")

        for page_num in range(len(doc)):
            page = doc[page_num]
            st.markdown(f"### 📜 Page {page_num + 1}")
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

                    st.image(image_pil, caption=f"🗈 Extracted Image {i+1}", width=350)
                    ela_img, highlight_img, std, regions = analyze_ela(image_pil)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(ela_img, caption="🔍 ELA", use_column_width=True)
                    with col2:
                        st.image(highlight_img, caption="📍 Highlights", use_column_width=True)

                    st.write(f"**ELA Std Dev:** `{std:.2f}` | **Suspicious Areas:** `{regions}`")

                    if regions > 3 or std > 35:
                        st.error("⚠️ Likely manipulated.")
                    elif regions > 0:
                        st.warning("⚠️ Some signs of editing.")
                    else:
                        st.success("✅ Appears authentic.")

    feedback = st.radio("🗣️ Was this analysis helpful?", ["Yes", "No", "Not Sure"])
    if st.button("Submit Feedback"):
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/feedback_log.csv", "a") as f:
                f.write(f"{filename},{feedback}\n")
            st.success("✅ Feedback submitted. Thank you!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
else:
    st.info("Upload a JPEG/PNG image or PDF to start analysis.")
