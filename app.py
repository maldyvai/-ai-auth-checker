import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import io
import fitz  # PyMuPDF
import os

# Custom CSS for streamlit
st.markdown("""
    <style>
        body {
            background-color: #F2F2F2;
        }
        .title {
            color: #003366;
        }
        .button {
            background-color: #28A745;
            color: white;
        }
        .button:hover {
            background-color: #1e7e34;
        }
        .stButton>button {
            border-radius: 5px;
            font-size: 16px;
            padding: 10px 20px;
        }
        .sidebar .sidebar-content {
            background-color: #003366;
            color: white;
        }
        .stTextInput>label {
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# Custom Header and Title
st.markdown("<h1 class='title'>AI Document & Image Authenticity Checker</h1>", unsafe_allow_html=True)
st.caption("Detect forged or manipulated documents and images using AI technology.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

def analyze_ela(image):
    # Convert image to numpy array for ELA analysis
    image_np = np.array(image)
    
    # Apply ELA (Error Level Analysis)
    image_pil = image.copy()
    enhancer = ImageEnhance.Contrast(image_pil)
    enhanced_image = enhancer.enhance(2.0)
    
    # Perform ELA and highlight suspicious regions
    ela_image = ImageChops.difference(image_pil, enhanced_image)
    std = np.std(np.array(ela_image))
    regions = np.count_nonzero(np.array(ela_image) > 50)
    
    highlight_image = np.array(image_pil)
    highlight_image[ela_image > 50] = [255, 0, 0]  # Highlight suspicious areas in red
    highlight_image_pil = Image.fromarray(highlight_image)
    
    return ela_image, highlight_image_pil, std, regions

if uploaded_file:
    filename = uploaded_file.name.lower()

    # Image Processing section
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        ela_img, highlight_img, std, regions = analyze_ela(image)
        st.image(ela_img, caption="🔎 ELA Image", width=350)
        st.image(highlight_img, caption="📍 Highlighted Suspicious Regions", use_column_width=True)

        st.subheader("📊 Results:")
        st.write(f"ELA Std Dev: `{std:.2f}` | Suspicious Areas: `{regions}`")

        # Result feedback based on analysis
        if regions > 3 or std > 35:
            st.error("⚠️ Likely manipulated.")
        elif regions > 0:
            st.warning("⚠️ Some signs of editing.")
        else:
            st.success("✅ Image appears authentic.")

    elif filename.endswith(".pdf"):
        # PDF Processing section
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        st.success(f"📄 PDF loaded: {len(doc)} pages")

        for page_num in range(len(doc)):
            page = doc[page_num]
            st.header(f"Page {page_num + 1}")

            text = page.get_text()
            if text.strip():
                st.subheader("🔤 Extracted Text")
                st.code(text)
            else:
                st.info("No text detected on this page.")

            # Handle embedded images inside PDFs
            image_list = page.get_images(full=True)
            if image_list:
                st.subheader("🖼 Embedded Images")
                for i, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    st.image(image_pil, caption=f"Original Image {i+1}", width=400)

                    ela_img, highlight_img, std, regions = analyze_ela(image_pil)
                    st.image(ela_img, caption="ELA Image", width=300)
                    st.image(highlight_img, caption="Suspicious Regions", width=300)

                    st.write(f"📈 ELA Std Dev: `{std:.2f}` | Suspicious Areas: `{regions}`")

                    if regions > 3 or std > 35:
                        st.error("⚠️ Likely manipulated.")
                    elif regions > 0:
                        st.warning("⚠️ Some signs of editing.")
                    else:
                        st.success("✅ Appears authentic.")

    # Feedback Section
    feedback = st.radio("📢 Was this analysis helpful?", ["Yes", "No", "Not Sure"])
    if st.button("Submit Feedback", key="feedback", help="Share your feedback on this result"):
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/feedback_log.csv", "a") as f:
                f.write(f"{filename},{feedback}\n")
            st.success("✅ Feedback submitted. Thank you!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
else:
    st.info("Please upload a JPEG/PNG image or a PDF document to begin analysis.")
