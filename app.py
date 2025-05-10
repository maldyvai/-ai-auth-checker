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

import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import io
import fitz  # PyMuPDF
import os

def convert_to_ela_image(image, quality=90):
    temp_io = io.BytesIO()
    image.save(temp_io, 'JPEG', quality=quality)
    temp_io.seek(0)
    compressed = Image.open(temp_io)
    ela_image = ImageChops.difference(image, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def analyze_ela(image, threshold_value=30):
    ela = convert_to_ela_image(image)
    ela_np = np.array(ela)
    gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_np = np.array(image.copy())
    suspicious_regions = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
            suspicious_regions += 1
    std_dev = np.std(ela_np)
    return ela, image_np, std_dev, suspicious_regions

st.set_page_config(page_title="AI Authenticity Checker", layout="wide")
st.title("🔍 Unified AI Authenticity Checker")
st.caption("Supports both images and PDFs. Detects possible edits, manipulations, or forgeries.")

uploaded_file = st.file_uploader("📤 Upload an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    filename = uploaded_file.name.lower()

    if filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        ela_img, highlight_img, std, regions = analyze_ela(image)
        st.image(ela_img, caption="🔎 ELA Image", width=350)
        st.image(highlight_img, caption="📍 Highlighted Suspicious Regions", use_column_width=True)

        st.subheader("📊 Results:")
        st.write(f"ELA Std Dev: `{std:.2f}` | Suspicious Areas: `{regions}`")

        if regions > 3 or std > 35:
            st.error("⚠️ Likely manipulated.")
        elif regions > 0:
            st.warning("⚠️ Some signs of editing.")
        else:
            st.success("✅ Image appears authentic.")

    elif filename.endswith(".pdf"):
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
            else:
                st.info("No images detected on this page.")

    feedback = st.radio("📢 Was this analysis helpful?", ["Yes", "No", "Not Sure"])
    if st.button("Submit Feedback"):
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/feedback_log.csv", "a") as f:
                f.write(f"{filename},{feedback}\n")
            st.success("✅ Feedback submitted. Thank you!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")

else:
    st.info("Please upload a JPEG/PNG image or a PDF document to begin analysis.")
