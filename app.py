import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import io
import os
from helper import analyze_ela

# ─── Streamlit Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="VeriCheck AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar Navigation ──────────────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("Document & Image Authenticity Checker")
    st.markdown("---")
    menu = st.radio("Menu", ["Upload & Analyze", "Log"])

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#003366; margin-bottom: 0;'>VeriCheck AI Platform</h2><hr>",
    unsafe_allow_html=True
)

# ─── Upload & Analyze ─────────────────────────────────────────────────────
if menu == "Upload & Analyze":
    st.markdown("## Upload an Invoice (Photo or PDF)")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","pdf"])

    if uploaded:
        name = uploaded.name.lower()

        # ─ Photo Path ────────────────────────────────────────────────────────
        if name.endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Original Image", use_container_width=True)

            ela_img, hl_img, std, regions = analyze_ela(img)

            st.markdown("### Analysis Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("ELA Std Dev", f"{std:.2f}")
            c2.metric("Suspicious Pixels", f"{regions}")
            score = ("⚠️ High Risk" if std > 35
                     else "✅ Low Risk" if std < 10
                     else "🔍 Uncertain")
            c3.metric("Tamper Score", score)

            tabs = st.tabs(["ELA Image", "Highlights"])
            with tabs[0]:
                st.image(ela_img, use_container_width=True)
            with tabs[1]:
                st.image(hl_img, use_container_width=True)

        # ─ PDF Path ──────────────────────────────────────────────────────────
        elif name.endswith(".pdf"):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            st.success(f"PDF loaded: {len(doc)} pages")
            for i, page in enumerate(doc):
                st.markdown(f"### Page {i+1}")
                text = page.get_text()
                st.code(text if text.strip() else "No extractable text.")

                imgs = page.get_images(full=True)
                for idx, img_meta in enumerate(imgs):
                    xref = img_meta[0]
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]
                    page_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    st.image(page_img, caption=f"Page Image {idx+1}", width=300)

                    ela_img, hl_img, std, regions = analyze_ela(page_img)
                    st.image(ela_img, caption="ELA", use_container_width=True)
                    st.image(hl_img, caption="Highlights", use_container_width=True)
                    st.write(f"**Std Dev:** `{std:.2f}` | **Pixels:** `{regions}`")
                    if std > 35:
                        st.error("⚠️ Likely manipulated.")
                    elif regions > 0:
                        st.warning("⚠️ Possible editing detected.")
                    else:
                        st.success("✅ Appears authentic.")

# ─── Feedback Log ─────────────────────────────────────────────────────────
elif menu == "Log":
    st.markdown("## Feedback & Activity Log")
    log_path = "logs/feedback_log.csv"
    if os.path.exists(log_path):
        st.code(open(log_path).read(), language="csv")
    else:
        st.info("No feedback logged yet.")
