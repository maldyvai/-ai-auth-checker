import streamlit as st
from PIL import Image, ImageDraw
import os, io, yaml
import numpy as np
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from helper import analyze_ela

# ─── Config & Model Load ────────────────────────────────────────────────────
st.set_page_config(page_title="VeriCheck AI", page_icon="📄", layout="wide")
@st.cache_resource
def load_model(path="best.pt"):
    return YOLO(path)
model = load_model()

# ─── Sidebar Controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("VeriCheck AI")
    st.markdown("**Tamper Detection & On‐The‐Fly Training**")
    threshold = st.slider("ELA sensitivity", 0, 255, 50)
    std_low   = st.slider("Low‐risk cutoff", 0.0, 100.0, 10.0)
    std_high  = st.slider("High‐risk cutoff", 0.0, 200.0, 35.0)
    st.markdown("---")
    st.markdown("## Retrain Model")
    retrain_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
    if st.button("Retrain Model"):
        # 1) Generate data.yaml for new_annotations
        data_cfg = {
            'path': os.getcwd(),
            'train': 'new_annotations/images',
            'val':   'new_annotations/images',
            'nc':    1,
            'names': ['tampered']
        }
        with open('new_annotations/data.yaml','w') as f:
            yaml.dump(data_cfg, f)
        # 2) Launch training
        st.info("🚀 Starting training...")
        results = model.train(
            data='new_annotations/data.yaml',
            epochs=int(retrain_epochs),
            imgsz=640,
            project='retrain_runs',
            name=f'model_{int(np.random.rand()*1e6)}',
            exist_ok=True
        )
        st.success("✅ Retraining complete!")
        # reload best weights
        best = os.path.join(results.path, 'weights', 'best.pt')
        model = load_model(best)

# ─── App Header ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#003366;'>VeriCheck AI Platform</h1><hr>", unsafe_allow_html=True)

# ─── Upload & Predict ────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload JPG/PNG invoice or photo", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    # ELA
    ela_img, hl_img, std, regs = analyze_ela(img, threshold=threshold, quality=60)
    st.markdown(f"**ELA Std Dev:** {std:.2f} **Regions:** {regs}")
    if std>std_high:   st.error("⚠️ High Risk")
    elif std<std_low:  st.success("✅ Low Risk")
    else:              st.warning("🔍 Uncertain")

    st.subheader("ELA Image");      st.image(ela_img, use_container_width=True)
    st.subheader("Highlight Map");  st.image(hl_img, use_container_width=True)

    # YOLO inference
    st.subheader("AI‐Detected Tamper Boxes")
    res = model(np.array(img))
    box_img = img.copy(); draw = ImageDraw.Draw(box_img)
    for r in res:
        for b in r.boxes:
            x1,y1,x2,y2 = map(int,b.xyxy[0].tolist())
            if b.conf[0]>0.3:
                draw.rectangle([x1,y1,x2,y2],outline="green",width=2)
    st.image(box_img, use_container_width=True)

    # Feedback & Correction
    st.subheader("Was the AI correct?")
    fb = st.radio("", ["Yes","No"], key="fb")
    if fb=="No":
        st.markdown("Draw the *true* tampered regions:")
        cnv = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=img,
            update_streamlit=True,
            height=img.height,
            width=img.width,
            drawing_mode="rect",
            key="canvas",
        )
        if st.button("Save Correction"):
            shapes = cnv.json_data["objects"]
            os.makedirs("new_annotations/images", exist_ok=True)
            os.makedirs("new_annotations/labels", exist_ok=True)
            fn = uploaded.name
            img.save(f"new_annotations/images/{fn}")
            lbl = os.path.splitext(fn)[0] + ".txt"
            with open(f"new_annotations/labels/{lbl}","w") as f:
                for o in shapes:
                    l,t = o["left"], o["top"]
                    w,h = o["width"], o["height"]
                    x_c, y_c = (l+w/2)/img.width, (t+h/2)/img.height
                    w_n, h_n = w/img.width, h/img.height
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
            st.success("🎉 Correction saved for next training!")

else:
    st.info("Upload an image to begin.")

# ─── End of File ─────────────────────────────────────────────────────────────
