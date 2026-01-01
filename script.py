import numpy as np
import streamlit as st
from ultralytics import YOLO
import copy
import cv2

# ----------------------------
# 1. Configuration & Model Loading
# ----------------------------
MODEL_PATH = "./best.pt"
CLASS_NAMES = ["afia", "nakheel", "noor", "shams"]

st.set_page_config(page_title="YOLO Detection Pro", layout="wide")

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

model = load_yolo_model(MODEL_PATH)

# ----------------------------
# 2. Processing Functions
# ----------------------------

def analyze_image(image_bytes):
    # Convert bytes to numpy array (OpenCV format)
    file_bytes = np.frombuffer(image_bytes.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # YOLO handles BGR arrays correctly
    results = model.predict(
        source=img_bgr,
        conf=0.25,
        imgsz=640,
        save=False
    )
    return results

def get_class_counts(results):
    counts = {name: 0 for name in CLASS_NAMES}
    if not results or results[0].boxes is None:
        return counts

    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    for cid in class_ids:
        if cid < len(CLASS_NAMES):
            counts[CLASS_NAMES[cid]] += 1
    return counts

def plot_filtered_results(results, selected_class):
    # Deep copy prevents permanent filtering of session data
    r = copy.deepcopy(results[0])

    if selected_class != "all":
        class_id = CLASS_NAMES.index(selected_class)
        mask = r.boxes.cls.cpu().numpy() == class_id
        
        if not mask.any():
            # If nothing found, return the original BGR image converted to RGB
            return cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB)
        
        r.boxes = r.boxes[mask]

    # r.plot() ALWAYS returns a BGR image
    res_bgr = r.plot()
    
    # Standard conversion for Streamlit display
    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    return res_rgb

# ----------------------------
# 3. Main App Layout
# ----------------------------
st.title("YOLO Object Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.yolo_results = None
        st.session_state.selected_class = "all"

    col_left, col_right = st.columns(2)

    with col_left:
        # Load for preview using OpenCV
        uploaded_file.seek(0)
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        img_preview = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Streamlit needs RGB for the preview
        st.image(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB), caption="Source Image", use_container_width=True)
        
        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                # Pass the file object to be read as BGR
                uploaded_file.seek(0)
                st.session_state.yolo_results = analyze_image(uploaded_file)

    with col_right:
        if st.session_state.yolo_results:
            counts = get_class_counts(st.session_state.yolo_results)
            total = sum(counts.values())
            st.subheader(f"Detections: {total}")
            
            m_cols = st.columns(len(CLASS_NAMES))
            for i, name in enumerate(CLASS_NAMES):
                m_cols[i].metric(name.capitalize(), counts[name])
            
            st.divider()
            
            st.write("Filter View:")
            f_cols = st.columns(len(CLASS_NAMES) + 1)
            if f_cols[0].button("Show All", use_container_width=True):
                st.session_state.selected_class = "all"
            
            for i, name in enumerate(CLASS_NAMES):
                if f_cols[i+1].button(name.capitalize(), use_container_width=True):
                    st.session_state.selected_class = name

            # This now handles the color conversion consistently
            final_output = plot_filtered_results(
                st.session_state.yolo_results, 
                st.session_state.selected_class
            )
            st.image(final_output, caption=f"Viewing: {st.session_state.selected_class}", use_container_width=True)
            
else:
    st.info("Please upload an image to begin.")