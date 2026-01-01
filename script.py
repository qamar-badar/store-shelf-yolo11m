import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# --- Load YOLO model ---
model_path = "./best.pt"
model = YOLO(model_path)

# --- Function to resize image ---
def resize_image(image, max_size=(640, 640)):
    img = Image.open(image)
    img.thumbnail(max_size)  # keeps aspect ratio
    return img

# --- Run YOLO inference ---
def analyze_image(image):
    img = Image.open(st.session_state.uploaded_image).convert("RGB")
    img_array = np.array(img)

    results = model.predict(
        source=img_array,
        conf=0.25,
        save=False
    )
    return results

# --- Initialize session state ---
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "evaluating_image" not in st.session_state:
    st.session_state.evaluating_image = False

if "yolo_results" not in st.session_state:
    st.session_state.yolo_results = None

# --- Center layout ---
col1, col2, col3 = st.columns([1, 5, 1])

with col2:
    st.header("YOLO Object Detection App")
    st.caption("Upload images and get count of items using our YOLO model")

    # --- File uploader ---
    if not st.session_state.image_uploaded:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file:
            st.session_state.uploaded_image = uploaded_file
            st.session_state.image_uploaded = True
            st.session_state.evaluating_image = False
            st.session_state.yolo_results = None
            st.rerun()

    else:
        # --- Display uploaded image ---
        resized_img = resize_image(st.session_state.uploaded_image, max_size=(640, 640))
        st.image(resized_img, caption="Uploaded Image")

        # --- Buttons ---
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("Upload another image"):
                st.session_state.image_uploaded = False
                st.session_state.uploaded_image = None
                st.session_state.evaluating_image = False
                st.session_state.yolo_results = None
                st.rerun()

        with col_btn2:
            if st.button("Analyze image") and not st.session_state.evaluating_image:
                st.session_state.evaluating_image = True
                # Run YOLO inference
                st.session_state.yolo_results = analyze_image(st.session_state.uploaded_image)
                st.session_state.evaluating_image = False

        # --- Display YOLO results ---
        if st.session_state.yolo_results is not None:
            output_img = st.session_state.yolo_results[0].plot()  # returns RGB array with boxes
            st.image(output_img, caption="YOLO Output")
