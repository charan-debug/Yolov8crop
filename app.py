import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import io

# --- Configuration ---
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.25

# --- Page Setup ---
st.set_page_config(
    page_title="YOLOv8 Crop Classification",
    page_icon="🪴",
    layout="wide"
)

st.title("🌿 YOLOv8 Crop Image Classification")
st.write("Upload an image to detect and classify crops using a YOLOv8 model.")

# --- Load YOLOv8 Model ---
@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found in the app directory.")
        return None
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_yolo_model()

# --- File Upload and Prediction ---
if model:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.subheader("🔍 Detection Results:")
            with st.spinner("Running detection..."):
                results = model(image, conf=CONFIDENCE_THRESHOLD)

            if results and results[0].boxes:
                annotated = results[0].plot()
                annotated_image = Image.fromarray(annotated[..., ::-1])  # Convert BGR to RGB
                st.image(annotated_image, caption="Detection Output", use_column_width=True)

                # Show class names and confidences
                st.subheader("📊 Detected Objects:")
                detections = []
                for i, box in enumerate(results[0].boxes):
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    confidence = float(box.conf)
                    detections.append({
                        "Object": i + 1,
                        "Class": class_name,
                        "Confidence": f"{confidence:.2f}"
                    })
                st.table(detections)
            else:
                st.info("No crops detected in this image.")
        except Exception as e:
            st.error(f"Error during detection: {e}")
    else:
        st.info("Please upload an image to get started.")
else:
    st.warning("Model could not be loaded. Make sure 'best.pt' is present in the repository.")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app uses a YOLOv8 model to detect and classify crops in aerial or field images.")
st.sidebar.markdown("Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io)")
