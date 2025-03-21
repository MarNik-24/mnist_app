import joblib
import streamlit as st
import numpy as np
import os
import cv2
import zipfile
from streamlit_drawable_canvas import st_canvas
import gdown

# --- Define model filenames and download URL ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=1fTNx4nSZpO0k1bZD8I4wV3SRMw1KzvRi"
ZIP_PATH = "mnist_ensemble_final.zip"
MODEL_PATH = "mnist_ensemble_final.pkl"

# --- Function to Download & Extract Model ---
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive using gdown...")
        gdown.download(MODEL_URL, ZIP_PATH, quiet=False, fuzzy=True)
        st.info("Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall()
    return joblib.load(MODEL_PATH)

# --- Load Model ---
model = download_and_load_model()

# Initialize session states
if "drawn_digit" not in st.session_state:
    st.session_state.drawn_digit = None

# UI
st.title("Classifying Handwritten Digits with Machine Learning")
st.write("Draw a digit between 0 and 9 on the canvas below. Use the Undo or Clear buttons below the canvas to correct or erase your drawing. For the best results, make sure the digit is large, centered, and fills most of the canvas area.")

# --- Function to Process Drawn Digits ---
def preprocess_drawn_digit(img):
    """ Process the drawn digit to match MNIST format """
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(img)
    
    if coords is None or len(coords) < 10:
        return None  

    x, y, w, h = cv2.boundingRect(coords)
    if w < 5 or h < 5:
        st.warning("Bounding box too small!")
        return None

    digit = img[y:y+h, x:x+w]  
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    centered_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    centered_img[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    centered_img = centered_img.astype(np.float32) / 255.0  
    return centered_img

# --- Drawing Canvas ---
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Process and classify drawn digit
if canvas_result.image_data is not None:
    img = canvas_result.image_data[..., :3]  
    processed_img = preprocess_drawn_digit(img)

    if processed_img is not None:
        st.session_state.drawn_digit = processed_img
        drawn_img = st.session_state.drawn_digit.reshape(1, -1)
        drawn_prediction = model.predict(drawn_img)
        st.success(f"Prediction: **{drawn_prediction[0]}**")

