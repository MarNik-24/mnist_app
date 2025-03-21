import joblib
import streamlit as st
import numpy as np
import os
from PIL import Image, ImageOps  # Use PIL instead of OpenCV
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
    """Process the drawn digit to match MNIST format (without OpenCV)."""
    img = Image.fromarray(img.astype(np.uint8))  # Convert to PIL image
    img = img.convert("L")  # Convert to grayscale
    ##img = ImageOps.invert(img)  # Invert colors (white digit on black)
    
    # Resize to 28x28 while maintaining aspect ratio and centering
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to NumPy array and normalize
    img = np.array(img).astype(np.float32) / 255.0  

    # Ensure it's in the shape expected by the model (1, 784)
    img = img.reshape(1, -1)

    return img

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

